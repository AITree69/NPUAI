"""Hardware detection for CPU / GPU / NPU on Windows.

The output of this module is the source of truth for the executor
to pick the right Execution Provider. We use only built-in Windows
APIs (WMI, DirectX) and fall back to dxdiag / registry when needed.

References
----------
- Win32_VideoController          (WMI): every GPU the OS knows about
- Win32_Processor                (WMI): CPU brand + cores
- dxgi.dll DXGI enum adapters    (DirectX): vendor-id 0x1002 AMD, 0x8086 Intel,
                                    0x10DE NVIDIA, 0x1414 Qualcomm
- onnxruntime.get_device()       (ORT): which EPs the runtime can see
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

# DXGI vendor IDs (PCI vendor table subset)
_VENDOR_NAMES = {
    0x1002: "AMD",
    0x1022: "AMD",          # AMD/ATI
    0x8086: "Intel",
    0x10DE: "NVIDIA",
    0x1414: "Qualcomm",     # rare on Windows x64, common on ARM
    0x13B5: "ARM",          # used by some Qualcomm chips
}

# Rough NPU fingerprints. These are NOT authoritative — there's no public
# NPU inventory API — but they cover the chips shipping in 2023-2025 boxes.
_NPU_FINGERPRINTS = (
    "ryzen ai",     # AMD XDNA / XDNA 2
    "hexagon",      # Qualcomm
    "npu 4",        # Intel Meteor Lake (informal)
    "npu 3720",     # Intel Lunar Lake
    "neural",       # generic
    "ai boost",     # Intel marketing
    "aip",          # MediaTek/AMD
)


@dataclass
class CPUInfo:
    name: str
    physical_cores: int
    logical_cores: int
    arch: str

    def __str__(self) -> str:
        return f"{self.name} ({self.physical_cores}c/{self.logical_cores}t, {self.arch})"


@dataclass
class GPUInfo:
    name: str
    vendor: str
    vram_mb: int
    driver_version: str
    adapter_index: int
    is_integrated: bool = False  # best-effort

    def __str__(self) -> str:
        # Some driver strings already start with the vendor ("NVIDIA GeForce...",
        # "AMD Radeon...") so we strip a duplicated prefix.
        n = self.name
        for prefix in (f"{self.vendor} ", self.vendor):
            if n.startswith(prefix):
                n = n[len(prefix):]
                break
        return f"{self.vendor} {n} ({self.vram_mb} MB, drv {self.driver_version})"


@dataclass
class NPUInfo:
    name: str
    vendor: str
    peak_tops: Optional[int]      # best-effort estimate, not measured
    driver: str
    detection_method: str         # "wmi", "dxgi", "registry", "inferred"

    def __str__(self) -> str:
        tops = f"~{self.peak_tops} TOPS" if self.peak_tops else "TOPS unknown"
        return f"{self.vendor} NPU: {self.name} [{tops}] ({self.detection_method})"


@dataclass
class DeviceReport:
    cpu: CPUInfo
    gpus: List[GPUInfo] = field(default_factory=list)
    npus: List[NPUInfo] = field(default_factory=list)
    os: str = ""
    python: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 64,
            f"  NPUAI hardware report  -  {self.os}",
            f"  Python {self.python}",
            "=" * 64,
            f"  CPU  : {self.cpu}",
        ]
        if self.gpus:
            for g in self.gpus:
                lines.append(f"  GPU  : {g}")
        else:
            lines.append("  GPU  : (none detected)")
        if self.npus:
            for n in self.npus:
                lines.append(f"  NPU  : {n}")
        else:
            lines.append("  NPU  : (none detected — NPUAI will fall back to GPU/CPU)")
        lines.append("=" * 64)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# WMI helpers
# ---------------------------------------------------------------------------

def _wmi(class_name: str) -> List[dict]:
    """List instances of a WMI class and return them as a list of dicts.

    We don't pass the WQL query through `powershell -Command` because the
    surrounding `'-Query @\"...\"@'` quoting tends to be eaten by intermediate
    shells. Get-CimInstance on the class name is more portable and returns
    the same data.
    """
    # Use a here-string via stdin so the quote handling is shell-agnostic.
    ps = (
        "$OutputEncoding = [System.Text.Encoding]::UTF8\n"
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
        "$ProgressPreference = 'SilentlyContinue'\n"
        f"Get-CimInstance {class_name} | "
        "ConvertTo-Json -Depth 3 -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, timeout=15,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return []
        text = None
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                text = out.stdout.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = out.stdout.decode("utf-8", errors="replace")
        import json
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, ValueError):
        return []


def _detect_cpu() -> CPUInfo:
    rows = _wmi("Win32_Processor")
    if rows:
        r = rows[0]
        return CPUInfo(
            name=str(r.get("Name", "Unknown CPU")).strip(),
            physical_cores=int(r.get("NumberOfCores", 0) or 0),
            logical_cores=int(r.get("NumberOfLogicalProcessors", 0) or 0),
            arch=platform.machine(),
        )
    return CPUInfo(name="Unknown CPU", physical_cores=0, logical_cores=0, arch=platform.machine())


def _detect_gpus() -> List[GPUInfo]:
    rows = _wmi("Win32_VideoController")
    gpus: List[GPUInfo] = []
    for r in rows:
        name = str(r.get("Name", "")).strip()
        if not name:
            continue
        compat = str(r.get("AdapterCompatibility", "")).strip()
        vram_bytes = int(r.get("AdapterRAM", 0) or 0)
        # Win32_VideoController reports AdapterRAM in bytes for some cards and
        # 0 for shared-memory iGPUs (it depends on driver).
        vram_mb = vram_bytes // (1024 * 1024) if vram_bytes > 0 else 0
        # Heuristic: iGPU shares system memory -> vram_mb often reported as 0
        is_igpu = vram_mb == 0 or any(
            kw in name.lower() for kw in ("radeon graphics", "iris", "uhd", "vega")
        )
        # Map "AdapterCompatibility" to vendor; fall back to name matching
        vendor = _compat_to_vendor(compat, name)
        gpus.append(
            GPUInfo(
                name=name,
                vendor=vendor,
                vram_mb=vram_mb,
                driver_version=str(r.get("DriverVersion", "")).strip(),
                adapter_index=int(r.get("Index", 0) or 0),
                is_integrated=is_igpu,
            )
        )
    return gpus


def _compat_to_vendor(compat: str, name: str) -> str:
    s = (compat + " " + name).lower()
    if "amd" in s or "ati" in s or "radeon" in s or "advanced micro" in s:
        return "AMD"
    if "nvidia" in s or "geforce" in s or "quadro" in s or "rtx" in s:
        return "NVIDIA"
    if "intel" in s or "iris" in s or "uhd" in s:
        return "Intel"
    if "qualcomm" in s or "hexagon" in s or "adreno" in s:
        return "Qualcomm"
    return "Unknown"


# ---------------------------------------------------------------------------
# NPU detection — the hard part
# ---------------------------------------------------------------------------

def _detect_npus(gpus: List[GPUInfo], cpu: CPUInfo) -> List[NPUInfo]:
    """Best-effort NPU detection.

    Strategy:
      1. Check Windows AI / DirectML device enumeration (most reliable when
         DirectML is on the path; we do that later in the executor).
      2. Look for known vendor registry keys / DLLs.
      3. Fingerprint the CPU name itself — AMD Ryzen AI and Intel Core Ultra
         both carry the NPU in the same package.
    """
    npus: List[NPUInfo] = []

    # (2) DLL / driver presence checks
    amd_npu = _try_find_amd_npu()
    if amd_npu:
        npus.append(amd_npu)
    intel_npu = _try_find_intel_npu()
    if intel_npu:
        npus.append(intel_npu)
    qualcomm_npu = _try_find_qualcomm_npu()
    if qualcomm_npu:
        npus.append(qualcomm_npu)

    # (3) Fingerprint the CPU name
    cpu_lower = cpu.name.lower()
    if not npus:
        if "ryzen ai" in cpu_lower:
            npus.append(NPUInfo(
                name=("XDNA 2" if "hx 3" in cpu_lower or "ai 9" in cpu_lower
                      else "XDNA (gen 1)"),
                vendor="AMD",
                peak_tops=50 if "ai 9" in cpu_lower or "hx 3" in cpu_lower else 16,
                driver="(inferred from CPU name; install Ryzen AI SDK for native EP)",
                detection_method="cpu-fingerprint",
            ))
        elif "core ultra" in cpu_lower or "meteor lake" in cpu_lower or "lunar lake" in cpu_lower:
            npus.append(NPUInfo(
                name="Intel NPU (Core Ultra)",
                vendor="Intel",
                peak_tops=48 if "lunar" in cpu_lower else 11,
                driver="(install OpenVINO for native EP; DirectML will use it as fallback)",
                detection_method="cpu-fingerprint",
            ))

    return npus


def _safe_json_loads(out: subprocess.CompletedProcess) -> list:
    """Decode PowerShell stdout with GBK/UTF-8 tolerance and parse JSON."""
    if out.returncode != 0 or not out.stdout.strip():
        return []
    text = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            text = out.stdout.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = out.stdout.decode("utf-8", errors="replace")
    try:
        import json
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    return data


def _try_find_amd_npu() -> Optional[NPUInfo]:
    # AMD NPU drivers use explicit names; do NOT match the bare "IPU" prefix
    # because Intel platforms also ship an IPU MCDM service that we'd
    # otherwise mis-attribute. Real AMD signatures: AMDXE, XDNA, AIE, RyzenAI.
    ps = (
        "Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Services' "
        "-ErrorAction SilentlyContinue "
        "| Where-Object { $_.Name -match 'AMDXE|XDNA|AIE|RyzenAI' } "
        "| Select-Object -ExpandProperty Name "
        "| ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True, timeout=10,
    )
    names = _safe_json_loads(out)
    if names:
        return NPUInfo(
            name="AMD XDNA NPU",
            vendor="AMD",
            peak_tops=None,
            driver=", ".join(n.split("\\")[-1] for n in names[:3]),
            detection_method="windows-registry",
        )
    return None


def _try_find_intel_npu() -> Optional[NPUInfo]:
    # Use a precise Intel-only fingerprint; "IPU" is too greedy and would
    # match the AMD IPU device that ships in some Ryzen AI platforms.
    ps = (
        "Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Services' "
        "-ErrorAction SilentlyContinue "
        "| Where-Object { $_.Name -match 'IntelNPU|IntelAI|IntelMovidius' } "
        "| Select-Object -ExpandProperty Name "
        "| ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True, timeout=10,
    )
    names = _safe_json_loads(out)
    if names:
        return NPUInfo(
            name="Intel NPU (Meteor/Lunar Lake)",
            vendor="Intel",
            peak_tops=None,
            driver=", ".join(n.split("\\")[-1] for n in names[:3]),
            detection_method="windows-registry",
        )
    return None


def _try_find_qualcomm_npu() -> Optional[NPUInfo]:
    # Qualcomm NPU on Windows ARM: look for Hexagon driver
    ps = (
        "Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Services' "
        "-ErrorAction SilentlyContinue "
        "| Where-Object { $_.Name -match 'qcn|hexagon|QNN|Adreno' } "
        "| Select-Object -ExpandProperty Name "
        "| ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True, timeout=10,
    )
    names = _safe_json_loads(out)
    if names:
        return NPUInfo(
            name="Qualcomm Hexagon NPU",
            vendor="Qualcomm",
            peak_tops=None,
            driver=", ".join(n.split("\\")[-1] for n in names[:3]),
            detection_method="windows-registry",
        )
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_devices() -> DeviceReport:
    """Detect CPU / GPUs / NPUs on the local machine."""
    cpu = _detect_cpu()
    gpus = _detect_gpus()
    npus = _detect_npus(gpus, cpu)
    return DeviceReport(
        cpu=cpu,
        gpus=gpus,
        npus=npus,
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        python=platform.python_version(),
    )


if __name__ == "__main__":
    # `python -m npurai.detect`
    report = detect_devices()
    print(report.summary())
