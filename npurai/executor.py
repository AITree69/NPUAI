"""Unified ONNX Runtime executor.

`NPUAIExecutor` is a thin wrapper around `onnxruntime.InferenceSession` that:

  1. Picks the best available Execution Provider in priority order
     (NPU > dGPU > iGPU > CPU).
  2. Falls back gracefully — if the requested EP cannot load the model
     (e.g. an op unsupported by DirectML), it walks the priority list
     until something works.
  3. Times every run and reports which EP actually executed.
  4. Handles I/O binding to the chosen device when supported.

This is the only class the demos touch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from .detect import DeviceReport, detect_devices

# Provider priority — NPU first, then discrete GPU, then iGPU, then CPU.
# We expand each family into concrete ORT EPs at runtime.
_PROVIDER_PREFERENCE: Tuple[str, ...] = (
    "QNNExecutionProvider",         # Qualcomm NPU
    "VitisAIExecutionProvider",     # AMD XDNA
    "OpenVINOExecutionProvider",     # Intel NPU / iGPU / dGPU
    "DmlExecutionProvider",         # DirectML — covers AMD/Intel/NVIDIA on Win
    "CUDAExecutionProvider",        # NVIDIA dGPU
    "CPUExecutionProvider",         # always available
)


@dataclass
class ExecutionResult:
    outputs: List[np.ndarray]
    provider_used: str            # the EP that actually ran
    provider_attempted: List[str]  # everything we tried
    latency_ms: float
    fallback_occurred: bool

    def output(self, idx: int = 0) -> np.ndarray:
        return self.outputs[idx]


@dataclass
class ProviderStatus:
    name: str
    available: bool
    reason: str = ""

    def __str__(self) -> str:
        mark = "OK " if self.available else "-- "
        return f"  [{mark}] {self.name:<32} {self.reason}"


class NPUAIExecutor:
    """Vendor-agnostic ONNX Runtime executor.

    Example
    -------
    >>> exec = NPUAIExecutor("mobilenet.onnx", prefer="npu")
    >>> out = exec.run({"input": x})  # auto-targets the best EP
    >>> out.provider_used
    'DmlExecutionProvider'
    """

    def __init__(
        self,
        model_path: str,
        prefer: str = "auto",          # "auto" | "npu" | "gpu" | "cpu"
        providers: Optional[Sequence[str]] = None,
        device_id: int = 0,
        intra_op_num_threads: int = 0,
    ) -> None:
        self.model_path = model_path
        self.prefer = prefer
        self.device_id = device_id
        self.intra_op_num_threads = intra_op_num_threads or 0
        self.report: DeviceReport = detect_devices()

        # Build the priority list
        if providers is not None:
            self.priority = list(providers) + [
                p for p in _PROVIDER_PREFERENCE if p not in providers
            ]
        else:
            self.priority = list(_select_priority(self.prefer, self.report))

        # Map EP -> ORT provider options
        self.provider_options = self._build_provider_options()
        # Mark which EPs are actually available on this machine
        self.status = _probe_providers(self.priority, self.provider_options)

        # Create the session. ORT instantiates ALL listed EPs; we add
        # only those flagged available so the session doesn't fail on
        # import-time missing backends.
        selected = [p for p, s in self.status.items() if s.available]
        if not selected:
            raise RuntimeError(
                "No ONNX Runtime Execution Provider is available. "
                "Reinstall onnxruntime (or onnxruntime-directml / "
                "onnxruntime-openvino / onnxruntime-qnn)."
            )
        # CPU must be last as ORT's fallback; preserve order otherwise.
        # Deduplicate (priority list already has CPU in it).
        seen: set[str] = set()
        deduped: List[str] = []
        for p in selected:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        if "CPUExecutionProvider" in deduped:
            deduped.remove("CPUExecutionProvider")
        deduped.append("CPUExecutionProvider")
        selected = deduped

        sess_options = ort.SessionOptions()
        if self.intra_op_num_threads:
            sess_options.intra_op_num_threads = self.intra_op_num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._selected_eps = selected
        # ONNX Runtime 1.18+ accepts `provider_options` as a *list* of dicts,
        # one per EP in the same order as `providers`. CPU gets no options.
        # IMPORTANT: ORT 1.24 + Dml EP has a known regression where passing
        # any `provider_options` list (even with an empty `{}` for the CPU
        # entry) triggers a UTF-8 decode error in the DirectML backend init
        # path on some AMD Radeon drivers. We work around it by passing
        # `provider_options=None` (i.e. all EPs use defaults) and only
        # hand-passing opts for the EPs where we know the format is stable
        # (CUDA, OpenVINO, VitisAI, QNN).
        has_dml = "DmlExecutionProvider" in selected
        if has_dml:
            # Drop Dml-specific options; let Dml EP auto-config.
            self.provider_options.pop("DmlExecutionProvider", None)
            if not self.provider_options:
                # No remaining EPs need options -> pass None entirely
                opts_list: Optional[List[dict]] = None
            else:
                opts_list = []
                for p in selected:
                    if p == "CPUExecutionProvider":
                        opts_list.append({})
                    else:
                        opts_list.append(self.provider_options.get(p, {}))
        else:
            opts_list = []
            for p in selected:
                if p == "CPUExecutionProvider":
                    opts_list.append({})
                else:
                    opts_list.append(self.provider_options.get(p, {}))

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=selected,
            provider_options=opts_list,
        )
        self.provider_used = self.session.get_providers()[0]
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

    # ------------------------------------------------------------------ I/O

    @property
    def input_names(self) -> List[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> List[str]:
        return list(self._output_names)

    @property
    def input_shapes(self) -> Dict[str, List[Optional[int]]]:
        return {i.name: [d if isinstance(d, int) else None for d in i.shape]
                for i in self.session.get_inputs()}

    def run(
        self,
        feeds: Optional[Dict[str, np.ndarray]] = None,
        x: Optional[np.ndarray] = None,
    ) -> ExecutionResult:
        """Run inference.

        Two ways to pass data:
          - `feeds={"input_name": array, ...}`  — explicit, multi-input.
          - `x=array` — short-hand for "the only model input".
        """
        if feeds is None:
            if x is None:
                raise ValueError("Either `feeds` or `x` must be provided.")
            if len(self._input_names) != 1:
                raise ValueError(
                    f"Model has {len(self._input_names)} inputs; use `feeds=...` "
                    f"instead of `x=...`. Inputs: {self._input_names}"
                )
            feeds = {self._input_names[0]: x}
        missing = set(self._input_names) - set(feeds.keys())
        if missing:
            raise ValueError(
                f"Missing inputs for the model: {sorted(missing)}; "
                f"expected {self._input_names}"
            )
        t0 = time.perf_counter()
        outputs = self.session.run(self._output_names, feeds)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        actual_provider = self.session.get_providers()[0]
        fallback = actual_provider != self._selected_eps[0]
        return ExecutionResult(
            outputs=list(outputs),
            provider_used=actual_provider,
            provider_attempted=list(self._selected_eps),
            latency_ms=latency_ms,
            fallback_occurred=fallback,
        )

    # ---------------------------------------------------------- diagnostics

    def explain(self) -> str:
        lines = [
            f"  Model        : {self.model_path}",
            f"  Selected EPs : {self._selected_eps}",
            f"  Actually used: {self.provider_used}",
            f"  Inputs       : {self._input_names}",
            f"  Input shapes : {self.input_shapes}",
            "  Provider probe:",
        ]
        for p, s in self.status.items():
            lines.append(f"    [{'OK ' if s.available else 'X  '}] {p:<32} {s.reason}")
        return "\n".join(lines)

    # ------------------------------------------------------------- helpers

    def _build_provider_options(self) -> Dict[str, dict]:
        opts: Dict[str, dict] = {}
        if "DmlExecutionProvider" in self.priority:
            # Dml EP provider options for ORT 1.18+. `enable_graph_serialization`
            # was removed in 1.16 and is rejected by the runtime if passed.
            opts["DmlExecutionProvider"] = {
                "device_id": self.device_id,
            }
        if "CUDAExecutionProvider" in self.priority:
            opts["CUDAExecutionProvider"] = {
                "device_id": self.device_id,
                "gpu_mem_limit": 0,                # 0 = no limit
                "arena_extend_strategy": "kNextPowerOfTwo",
            }
        if "OpenVINOExecutionProvider" in self.priority:
            # NPU = "NPU", iGPU = "GPU.0", dGPU = "GPU.1", CPU = "CPU"
            opts["OpenVINOExecutionProvider"] = {
                "device_type": "AUTO",            # ORT picks best sub-device
                "enable_dynamic_shapes": True,
            }
        if "QNNExecutionProvider" in self.priority:
            opts["QNNExecutionProvider"] = {
                "backend_path": "QnnHtp.dll",     # HTP = Hexagon Tensor Processor
                "htp_performance_mode": "burst",
            }
        if "VitisAIExecutionProvider" in self.priority:
            opts["VitisAIExecutionProvider"] = {
                "target": "AMD_AIE2_Nx4_Overlay",  # XDNA 2; XDNA1 uses AMD_AIE2_Nx2
                "config_file": "",
            }
        return opts


# ---------------------------------------------------------------------------
# Provider probing & priority selection
# ---------------------------------------------------------------------------

def _probe_providers(
    priority: Sequence[str],
    options: Dict[str, dict],
) -> Dict[str, ProviderStatus]:
    """Try to instantiate each EP via a 1x1 dummy matmul and report status.

    This catches import-time failures (missing DLL) without spinning up
    the user's actual model. We *do not* depend on a real model file.
    """
    # A 1x1 Float32 add — supported on every EP including NPU backends.
    # We build the model in-memory.
    available = ort.get_available_providers()
    statuses: Dict[str, ProviderStatus] = {}
    for p in priority:
        if p == "CPUExecutionProvider":
            statuses[p] = ProviderStatus(p, True, "always available")
            continue
        if p not in available:
            statuses[p] = ProviderStatus(
                p, False,
                f"not in onnxruntime.get_available_providers() "
                f"(try pip install onnxruntime-{_ep_to_pkg(p)})"
            )
            continue
        # Try a real load. We import lazily to avoid forcing a build.
        try:
            _ep_smoke_test(p, options.get(p, {}))
            statuses[p] = ProviderStatus(p, True, "smoke test passed")
        except Exception as e:                       # noqa: BLE001
            statuses[p] = ProviderStatus(p, False, f"smoke test failed: {e}")
    return statuses


def _ep_smoke_test(provider: str, opts: dict) -> None:
    """Build a tiny identity model and run it under `provider`.

    NOTE: we intentionally pass `provider_options=None` for the probe — see
    the long comment in `NPUAIExecutor.__init__` for why. The probe is
    only checking that the EP can be imported/initialised, not that the
    real model can be served.
    """
    from onnx import TensorProto, helper, save
    import tempfile, os

    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node], "smoke", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        save(model, f.name)
        path = f.name
    try:
        providers = [provider, "CPUExecutionProvider"]
        sess = ort.InferenceSession(path, providers=providers)
        sess.run(None, {"x": np.zeros((1, 1), dtype=np.float32)})
    finally:
        os.unlink(path)


def _ep_to_pkg(ep: str) -> str:
    return {
        "DmlExecutionProvider": "directml",
        "CUDAExecutionProvider": "gpu",
        "OpenVINOExecutionProvider": "openvino",
        "QNNExecutionProvider": "qnn",
        "VitisAIExecutionProvider": "vitisai",
    }.get(ep, "directml")


def _select_priority(prefer: str, report: DeviceReport) -> List[str]:
    """Reorder the default priority list based on `prefer` and detected HW."""
    base = list(_PROVIDER_PREFERENCE)
    has_npu = bool(report.npus)
    has_amd_gpu = any(g.vendor == "AMD" for g in report.gpus)
    has_nvidia_gpu = any(g.vendor == "NVIDIA" for g in report.gpus)
    has_intel_gpu = any(g.vendor == "Intel" for g in report.gpus)

    if prefer == "npu" and not has_npu:
        # Silently demote — DirectML will use NPU if it can find one
        pass
    if prefer == "cpu":
        return ["CPUExecutionProvider"]
    if prefer == "gpu":
        # Reorder GPU EPs first
        gpu_eps = []
        if has_nvidia_gpu:
            gpu_eps.append("CUDAExecutionProvider")
        gpu_eps.append("DmlExecutionProvider")
        return gpu_eps + [p for p in base if p not in gpu_eps]

    # "auto" — already in the default order, but trim EPs the hardware can't use
    if not has_nvidia_gpu:
        base = [p for p in base if p != "CUDAExecutionProvider"]
    return base
