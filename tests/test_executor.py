"""Tests for npurai.executor — pure logic, no real model needed."""
from __future__ import annotations

import numpy as np
import pytest

from npurai.executor import _select_priority, _PROVIDER_PREFERENCE
from npurai.detect import CPUInfo, DeviceReport, GPUInfo, NPUInfo


def _report(npus=0, gpus=()):
    return DeviceReport(
        cpu=CPUInfo(name="test cpu", physical_cores=4, logical_cores=8, arch="AMD64"),
        gpus=list(gpus),
        npus=[NPUInfo(name=f"npu{i}", vendor="AMD", peak_tops=50, driver="",
                      detection_method="test")
              for i in range(npus)],
        os="Windows 11",
        python="3.12",
    )


def test_select_priority_npu_first_when_present() -> None:
    p = _select_priority("auto", _report(npus=1))
    # All vendor NPU EPs are still listed in the global order
    assert p[0] in {"QNNExecutionProvider", "VitisAIExecutionProvider",
                    "OpenVINOExecutionProvider"}


def test_select_priority_cpu_only() -> None:
    p = _select_priority("cpu", _report())
    assert p == ["CPUExecutionProvider"]


def test_select_priority_gpu_promotes_cuda_when_nvidia_present() -> None:
    gpus = [GPUInfo(name="RTX 5070", vendor="NVIDIA", vram_mb=8192,
                    driver_version="1.0", adapter_index=0)]
    p = _select_priority("gpu", _report(gpus=gpus))
    assert p[0] == "CUDAExecutionProvider"


def test_select_priority_drops_cuda_when_no_nvidia() -> None:
    gpus = [GPUInfo(name="Radeon 890M", vendor="AMD", vram_mb=512,
                    driver_version="1.0", adapter_index=0)]
    p = _select_priority("auto", _report(gpus=gpus))
    assert "CUDAExecutionProvider" not in p


def test_preference_list_covers_all_major_vendors() -> None:
    """Sanity check: the static provider list must cover AMD/Intel/Qualcomm
    /NVIDIA dGPUs and a CPU fallback. If a vendor is added, this test breaks
    on purpose so the priority list is reviewed."""
    expected = {
        "QNNExecutionProvider",            # Qualcomm
        "VitisAIExecutionProvider",        # AMD XDNA
        "OpenVINOExecutionProvider",       # Intel
        "DmlExecutionProvider",            # cross-vendor GPU
        "CUDAExecutionProvider",           # NVIDIA dGPU
        "CPUExecutionProvider",            # fallback
    }
    assert expected.issubset(set(_PROVIDER_PREFERENCE))
