"""Tests for npurai.detect — dataclasses & report rendering."""
from __future__ import annotations

from npurai.detect import (
    CPUInfo, DeviceReport, GPUInfo, NPUInfo,
)


def test_report_summary_includes_cpu() -> None:
    r = DeviceReport(
        cpu=CPUInfo(name="Ryzen 9", physical_cores=12, logical_cores=24, arch="AMD64"),
    )
    s = r.summary()
    assert "Ryzen 9" in s
    assert "12c/24t" in s
    assert "AMD64" in s
    assert "NPU  : (none detected" in s


def test_report_summary_includes_gpu_with_no_double_vendor() -> None:
    """Some driver strings already start with the vendor name; make sure
    we don't print 'NVIDIA NVIDIA GeForce'."""
    r = DeviceReport(
        cpu=CPUInfo(name="i7", physical_cores=8, logical_cores=16, arch="AMD64"),
        gpus=[GPUInfo(name="NVIDIA GeForce RTX 5070", vendor="NVIDIA",
                      vram_mb=4095, driver_version="1.0", adapter_index=0)],
    )
    s = r.summary()
    assert "NVIDIA NVIDIA" not in s
    assert "NVIDIA GeForce RTX 5070" in s


def test_report_summary_includes_npu() -> None:
    r = DeviceReport(
        cpu=CPUInfo(name="Ryzen AI", physical_cores=12, logical_cores=24, arch="AMD64"),
        npus=[NPUInfo(name="XDNA 2", vendor="AMD", peak_tops=50, driver="",
                      detection_method="cpu-fingerprint")],
    )
    s = r.summary()
    assert "XDNA 2" in s
    assert "50 TOPS" in s


def test_report_summary_handles_no_gpus() -> None:
    r = DeviceReport(
        cpu=CPUInfo(name="i5", physical_cores=4, logical_cores=8, arch="AMD64"),
    )
    s = r.summary()
    assert "GPU  : (none detected)" in s
