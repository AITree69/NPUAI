# NPUAI: vendor-agnostic NPU/CPU/GPU inference for Windows.

__all__ = ["detect_devices", "DeviceReport", "NPUAIExecutor", "ExecutionResult"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy import: keeps `import npurai` cheap (no ORT, no WMI calls) and
    # also avoids runpy's "submodule already in sys.modules" warning when
    # the user does `python -m npurai.detect`.
    if name in ("detect_devices", "DeviceReport"):
        from . import detect as _detect
        return getattr(_detect, name)
    if name in ("NPUAIExecutor", "ExecutionResult"):
        from . import executor as _exec
        return getattr(_exec, name)
    raise AttributeError(f"module 'npurai' has no attribute {name!r}")
