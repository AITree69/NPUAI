"""Quick smoke test: load a small ONNX model, run it on DirectML, then CPU.

Uses MobileNet v2 (FP32, ~13 MB) from the onnx model zoo — small enough to
download in seconds, yet covers the same operator set the demos need.

Strategy: we try every EP the user has installed in priority order and
report which one actually executes the model. This is the contract the
rest of the package relies on — the executor picks the first working EP.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "mobilenetv2-12.onnx"
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "mobilenet/model/mobilenetv2-12.onnx"
)


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"Downloading {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def try_each_ep(path: str, x: np.ndarray) -> None:
    """Try every available EP individually, with CPU as the safety net."""
    available = ort.get_available_providers()
    print(f"Available EPs: {available}")

    for ep in available:
        try:
            print(f"\n--- Trying {ep} ---")
            t0 = time.perf_counter()
            sess = ort.InferenceSession(
                path, providers=[ep, "CPUExecutionProvider"],
            )
            used = sess.get_providers()[0]
            for _ in range(3):
                sess.run(None, {sess.get_inputs()[0].name: x})
            t_setup = (time.perf_counter() - t0) * 1000.0
            n = 20
            t1 = time.perf_counter()
            for _ in range(n):
                sess.run(None, {sess.get_inputs()[0].name: x})
            t_total = (time.perf_counter() - t1) * 1000.0
            print(f"  selected: {used}, "
                  f"setup {t_setup:.0f} ms, "
                  f"avg {t_total / n:.2f} ms/iter")
        except Exception as e:                       # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")


def main() -> int:
    path = ensure_model()
    print(f"Model: {path}  ({path.stat().st_size // 1024 // 1024} MB)")

    from npurai import NPUAIExecutor

    # 1. Per-EP probe — see exactly which EPs work
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    try_each_ep(str(path), x)

    # 2. The actual NPUAIExecutor — what the demos see
    print("\n" + "=" * 60)
    print("NPUAIExecutor (priority-scheduled)")
    print("=" * 60)
    exec_ = NPUAIExecutor(str(path), prefer="auto")
    print(exec_.explain())
    # Warm-up: more iterations amortize ORT's first-call graph compile cost.
    for _ in range(10):
        exec_.run(x=x)
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        r = exec_.run(x=x)
    total_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\n  Ran {n} iters on {r.provider_used}: "
          f"{total_ms / n:.2f} ms/iter, "
          f"fallback={r.fallback_occurred}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
