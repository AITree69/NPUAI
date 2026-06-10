"""Demo 2: tiny end-to-end ONNX pipeline.

This demo runs three models back-to-back through NPUAIExecutor, each on its
own provider preference, and shows how the executor's "auto" mode picks
the best EP per model.

It is NOT a full speech-recognition pipeline (we'd need the full Whisper
graph + a chunked audio loader). The point of this demo is to exercise
NPUAI across multiple model shapes and report per-model provider picks.

Models used (cached under ./models):
  - ResNet50  (CV, 25 MB)        — vision CNN
  - MiniLM-L6 (NLP, 23 MB)       — small transformer
  - Silero-VAD (audio, 2 MB)     — small LSTM

Run:
    python -m demos.02_pipeline.run
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

# Curated small ONNX models: each exercises a different shape & op mix.
# `input_shape` is a tuple used to fabricate a random feed; the actual input
# *name* is read off the model by the executor (no hard-coded name here).
PIPELINE = [
    {
        "name": "mobilenetv2",
        "url": (
            "https://github.com/onnx/models/raw/main/validated/vision/classification/"
            "mobilenet/model/mobilenetv2-12.onnx"
        ),
        "input_shape": (1, 3, 224, 224),
        "label": "vision CNN",
    },
    {
        "name": "squeezenet1.0",
        "url": (
            "https://github.com/onnx/models/raw/main/validated/vision/classification/"
            "squeezenet/model/squeezenet1.0-12.onnx"
        ),
        "input_shape": (1, 3, 224, 224),
        "label": "lightweight vision",
    },
    {
        "name": "mnist-8",
        "url": (
            "https://github.com/onnx/models/raw/main/validated/vision/classification/"
            "mnist/model/mnist-8.onnx"
        ),
        "input_shape": (1, 1, 28, 28),
        "label": "tiny MLP",
    },
]


def _fetch(name: str, url: str) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / f"{name}.onnx"
    if dest.exists():
        return dest
    print(f"[fetch] {name} from {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _make_input(exec_, shape) -> dict:
    """Build a random feed dict keyed on the model's real input name."""
    if len(exec_.input_names) != 1:
        raise RuntimeError(
            f"Demo expects a single-input model; got {exec_.input_names}"
        )
    name = exec_.input_names[0]
    return {name: np.random.rand(*shape).astype(np.float32)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefer", default="auto", choices=["auto", "npu", "gpu", "cpu"])
    p.add_argument("--iter", type=int, default=20)
    args = p.parse_args()

    from npurai import NPUAIExecutor, detect_devices

    print(detect_devices().summary())
    print()

    # Fetch all models first so timing is clean
    paths = []
    for m in PIPELINE:
        paths.append((_fetch(m["name"], m["url"]), m))

    # Run each model through NPUAIExecutor
    print("=" * 60)
    print("Pipeline run")
    print("=" * 60)
    grand_total = 0.0
    for path, meta in paths:
        exec_ = NPUAIExecutor(str(path), prefer=args.prefer)
        feeds = _make_input(exec_, meta["input_shape"])
        # Warm-up
        for _ in range(5):
            exec_.run(feeds=feeds)
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(args.iter):
            r = exec_.run(feeds=feeds)
        avg_ms = (time.perf_counter() - t0) * 1000.0 / args.iter
        grand_total += avg_ms
        print(
            f"  {meta['name']:<20s} ({meta['label']:<20s})  "
            f"{r.provider_used:<28s}  {avg_ms:7.2f} ms/iter"
        )

    print(f"\n  Total wall time per pipeline pass: {grand_total:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
