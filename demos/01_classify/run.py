"""Demo 1: image classification on a single image.

Run:
    python -m demos.01_classify.run
    python -m demos.01_classify.run --image path/to/dog.jpg
    python -m demos.01_classify.run --prefer cpu

What it does:
    1. Loads MobileNet v2 (ONNX, FP32, 1000 classes) via NPUAIExecutor.
    2. Detects hardware and reports which EP was selected.
    3. Loads an image, runs preprocessing (resize/center-crop/normalize),
       and prints the top-5 ImageNet labels with confidence.
    4. Prints a one-line "where did this run?" summary.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "mobilenetv2-12.onnx"
LABELS_PATH = REPO_ROOT / "demos" / "01_classify" / "imagenet_classes.txt"
SAMPLE_IMAGE_PATH = REPO_ROOT / "demos" / "01_classify" / "sample.jpg"
# Picsum is a deterministic random-photo service; we fall back to a locally
# synthesised gradient image if the network is blocked.
SAMPLE_IMAGE_URL = "https://picsum.photos/seed/cat/640/480"

# ImageNet preprocessing constants (Pytorch convention)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE = 256
CROP = 224


def _ensure_artefacts() -> None:
    REPO_ROOT.joinpath("models").mkdir(exist_ok=True)
    if not MODEL_PATH.exists():
        url = (
            "https://github.com/onnx/models/raw/main/validated/vision/classification/"
            "mobilenet/model/mobilenetv2-12.onnx"
        )
        print(f"[fetch] {MODEL_PATH.name} from {url}")
        urllib.request.urlretrieve(url, MODEL_PATH)
    if not LABELS_PATH.exists():
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        print(f"[fetch] imagenet_classes.txt from {url}")
        urllib.request.urlretrieve(url, LABELS_PATH)
    if not SAMPLE_IMAGE_PATH.exists():
        try:
            print(f"[fetch] sample.jpg from {SAMPLE_IMAGE_URL}")
            urllib.request.urlretrieve(SAMPLE_IMAGE_URL, SAMPLE_IMAGE_PATH)
        except Exception as e:                       # noqa: BLE001
            print(f"[fallback] network unavailable ({e}); synthesising gradient image")
            arr = _synth_gradient_image(640, 480)
            Image.fromarray(arr, "RGB").save(SAMPLE_IMAGE_PATH, "JPEG", quality=85)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _synth_gradient_image(w: int, h: int) -> np.ndarray:
    """Generate a smooth RGB gradient. Used when the demo cannot reach
    the internet; classification will still run end-to-end (the gradient
    will map to a real ImageNet class with low confidence)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (x / w * 255).astype(np.uint8)
    g = (y / h * 255).astype(np.uint8)
    b = ((x + y) / (w + h) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _preprocess(img_path: Path) -> np.ndarray:
    """Resize, center-crop, normalize, NCHW float32."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = RESIZE / min(w, h)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    img = img.resize(new_size, Image.BILINEAR)
    nw, nh = img.size
    left = (nw - CROP) // 2
    top = (nh - CROP) // 2
    img = img.crop((left, top, left + CROP, top + CROP))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)             # HWC -> CHW
    arr = np.expand_dims(arr, 0)             # NCHW
    return np.ascontiguousarray(arr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=SAMPLE_IMAGE_PATH)
    p.add_argument("--prefer", default="auto", choices=["auto", "npu", "gpu", "cpu"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--iter", type=int, default=10, help="benchmark iterations")
    args = p.parse_args()

    _ensure_artefacts()

    from npurai import NPUAIExecutor, detect_devices

    print(detect_devices().summary())
    print()

    exec_ = NPUAIExecutor(str(MODEL_PATH), prefer=args.prefer)
    print(exec_.explain())
    print()

    x = _preprocess(args.image)
    print(f"[input] image={args.image.name}  shape={x.shape}  dtype={x.dtype}")

    # Warm-up
    for _ in range(5):
        exec_.run(x=x)
    # Benchmark
    t0 = time.perf_counter()
    for _ in range(args.iter):
        r = exec_.run(x=x)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / args.iter
    print(f"[bench] {args.iter} iters @ {r.provider_used}: "
          f"{avg_ms:.2f} ms/iter  (fallback={r.fallback_occurred})")

    # Top-K
    probs = _softmax(r.outputs[0][0])
    top = np.argsort(probs)[::-1][: args.topk]
    labels = LABELS_PATH.read_text(encoding="utf-8").splitlines()
    print(f"\n[top-{args.topk}] for {args.image.name}:")
    for i in top:
        print(f"  {probs[i] * 100:5.2f}%  {labels[i]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
