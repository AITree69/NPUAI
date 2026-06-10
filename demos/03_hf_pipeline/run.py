"""Demo 3: HuggingFace -> NPUAI end-to-end text classification.

Pipeline:
    1. `hub.fetch_onnx(model_id)`            -- download or export ONNX
    2. `hub.prepare_text_input(model_id, ...)`-- tokenize via HF AutoTokenizer
    3. `NPUAIExecutor(onnx_path).run(feeds)`  -- run, auto-pick EP, fallback
    4. Decode logits via the model's `id2label` config

Run:
    # default: SST-2 sentiment (positive / negative)
    python -m demos.03_hf_pipeline.run

    # supply your own text
    python -m demos.03_hf_pipeline.run --text "I love this laptop"

    # batch + benchmark
    python -m demos.03_hf_pipeline.run --text "great" "terrible" "meh" --iter 30

    # a different model: any HF text-classification id that ships ONNX
    python -m demos.03_hf_pipeline.run --model SamLowe/roberta-base-go_emotions-onnx \
        --text "I am so excited about this!"

    # skip the network entirely: point at a pre-downloaded ONNX file
    python -m demos.03_hf_pipeline.run --local-path models/hf/.../model.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default model: a binary sentiment classifier with INT8 ONNX weights
# (~64 MB) and a 2-class output (POSITIVE / NEGATIVE). Good for first run.
DEFAULT_MODEL = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_TEXTS = [
    "I absolutely love this laptop, the new NPU is amazing!",
    "The screen is broken and the battery dies in two hours. Awful.",
    "It works, I guess. Nothing special.",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _decode(logits: np.ndarray, id2label: Optional[dict]) -> list:
    """Return [(label, confidence), ...] for each row in the batch."""
    probs = _softmax(logits)
    out = []
    for row in probs:
        top = int(np.argmax(row))
        label = (id2label or {}).get(top, f"class_{top}")
        out.append((label, float(row[top])))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HuggingFace model id (default: {DEFAULT_MODEL}). "
                        f"Ignored when --local-path is given.")
    p.add_argument("--local-path", type=Path, default=None,
                   help="Use a pre-downloaded ONNX file. Skips hub.fetch_onnx "
                        "and tokenizer-from-hf. Tokenizer is still loaded "
                        "from --model so the input tensors match the model's "
                        "expectations.")
    p.add_argument("--text", nargs="+", default=DEFAULT_TEXTS,
                   help="One or more strings to classify")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--prefer", default="auto", choices=["auto", "npu", "gpu", "cpu"])
    p.add_argument("--iter", type=int, default=10)
    p.add_argument("--force-refetch", action="store_true",
                   help="Re-download the ONNX file even if cached")
    p.add_argument("--explain", action="store_true",
                   help="Print hub metadata then exit")
    args = p.parse_args()

    # Lazy import: hub pulls in heavy stuff (HF + optimum stack)
    from npurai import hub

    if args.explain:
        print(hub.explain(args.model))
        return 0

    # Step 1: fetch ONNX (or use local)
    if args.local_path is not None:
        onnx_path = args.local_path
        if not onnx_path.exists():
            print(f"[err] --local-path {onnx_path} does not exist", file=sys.stderr)
            return 2
        print(f"[onnx]  (local) {onnx_path}  "
              f"({onnx_path.stat().st_size // 1024 // 1024} MB)")
    else:
        onnx_path = hub.fetch_onnx(
            args.model, task="text-classification", force=args.force_refetch,
        )
        print(f"[onnx]  {onnx_path}  "
              f"({onnx_path.stat().st_size // 1024 // 1024} MB)")

    # Step 2: load label map and tokenize
    id2label = hub.load_id2label(args.model)
    if id2label:
        print(f"[labels] {id2label}")
    else:
        print("[labels] (no id2label in config — will print class indices)")

    t0 = time.perf_counter()
    feeds = hub.prepare_text_input(
        args.model, args.text, max_length=args.max_length,
    )
    t_tok = (time.perf_counter() - t0) * 1000.0
    print(f"[tokenize] {len(args.text)} text(s) -> "
          f"{ {k: v.shape for k, v in feeds.items()} }  in {t_tok:.0f} ms")

    # Step 3: NPUAI
    from npurai import NPUAIExecutor
    exec_ = NPUAIExecutor(str(onnx_path), prefer=args.prefer)
    print(exec_.explain())

    # Warm-up
    for _ in range(3):
        exec_.run(feeds=feeds)
    # Benchmark
    t0 = time.perf_counter()
    for _ in range(args.iter):
        r = exec_.run(feeds=feeds)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / args.iter
    print(f"\n[bench] {args.iter} iters @ {r.provider_used}: "
          f"{avg_ms:.2f} ms/iter  (fallback={r.fallback_occurred})")

    # Step 4: decode
    logits = r.outputs[0]
    if logits.ndim == 2 and logits.shape[0] == len(args.text):
        decoded = _decode(logits, id2label)
        print(f"\n[results] for {len(args.text)} text(s):")
        for text, (label, conf) in zip(args.text, decoded):
            print(f"  {conf * 100:5.2f}%  {label:<10s}  | {text}")
    else:
        print(f"[results] raw logits shape={logits.shape}")
        print(logits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
