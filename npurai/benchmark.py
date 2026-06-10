"""Cross-EP benchmark for a single model.

Runs the same model under every available Execution Provider in turn and
prints a comparison table. The goal is to give the user a clear picture of
which EPs are *present* on this machine, which ones *work* for the model,
and how fast they go.

Output:
    reports/benchmark_<model>_<ts>.md

Run:
    python -m npurai.benchmark --model models/mobilenetv2-12.onnx
    python -m npurai.benchmark --model models/mobilenetv2-12.onnx --iter 50 --batch 8
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

ALL_EPS = [
    "DmlExecutionProvider",
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
    "CPUExecutionProvider",
]


def _make_feed(model_path: str, batch: int) -> Tuple[str, np.ndarray]:
    """Build a feed for the model's first input, batched to `batch`."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    name = inp.name
    shape = []
    for d in inp.shape:
        if isinstance(d, int) and d > 0:
            shape.append(d)
        else:
            # ORT may represent a dynamic dim as 0, -1, "N", or a symbolic name.
            shape.append(batch)
    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
    }
    np_dtype = dtype_map.get(inp.type, np.float32)
    return name, np.random.rand(*shape).astype(np_dtype)


def _bench_one_ep(
    model_path: str,
    ep: str,
    feed_name: str,
    feed_x: np.ndarray,
    warmup: int,
    iters: int,
) -> Dict:
    """Run `iters` after `warmup` warmup iters; return a stats dict."""
    try:
        # Note: do NOT append CPUExecutionProvider here — ORT warns about
        # duplicates if `ep` is CPU. Each EP is tested on its own.
        providers = [ep] if ep == "CPUExecutionProvider" else [ep, "CPUExecutionProvider"]
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:                       # noqa: BLE001
        return {"ep": ep, "ok": False, "error": f"{type(e).__name__}: {e}"}
    actual = sess.get_providers()[0]
    try:
        for _ in range(warmup):
            sess.run(None, {feed_name: feed_x})
    except Exception as e:                       # noqa: BLE001
        return {"ep": ep, "ok": False, "error": f"warmup failed: {e}",
                "actual_provider": actual}
    timings: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {feed_name: feed_x})
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(timings)
    return {
        "ep": ep,
        "ok": True,
        "actual_provider": actual,
        "iters": iters,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "throughput_fps": float(1000.0 / arr.mean()),
    }


def _render_markdown(model_path: str, results: List[Dict], batch: int) -> str:
    lines = [
        f"# NPUAI benchmark — `{Path(model_path).name}`",
        "",
        f"- batch size : **{batch}**",
        f"- generated  : {_dt.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- onnxruntime: `{ort.__version__}`",
        "",
        "| EP                          | ok | actual provider              |   mean (ms) |   p50 (ms) |   p95 (ms) |   min (ms) |   max (ms) |   FPS |",
        "|-----------------------------|----|------------------------------|------------:|-----------:|-----------:|-----------:|-----------:|------:|",
    ]
    for r in results:
        if not r["ok"]:
            mark = "x"
            lines.append(
                f"| `{r['ep']}` | {mark} | - | - | - | - | - | - | - |"
            )
            continue
        mark = "OK"
        lines.append(
            f"| `{r['ep']}` | {mark} | `{r['actual_provider']}` "
            f"| {r['mean_ms']:9.2f} | {r['p50_ms']:9.2f} | {r['p95_ms']:9.2f} "
            f"| {r['min_ms']:9.2f} | {r['max_ms']:9.2f} | {r['throughput_fps']:5.0f} |"
        )
    # Pick the winner
    good = [r for r in results if r["ok"]]
    if good:
        winner = min(good, key=lambda r: r["mean_ms"])
        lines.append("")
        lines.append(f"**Fastest**: `{winner['ep']}` at "
                     f"{winner['mean_ms']:.2f} ms/iter "
                     f"({winner['throughput_fps']:.0f} FPS).")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to an ONNX model")
    ap.add_argument("--iter", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument(
        "--eps", nargs="*", default=ALL_EPS,
        help="Subset of EPs to try (default: all)",
    )
    ap.add_argument("--out", type=Path, default=None,
                    help="Output markdown file (default: reports/benchmark_<ts>.md)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 2

    print(f"Model: {args.model}")
    feed_name, feed_x = _make_feed(args.model, args.batch)
    print(f"Feed  : name={feed_name}  shape={feed_x.shape}  dtype={feed_x.dtype}")

    available = ort.get_available_providers()
    print(f"Available EPs reported by ORT: {available}\n")

    results: List[Dict] = []
    for ep in args.eps:
        if ep not in available:
            results.append({"ep": ep, "ok": False, "error": "not installed"})
            print(f"[{ep}] skipped (not in available_providers)")
            continue
        print(f"[{ep}] running {args.iter} iters ...")
        r = _bench_one_ep(args.model, ep, feed_name, feed_x,
                          args.warmup, args.iter)
        results.append(r)
        if r["ok"]:
            print(f"  mean {r['mean_ms']:.2f} ms, p50 {r['p50_ms']:.2f} ms, "
                  f"p95 {r['p95_ms']:.2f} ms  -> {r['throughput_fps']:.0f} FPS "
                  f"(actual={r['actual_provider']})")
        else:
            print(f"  FAILED: {r.get('error', 'unknown')}")

    md = _render_markdown(args.model, results, args.batch)
    print("\n" + md)
    if args.out is None:
        out = Path("reports") / f"benchmark_{Path(args.model).stem}_{int(time.time())}.md"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote: {out}")
    print(f"Wrote: {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
