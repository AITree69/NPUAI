# Demo 1: image classification (MobileNet v2)

A 50-line demo that exercises the full NPUAI stack:

1. Calls `detect_devices()` and prints the hardware report.
2. Loads MobileNet v2 (ONNX, FP32) via `NPUAIExecutor`.
3. Downloads a sample image (or synthesises a gradient if offline).
4. Runs preprocessing (resize/center-crop/normalize, PyTorch-style).
5. Prints top-5 ImageNet labels with confidence and a per-iter latency.

## Run

```powershell
# default: auto-pick best EP, random sample image
python -m demos.01_classify.run

# explicit: force CPU and supply your own image
python -m demos.01_classify.run --prefer cpu --image path/to/dog.jpg

# benchmark with 100 iters
python -m demos.01_classify.run --iter 100
```

## What you should see

```
================================================================
  NPUAI hardware report  -  Windows 11 (10.0.26200)
================================================================
  CPU  : AMD Ryzen AI 9 HX 370 w/ Radeon 890M (12c/24t, AMD64)
  GPU  : AMD Radeon(TM) 890M Graphics (...)
  GPU  : NVIDIA GeForce RTX 5070 Laptop GPU (...)
  NPU  : AMD NPU: XDNA 2 [~50 TOPS] (cpu-fingerprint)
================================================================
  ...
  [bench] 10 iters @ DmlExecutionProvider: 1.54 ms/iter
  [top-5] for sample.jpg:
     6.88%  desk
     6.35%  altar
     6.20%  restaurant
     4.29%  dining table
     3.31%  barber chair
```

The model + sample image are auto-fetched on first run (cached under
`models/` and `demos/01_classify/` respectively). Subsequent runs are
fully offline.
