# NPUAI — vendor-agnostic NPU / GPU / CPU inference for Windows

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]() [![ONNX Runtime](https://img.shields.io/badge/onnxruntime-1.18%2B-green)]() [![Windows 11](https://img.shields.io/badge/windows-11-0078d4)]() [![License: MIT](https://img.shields.io/badge/license-MIT-yellow)]()

**NPUAI** is a thin, vendor-agnostic Python layer that lets the **CPU / iGPU / dGPU / NPU** on a modern PC cooperate on ONNX inference — without writing per-vendor glue code.

It targets laptops and mini-PCs that ship with an NPU:

| Vendor    | NPU silicon                | Peak (TOPS) | EP path (install)              |
|-----------|----------------------------|-------------|--------------------------------|
| AMD       | XDNA (Ryzen 7040/8040)     | 10–16       | DirectML (built-in)            |
| AMD       | XDNA 2 (Ryzen AI 300, HX 370+) | 50      | DirectML / VitisAI EP          |
| Intel     | Meteor / Lunar Lake NPU    | 11 / 48     | OpenVINO EP                    |
| Qualcomm  | Hexagon NPU (Win ARM)      | 45          | QNN EP                         |
| Apple     | ANE (reference)            | 38          | CoreML EP                      |

The scheduler auto-picks the best EP, falls back gracefully, and prints a clear hardware report so users can see *which* device is running their model.

---

## Why

NPU ecosystems in 2025 are still fragmented:

- AMD ships **Ryzen AI SDK** (VitisAI EP) — ~1.5 GB install, great perf, but Intel/Qualcomm-only users get nothing.
- Intel ships **OpenVINO** — same story, AMD only gets GPU fallback.
- Qualcomm ships **QNN SDK** — Windows on ARM is even more of a wasteland.
- The only thing they all share is `onnxruntime`.

A single Python call should not care which NPU is in the box. **NPUAI** is the missing 200-line scheduler that wraps all of them behind one `NPUAIExecutor` class.

---

## Quick start

```powershell
git clone https://github.com/<you>/NPUAI.git
cd NPUAI
pip install -r requirements.txt

# 1. See what your box has
python -m npurai.detect

# 2. Run a real model on whatever is best
python -m demos.01_classify.run
python -m demos.02_pipeline.run

# 3. Cross-EP benchmark
python -m npurai.benchmark --model models/mobilenetv2-12.onnx --batch 8
```

---

## What you get on the author's box

Machine: **AMD Ryzen AI 9 HX 370, Radeon 890M, RTX 5070 Laptop, Windows 11 24H2**

```
================================================================
  NPUAI hardware report  -  Windows 11 (10.0.26200)
  Python 3.12.10
================================================================
  CPU  : AMD Ryzen AI 9 HX 370 w/ Radeon 890M (12c/24t, AMD64)
  GPU  : AMD Radeon(TM) 890M Graphics (512 MB, drv 32.0.22018.2001)
  GPU  : NVIDIA GeForce RTX 5070 Laptop GPU (4095 MB, drv 32.0.15.9200)
  NPU  : AMD NPU: XDNA 2 [~50 TOPS] (cpu-fingerprint)
================================================================
```

Cross-EP benchmark (MobileNet v2, batch 8, ONNX 1.24.4) — see
[`reports/sample_benchmark_batch8.md`](reports/sample_benchmark_batch8.md):

| EP                          | ok | actual provider              |   mean (ms) |   p50 (ms) |   p95 (ms) |   min (ms) |   max (ms) |   FPS |
|-----------------------------|----|------------------------------|------------:|-----------:|-----------:|-----------:|-----------:|------:|
| `DmlExecutionProvider`      | OK | `DmlExecutionProvider`       |       4.28  |       4.22 |       4.72 |       4.08 |       5.02 |  234  |
| `CUDAExecutionProvider`     | x  | -                            | -           | -          | -          | -          | -          | -     |
| `OpenVINOExecutionProvider` | x  | -                            | -           | -          | -          | -          | -          | -     |
| `QNNExecutionProvider`      | x  | -                            | -           | -          | -          | -          | -          | -     |
| `VitisAIExecutionProvider`  | x  | -                            | -           | -          | -          | -          | -          | -     |
| `CPUExecutionProvider`      | OK | `CPUExecutionProvider`       |      12.94  |      12.94 |      13.58 |      12.40 |      14.08 |   77  |

**Dml 3× faster than CPU** at batch 8. RTX 5070 dGPU is sitting idle because
`onnxruntime-directml` doesn't see it (DirectML on Win picks the iGPU
when an NPU is absent on the active session). Install
`onnxruntime-gpu` to get CUDA — see the EP matrix below.

---

## Repository layout

```
NPUAI/
├── npurai/                  # core package
│   ├── __init__.py
│   ├── detect.py            # hardware inventory (WMI, registry, DXGI)
│   ├── executor.py          # NPUAIExecutor: pick + run + fall back
│   └── benchmark.py         # cross-EP benchmark, markdown output
├── demos/
│   ├── 01_classify/         # MobileNet v2 image classification
│   ├── 02_pipeline/         # multi-model pipeline (CNN + SqueezeNet + MNIST)
│   └── (more to come)
├── models/                  # cached ONNX models (gitignored)
├── reports/                 # benchmark markdown + json (sample reports committed)
├── tests/
│   └── smoke_executor.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## How it works

### `NPUAIExecutor` (the only class demos touch)

```python
from npurai import NPUAIExecutor

# Auto-pick: prefer Dml/NPU, fall back to CPU.
exec_ = NPUAIExecutor("mobilenetv2.onnx", prefer="auto")
print(exec_.explain())         # see which EPs were tried and what worked

# Two ways to pass inputs:
out = exec_.run(x=img_array)            # short-hand for single-input
out = exec_.run(feeds={"input": x})     # explicit, for multi-input

print(out.provider_used, out.latency_ms, out.fallback_occurred)
```

### Provider priority

```
QNNExecutionProvider         (Qualcomm Hexagon NPU)
VitisAIExecutionProvider     (AMD XDNA — install onnxruntime-vitisai)
OpenVINOExecutionProvider    (Intel NPU/iGPU/dGPU)
DmlExecutionProvider         (DirectML — covers AMD/Intel/NVIDIA on Win)
CUDAExecutionProvider        (NVIDIA dGPU — install onnxruntime-gpu)
CPUExecutionProvider         (always available)
```

The priority list is filtered by the local hardware report (e.g. CUDA
EP is dropped from the list when no NVIDIA GPU is present) and trimmed
by what `onnxruntime.get_available_providers()` reports. The first EP
that loads the model and runs a 1×1 identity smoke-test wins.

### Hardware detection

Three sources, in order:

1. **WMI** (`Win32_Processor`, `Win32_VideoController`) — gives CPU
   name/cores and every GPU the OS sees.
2. **Windows registry** — looks for NPU driver services:
   `AMDXE | XDNA | AIE | RyzenAI` (AMD), `IntelNPU | IntelAI | IntelMovidius` (Intel),
   `qcn | hexagon | QNN | Adreno` (Qualcomm). Intel IPU MCDM is
   explicitly *not* matched (it lives on every Ryzen AI box but isn't
   the NPU).
3. **CPU-name fingerprint** — `"Ryzen AI"` → XDNA / XDNA 2,
   `"Core Ultra" / "Meteor" / "Lunar"` → Intel NPU.

If a real NPU driver is installed but the registry never gets matched
(W11 24H2 sometimes does this for XDNA 2), the CPU-name fingerprint
still tells you the silicon class. TOPS comes from a hard-coded
table; **the actual measured TOPS on a real model is what
`npurai.benchmark` reports.**

---

## Extending to a new vendor

1. `pip install onnxruntime-<vendor>` (whl published by ORT).
2. Add an entry to `_PROVIDER_PREFERENCE` in
   `npurai/executor.py` (priority is list order, top = most preferred).
3. Add a provider-options builder in `NPUAIExecutor._build_provider_options`.
4. If the vendor's NPU registry keys are unique, extend
   `_try_find_<vendor>_npu` in `npurai/detect.py`.

That's it. The benchmarks and demos pick the new EP up automatically
once it's available.

---

## EP coverage matrix

| EP                       | Install pip extra             | Target hardware                 | Status on author's box  |
|--------------------------|-------------------------------|---------------------------------|-------------------------|
| `DmlExecutionProvider`   | `onnxruntime-directml`        | AMD / Intel / NVIDIA GPU        | works (3× CPU on batch 8) |
| `CPUExecutionProvider`   | (built in)                    | everything                      | works                   |
| `CUDAExecutionProvider`  | `onnxruntime-gpu`             | NVIDIA dGPU                     | not installed           |
| `OpenVINOExecutionProvider` | `onnxruntime-openvino`     | Intel NPU / iGPU / dGPU         | not installed           |
| `QNNExecutionProvider`   | `onnxruntime-qnn`             | Qualcomm Hexagon NPU            | not installed           |
| `VitisAIExecutionProvider` | `onnxruntime-vitisai` + Ryzen AI SDK | AMD XDNA / XDNA 2    | not installed           |

If you test on Intel / Qualcomm / Apple silicon, **open a PR** with
the corresponding `reports/sample_benchmark_*.md` — the matrix above
is the project's changelog of "we actually ran this".

---

## Known limitations

- **NPU peaks are estimates, not measurements.** The detector reports
  vendor-marketed TOPS; the real sustained throughput on your model
  is whatever `npurai.benchmark` prints.
- **DirectML on this machine hits a known UTF-8 regression in ORT 1.24**
  when `provider_options` is non-`None`. NPUAI works around it by
  passing `None` for the Dml entry. The fix is in upstream ORT.
- **External-data ONNX models** (`.onnx` + `.data` file) need the
  data file next to the model. The ONNX zoo models in the demos
  are inline-data, so this isn't an issue out of the box.
- **No int8 quantisation yet** — that's the next sprint. The plan is
  to wrap `onnxruntime.quantization` with a CLI so you can quantise
  any model with one command and then run it via the same executor.

---

## Roadmap

- [ ] `npuai quantize` — int8 / int4 quantisation helper.
- [ ] `npuai serve` — minimal HTTP server (FastAPI) that exposes
      model + EP via a single endpoint.
- [ ] Whisper-tiny int8 demo (real NPU utilisation, not just CNN).
- [ ] Real-time webcam classify demo with frame timing.
- [ ] AMD VitisAI EP install + benchmark (XDNA 2, 50 TOPS path).
- [ ] CI matrix that runs the benchmark on at least one AMD and one
      Intel box per release.

---

## License

MIT. See [`LICENSE`](LICENSE).
