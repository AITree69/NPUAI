# NPUAI benchmark — `mobilenetv2-12.onnx`

- batch size : **1**
- generated  : 2026-06-10 18:03:28
- onnxruntime: `1.24.4`

| EP                          | ok | actual provider              |   mean (ms) |   p50 (ms) |   p95 (ms) |   min (ms) |   max (ms) |   FPS |
|-----------------------------|----|------------------------------|------------:|-----------:|-----------:|-----------:|-----------:|------:|
| `DmlExecutionProvider` | OK | `DmlExecutionProvider` |      1.43 |      1.43 |      1.54 |      1.30 |      1.64 |   698 |
| `CUDAExecutionProvider` | x | - | - | - | - | - | - | - |
| `OpenVINOExecutionProvider` | x | - | - | - | - | - | - | - |
| `QNNExecutionProvider` | x | - | - | - | - | - | - | - |
| `VitisAIExecutionProvider` | x | - | - | - | - | - | - | - |
| `CPUExecutionProvider` | OK | `CPUExecutionProvider` |      3.18 |      1.99 |      7.13 |      1.76 |     26.01 |   314 |

**Fastest**: `DmlExecutionProvider` at 1.43 ms/iter (698 FPS).
