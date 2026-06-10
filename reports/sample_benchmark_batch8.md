# NPUAI benchmark — `mobilenetv2-12.onnx`

- batch size : **8**
- generated  : 2026-06-10 18:02:27
- onnxruntime: `1.24.4`

| EP                          | ok | actual provider              |   mean (ms) |   p50 (ms) |   p95 (ms) |   min (ms) |   max (ms) |   FPS |
|-----------------------------|----|------------------------------|------------:|-----------:|-----------:|-----------:|-----------:|------:|
| `DmlExecutionProvider` | OK | `DmlExecutionProvider` |      4.28 |      4.22 |      4.72 |      4.08 |      5.02 |   234 |
| `CUDAExecutionProvider` | x | - | - | - | - | - | - | - |
| `OpenVINOExecutionProvider` | x | - | - | - | - | - | - | - |
| `QNNExecutionProvider` | x | - | - | - | - | - | - | - |
| `VitisAIExecutionProvider` | x | - | - | - | - | - | - | - |
| `CPUExecutionProvider` | OK | `CPUExecutionProvider` |     12.94 |     12.94 |     13.58 |     12.40 |     14.08 |    77 |

**Fastest**: `DmlExecutionProvider` at 4.28 ms/iter (234 FPS).
