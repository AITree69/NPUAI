# Demo 2: multi-model pipeline

A 3-stage pipeline (CNN → tiny CNN → MLP) that exercises the executor
across models of very different shapes. Each model goes through its
own `NPUAIExecutor` and may end up on a different EP — this demo
shows that the scheduler does *not* lock the whole pipeline to a
single device.

## Run

```powershell
python -m demos.02_pipeline.run
python -m demos.02_pipeline.run --prefer cpu
python -m demos.02_pipeline.run --iter 50
```

## What you should see

```
  mobilenetv2          (vision CNN          )  DmlExecutionProvider  1.78 ms/iter
  squeezenet1.0        (lightweight vision  )  DmlExecutionProvider  1.20 ms/iter
  mnist-8              (tiny MLP            )  DmlExecutionProvider  0.17 ms/iter

  Total wall time per pipeline pass: 3.2 ms
```

All three land on the iGPU via DirectML in our reference box. If you
have an actual NPU driver installed (AMD VitisAI, Intel OpenVINO, or
Qualcomm QNN) the small models tend to *also* land on the NPU — the
scheduler prefers NPU > dGPU > iGPU > CPU.
