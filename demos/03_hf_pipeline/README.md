# Demo 3: HuggingFace -> NPUAI end-to-end

The flagship demo of NPUAI's "anything that ships ONNX" promise.

Pipeline:
```
huggingface.co  ──►  onnx/model_quantized.onnx  ──►  NPUAIExecutor
                              │                              │
                              ▼                              ▼
                    AutoTokenizer  ──►  numpy feeds  ──►  CPU / Dml / NPU
                                                            │
                                                            ▼
                                                     softmax + id2label
```

## Run

```powershell
# default: SST-2 sentiment with INT8 ONNX weights (~64 MB)
python -m demos.03_hf_pipeline.run

# your own text
python -m demos.03_hf_pipeline.run --text "I love this NPU" "Awful experience"

# different model: any HF text-classification repo that ships ONNX
python -m demos.03_hf_pipeline.run --model SamLowe/roberta-base-go_emotions-onnx

# skip the network entirely
python -m demos.03_hf_pipeline.run --local-path models/hf/.../model.onnx

# show what the model looks like on the Hub without downloading
python -m demos.03_hf_pipeline.run --model bert-base-uncased --explain
```

## What you should see

```
[onnx]  D:\NPUAI\models\hf\Xenova__distilbert-base-uncased-finetuned-sst-2-english\onnx\model_quantized.onnx  (64 MB)
[labels] {0: 'NEGATIVE', 1: 'POSITIVE'}
[tokenize] 3 text(s) -> {'input_ids': (3, 128), 'attention_mask': (3, 128)}  in 0 ms

  Model        : ...
  Selected EPs : ['DmlExecutionProvider', 'CPUExecutionProvider']
  Actually used: DmlExecutionProvider
  ...

[bench] 10 iters @ DmlExecutionProvider: 8.41 ms/iter  (fallback=False)

[results] for 3 text(s):
   99.83%  POSITIVE    | I absolutely love this laptop, the new NPU is amazing!
   99.45%  NEGATIVE    | The screen is broken and the battery dies in two hours. Awful.
   58.32%  NEGATIVE    | It works, I guess. Nothing special.
```

## What does `npurai.hub` do?

`hub.fetch_onnx(model_id)` performs three-step acquisition:

1. **Cache hit** — if `models/hf/<safe_model_id>/model.onnx` (or
   `…/onnx/model.onnx`) already exists, return it. No network.
2. **ONNX in repo** — if the HF repo ships ONNX files (most `Xenova/*`
   repos do), pick the best variant (quantized > FP16 > FP32) and
   download just that one file.
3. **`optimum-cli export onnx`** — last-resort path: download PyTorch
   weights, export to ONNX. Slowest but works for every transformers
   model on the Hub.

`hub.prepare_text_input(model_id, texts)` wraps
`transformers.AutoTokenizer` to produce a numpy feed dict whose keys
exactly match the ONNX model's declared inputs.

`hub.load_id2label(model_id)` returns `{0: 'NEGATIVE', 1: 'POSITIVE'}`,
loaded from the model's `config.json`.

`hub.explain(model_id)` is a debugging helper that prints metadata about
a model without downloading anything.

## Network notes (mainland China)

`huggingface.co` is often slow from mainland ISPs. Set `HF_ENDPOINT`
before running:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"        # the canonical mirror
python -m demos.03_hf_pipeline.run
```

We bypass `huggingface_hub`'s built-in downloader and use a raw HTTP
GET against the endpoint you set, because the built-in helper does a
HEAD request and rejects responses that don't carry the original
Hub's `X-Repo-Commit` header — which mirrors sometimes omit.

If a file is partially downloaded, the next run will skip it (the
cache check is by path + non-zero size). To force a re-download, pass
`--force-refetch`.

## Adapting to a different task

The demo is hardcoded for `text-classification`, but `npurai.hub` does
not care which task the model solves. To use it for, say,
`token-classification` (NER) or `question-answering`, you would:

1. Pass `--model` pointing to a HF repo that ships ONNX for your task.
2. Write a small post-processor that decodes the model's output
   shape (different per task). Most tasks already have a reference
   implementation in the model's HF page — the hard part is always
   "where do I get the ONNX", which `hub.fetch_onnx` solves.
