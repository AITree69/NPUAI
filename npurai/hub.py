"""HuggingFace Hub integration for NPUAI.

This module is the bridge between the HuggingFace model zoo and the
NPUAIExecutor. It is intentionally thin: it does not redefine inference,
tokenisation, or post-processing — it composes `huggingface_hub` +
`optimum-cli` + `transformers` and hands a standard ONNX file + numpy
feed dict to the rest of the package.

Typical use
-----------
    from npurai.hub import fetch_onnx, prepare_text_input, load_id2label

    onnx_path = fetch_onnx("bert-base-uncased", task="text-classification")
    feeds     = prepare_text_input("bert-base-uncased", ["I love this"])
    id2label  = load_id2label("bert-base-uncased")
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Default cache root: <repo>/models/hf/<model_id>
REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_ROOT = REPO_ROOT / "models" / "hf"


# ---------------------------------------------------------------------------
# ONNX model acquisition
# ---------------------------------------------------------------------------

def _sanitize(model_id: str) -> str:
    """HF model ids contain '/' (e.g. 'openai/whisper-tiny'); use '__'."""
    return model_id.replace("/", "__")


def _find_onnx_in_dir(d: Path) -> Optional[Path]:
    """Pick the main ONNX file in a directory.

    Preference order:
      1. `model.onnx`             — what `optimum-cli export onnx` writes
      2. `onnx/model.onnx`        — some HF repos have an `onnx/` subdir
      3. any other `*.onnx` file
    """
    if (d / "model.onnx").exists():
        return d / "model.onnx"
    if (d / "onnx" / "model.onnx").exists():
        return d / "onnx" / "model.onnx"
    others = sorted(d.rglob("*.onnx"))
    return others[0] if others else None


def _has_onnx_in_repo(model_id: str) -> List[str]:
    """Return ONNX file paths in the HF repo's tree, or []."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(model_id)
        return [f for f in files if f.endswith(".onnx")]
    except Exception:                            # noqa: BLE001
        return []


def _pick_preferred_onnx(onnx_files: List[str]) -> List[str]:
    """Choose the single best ONNX file from a repo.

    Many `-onnx` repos on the Hub ship every quantisation variant the
    maintainer exported (FP32, FP16, INT8, INT4, …). Downloading them all
    wastes bandwidth, so we keep at most one:

      1. `model_quantized.onnx` or `model_int8.onnx`   — preferred for NPU
      2. `model_fp16.onnx`                             — fallback
      3. `model.onnx`                                  — last resort (FP32)

    Returns the list with the single chosen file. If `onnx_files` is
    empty, returns it unchanged.
    """
    if not onnx_files:
        return onnx_files
    by_name = {Path(f).stem: f for f in onnx_files}
    for candidate in ("model_quantized", "model_int8", "model_uint8",
                      "model_q4", "model_q4f16", "model_bnb4"):
        if candidate in by_name:
            return [by_name[candidate]]
    for candidate in ("model_fp16", "model_quantized.onnx"):
        if candidate in by_name:
            return [by_name[candidate]]
    # Default: first file that contains 'model' and not 'data'/'config'
    preferred = [f for f in onnx_files
                 if "data" not in Path(f).stem and "config" not in Path(f).stem]
    return preferred[:1] or onnx_files[:1]


def _download_onnx_from_hf(model_id: str, dest: Path) -> Optional[Path]:
    """Download ONNX files from HuggingFace Hub.

    Strategy
    --------
    1. If `HF_ENDPOINT` (or `HF_HUB_ENDPOINT`) is set, hit that mirror
       directly via `urllib`. This is the path users in mainland China
       need — `hf-mirror.com` is the canonical mirror, and we cannot
       route through `hf_hub_download` because that helper does a HEAD
       request and refuses to recognise the mirror's response when the
       `X-Repo-Commit` header is missing.
    2. Otherwise, call `hf_hub_download` per file. Companion files
       (tokenizer / config) are NOT pulled here — `transformers`
       handles those through its own cache the first time a tokenizer
       is loaded.
    """
    onnx_files = _has_onnx_in_repo(model_id)
    if not onnx_files:
        return None
    onnx_files = _pick_preferred_onnx(onnx_files)
    endpoint = os.environ.get("HF_ENDPOINT") or os.environ.get(
        "HF_HUB_ENDPOINT") or "https://huggingface.co"
    print(f"[hf] downloading {len(onnx_files)} ONNX file(s) from {model_id} "
          f"via {endpoint}")
    import urllib.request
    for f in onnx_files:
        # `f` may start with "onnx/" — keep that as the local subdirectory.
        target = dest / f
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint.rstrip('/')}/{model_id}/resolve/main/{f}"
        if target.exists() and target.stat().st_size > 0:
            print(f"  - {f}  (cached, {target.stat().st_size // 1024 // 1024} MB)")
            continue
        print(f"  - {f}  ...", end="", flush=True)
        try:
            _urlretrieve_with_progress(url, target)
        except Exception as e:                   # noqa: BLE001
            print(f" FAILED ({e})")
            raise
        print(f" ok ({target.stat().st_size // 1024 // 1024} MB)")
    return _find_onnx_in_dir(dest)


def _urlretrieve_with_progress(url: str, dest: Path, chunk: int = 1 << 16) -> None:
    """`urllib.request.urlretrieve` is silent; this version prints
    download progress in 10% steps. We use stdlib only — no tqdm — so
    the demos can run in any environment."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        done = 0
        next_mark = 10
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total > 0:
                pct = int(done * 100 / total)
                while pct >= next_mark:
                    print(f" {next_mark}%", end="", flush=True)
                    next_mark += 10
    print("", flush=True)  # newline after the percentage markers


def _export_via_optimum(
    model_id: str,
    dest: Path,
    task: Optional[str] = None,
) -> Path:
    """Shell out to `optimum-cli export onnx`. This is the fallback path
    when the HF repo does not ship ONNX files of its own."""
    cmd = ["optimum-cli", "export", "onnx"]
    if task:
        cmd += ["--task", task]
    cmd += ["--model", model_id, str(dest)]
    print(f"[optimum] {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(
            f"`optimum-cli export onnx` failed for {model_id!r} "
            f"with return code {rc}."
        )
    onnx_path = _find_onnx_in_dir(dest)
    if onnx_path is None:
        raise FileNotFoundError(
            f"After `optimum-cli export onnx` no *.onnx file was found "
            f"under {dest}. Check the export logs above."
        )
    return onnx_path


def fetch_onnx(
    model_id: str,
    output_dir: Optional[Path] = None,
    *,
    task: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Fetch or export an ONNX model from HuggingFace Hub.

    Returns the path to the local ONNX file.

    Strategy
    --------
    1. If `output_dir` already has an ONNX file and `force=False`, return it.
    2. If the HF repo ships ONNX files (some `-onnx` sibling repos do),
       download them via `snapshot_download` — much smaller than pulling
       the PyTorch weights.
    3. Otherwise, run `optimum-cli export onnx` to convert from PyTorch.
       This is the slowest path (PyTorch download + export) but works for
       every text/vision/audio model on the Hub.
    """
    dest = Path(output_dir) if output_dir else HF_CACHE_ROOT / _sanitize(model_id)
    dest.mkdir(parents=True, exist_ok=True)

    if not force:
        cached = _find_onnx_in_dir(dest)
        if cached is not None:
            print(f"[hf] using cached {cached}")
            return cached

    # Strategy 2
    onnx_path = _download_onnx_from_hf(model_id, dest)
    if onnx_path is not None:
        return onnx_path

    # Strategy 3
    return _export_via_optimum(model_id, dest, task=task)


# ---------------------------------------------------------------------------
# Tokenisation / input prep
# ---------------------------------------------------------------------------

def prepare_text_input(
    model_id: str,
    texts: Iterable[str],
    *,
    max_length: int = 128,
    padding: bool = True,
    truncation: bool = True,
) -> Dict[str, np.ndarray]:
    """Run the model's AutoTokenizer on a list of strings.

    Returns a dict of numpy arrays, e.g. `{"input_ids": ..., "attention_mask": ...}`
    ready to be passed to `NPUAIExecutor.run(feeds=...)`. The dict only
    contains the keys the model actually expects — `token_type_ids` is
    included only if the tokenizer produced it.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    enc = tok(
        list(texts),
        max_length=max_length,
        padding="max_length" if padding else False,
        truncation=truncation,
        return_tensors="np",
    )
    keep = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    return {k: np.asarray(v) for k, v in enc.items() if k in keep}


def load_id2label(
    model_id_or_path: str,
) -> Optional[Dict[int, str]]:
    """Load the `id2label` mapping from a HF model's config.

    Works with both online model ids (`bert-base-uncased`) and local
    paths (e.g. our `models/hf/bert-base-uncased/`).
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id_or_path)
        if getattr(cfg, "id2label", None):
            return {int(k): str(v) for k, v in cfg.id2label.items()}
    except Exception:                            # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def list_onnx_files(model_id: str) -> List[str]:
    """Convenience helper: which ONNX files does the HF repo have?
    Useful for picking which sub-model to load in a multi-file repo."""
    return _has_onnx_in_repo(model_id)


def explain(model_id: str) -> str:
    """Pretty-print everything we know about a model id."""
    from huggingface_hub import HfApi
    api = HfApi()
    lines = [f"Model: {model_id}"]
    try:
        info = api.model_info(model_id)
        lines.append(f"  Tags    : {', '.join(info.tags or [])}")
        lines.append(f"  Pipeline: {info.pipeline_tag}")
    except Exception as e:                       # noqa: BLE001
        lines.append(f"  (could not fetch metadata: {e})")
    onnx = _has_onnx_in_repo(model_id)
    if onnx:
        lines.append(f"  ONNX in repo: {onnx}")
    else:
        lines.append("  ONNX in repo: <none — will need optimum-cli export>")
    return "\n".join(lines)
