"""Shared helpers: seeding, device/dtype resolution, model+tokenizer loading, token gathers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def configure_stdout() -> None:
    """Make stdout/stderr UTF-8 so printing can never abort a run.

    Windows consoles default to cp1252, which cannot encode the arrows, deltas and
    em-dashes used throughout this package's progress output — `print` then raises
    UnicodeEncodeError and kills the stage *after* the expensive work is done.
    (report_highlights printing "base -> final" was doing exactly that; stats.py
    worked around it by hand-restricting one header to ASCII, which does not
    generalise.) errors="replace" means an unrenderable glyph degrades to '?'
    instead of taking the pipeline down.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # already detached / not a text stream — nothing to do


def disable_tf32() -> None:
    """Keep fp32 matmuls in real fp32.

    On Ampere+ (A100) PyTorch may route fp32 matmuls through TF32 tensor cores,
    which keep only ~10 mantissa bits — barely better than fp16 and enough to
    reintroduce the near-tie rank flipping that model.inference_dtype=float32 is
    here to eliminate (see _check_base_final_layer in analysis.py). No-op on
    T4/P100, which have no TF32 path.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def measurement_dtype(cfg, critical: bool = False) -> str:
    """Dtype name for a measurement forward pass.

    `critical=True` marks the two passes that define and then re-check the
    condition split (score_base, and the base variant in analyze). They can be
    pinned above the bulk precision via model.scoring_dtype so the split stays
    reproducible even when the bulk passes run in fp16 — the fallback for
    hardware without fast fp32 (T4/P100). Leave scoring_dtype null to measure
    everything at the same precision, which is what A100 runs should do.
    """
    if critical:
        override = cfg.model.get("scoring_dtype")
        if override:
            return override
    return cfg.model.inference_dtype


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg) -> torch.device:
    if cfg.model.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.model.device)


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding + explicit last-real-token gather (pitfall 5). Left padding
    # silently corrupts logits on models that need manual position_ids.
    tok.padding_side = "right"
    return tok


def load_model(cfg, device=None, dtype: str | None = None, adapter_path: str | Path | None = None,
               train: bool = False):
    """Load the base model, optionally with a LoRA adapter attached.

    Inference defaults to cfg.model.inference_dtype; measurement call sites pass
    `dtype=measurement_dtype(cfg, ...)` explicitly. Training loads fp32 master
    weights and relies on fp16 autocast in the train loop instead.
    """
    device = device or resolve_device(cfg)
    torch_dtype = DTYPES[dtype or ("float32" if train else cfg.model.inference_dtype)]
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, revision=cfg.model.get("revision"), torch_dtype=torch_dtype
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=train)
    model.to(device)
    model.train(train)
    return model


def encode_answer(tokenizer, answer: str) -> list[int]:
    """Tokenize an answer as it appears after a prompt — WITH the leading space.

    '"Paris"' and '" Paris"' are different BPE tokens; after "...the city of" the
    model predicts the space-prefixed variant (pitfall 1).
    """
    return tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last real (non-pad) token per row, for right-padded batches."""
    return attention_mask.sum(dim=1) - 1


def gather_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Gather [B, T, D] -> [B, D] at each row's last real token."""
    idx = last_token_index(attention_mask)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def free_model(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
