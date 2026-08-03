"""Shared helpers: seeding, device/dtype resolution, model+tokenizer loading, token gathers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def configure_stdout() -> None:
    """UTF-8 stdout/stderr so a cp1252 console cannot kill a stage on a print."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def disable_tf32() -> None:
    """Stop Ampere+ routing fp32 matmuls through TF32's ~10 mantissa bits."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def measurement_dtype(cfg, critical: bool = False) -> str:
    """Dtype for a measurement pass; critical ones (score_base, base variant) may be
    pinned higher via model.scoring_dtype."""
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
