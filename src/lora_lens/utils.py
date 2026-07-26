"""Shared helpers: seeding, device/dtype resolution, model+tokenizer loading, token gathers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


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

    Inference uses cfg.model.inference_dtype (fp16 on Kaggle T4/P100 — no bf16
    support, pitfall 6). Training loads fp32 master weights and relies on fp16
    autocast in the train loop instead.
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
