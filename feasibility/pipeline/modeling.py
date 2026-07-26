"""Model / tokenizer loading with the device and dtype rules for Kaggle GPUs."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    # Kaggle's T4 (sm 7.5) and P100 (sm 6.0) do not support bf16 — fp16 only.
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {"float16": torch.float16, "float32": torch.float32}[dtype]


def load_model_and_tokenizer(cfg: ModelConfig):
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    print(f"model={cfg.name}  device={device}  dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    # Right padding + explicit gather at the last real token (see scoring.py).
    # Left padding silently corrupts logits for GPT-2-style models.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded: {n_layers} layers, {n_params / 1e6:.0f}M params")
    return model, tokenizer, device
