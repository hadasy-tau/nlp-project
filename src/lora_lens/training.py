"""LoRA fine-tuning on factual statements, with per-step adapter checkpoints.

Training text is prompt + " " + answer; loss is computed on the answer token(s)
only. Adapters are saved every cfg.lora.checkpoint_every optimizer steps so that
layer metrics can be reported as a function of training step (pitfall 4 — a fixed
step count confounds fact type with amount of learning).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import encode_answer, load_model


class FactDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.examples = []
        for _, r in df.iterrows():
            prompt_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
            answer_ids = encode_answer(tokenizer, r["answer"])  # leading space handled here
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            self.examples.append((input_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids, labels, mask = [], [], []
        for ids, lab in batch:  # right padding (pitfall 5)
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    return collate


def train_lora(cfg, tokenizer, conditions: pd.DataFrame, device) -> Path:
    """Fine-tune LoRA on all three conditions jointly; returns the lora output dir."""
    # Imported lazily (like load_model does) so that list_checkpoints — and therefore
    # the analysis/patching/locality stages that only need it — stay importable in a
    # peft-free environment, e.g. post-hoc analysis of downloaded parquet artifacts.
    from peft import LoraConfig, get_peft_model

    out_dir = Path(cfg.output_dir) / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    # fp32 master weights + fp16 autocast: stable on T4/P100 (no bf16, pitfall 6).
    model = load_model(cfg, device=device, train=True)
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = FactDataset(conditions, tokenizer)
    loader = DataLoader(ds, batch_size=cfg.lora.batch_size, shuffle=True,
                        collate_fn=make_collate(tokenizer.pad_token_id))
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lora.lr)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    log_rows = []
    for epoch in range(cfg.lora.epochs):
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.lora.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.lora.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            step += 1
            log_rows.append({"step": step, "epoch": epoch + 1, "loss": loss.item()})
            if step % cfg.lora.checkpoint_every == 0:
                model.save_pretrained(str(out_dir / f"step_{step:05d}"))

    model.save_pretrained(str(out_dir / "final"))
    with open(out_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train] {step} steps, adapters in {out_dir}")
    return out_dir


def list_checkpoints(cfg) -> list[tuple[str, Path]]:
    """(label, adapter_path) pairs for saved checkpoints, ordered by step, final last."""
    out_dir = Path(cfg.output_dir) / "lora"
    ckpts = sorted(out_dir.glob("step_*"))
    wanted = cfg.analysis.checkpoints
    if wanted != "all":
        keep = {int(s) for s in wanted}
        ckpts = [c for c in ckpts if int(c.name.split("_")[1]) in keep]
    pairs = [(c.name, c) for c in ckpts]
    final = out_dir / "final"
    if final.exists():
        pairs.append(("final", final))
    return pairs
