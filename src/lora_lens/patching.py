"""Lens-free causal check: layer-wise activation patching.

Patch the residual stream after layer l from the LoRA model into the base model
and find the earliest layer at which the base model's output flips to the LoRA
answer. Causal rather than correlational, and it sidesteps the lens-validity
debate entirely.

Patching runs across every checkpoint (same set used in analyze stage), so you
can see at which training step the causal structure shifts — not just before/after.

Output: <output_dir>/results/patching.csv with one row per (variant, fact, layer) plus a
first_flip_layer summary row per (variant, fact).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .lenses import get_decoder_parts
from .training import list_checkpoints
from .utils import batched, free_model, gather_last, load_model


@torch.no_grad()
def _patched_top1(base_model, enc, replacement: torch.Tensor, layer_module) -> torch.Tensor:
    """Run base_model with layer_module's output hidden states replaced; return top-1 ids."""

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (replacement,) + output[1:]
        return replacement

    handle = layer_module.register_forward_hook(hook)
    try:
        logits = base_model(**enc).logits
    finally:
        handle.remove()
    last = gather_last(logits, enc["attention_mask"])
    return last.float().argmax(dim=-1)


def _patch_one_checkpoint(base_model, lora_model, base_layers, n_layers,
                           sample, tokenizer, cfg, device, variant, step):
    """Run patching for a single LoRA checkpoint; return list of row dicts."""
    rows = []
    idx = list(range(len(sample)))
    for chunk in tqdm(list(batched(idx, cfg.patching.batch_size)),
                      desc=f"patching[{variant}]"):
        sub = sample.iloc[chunk]
        enc = tokenizer(sub["prompt"].tolist(), return_tensors="pt", padding=True).to(device)
        ans = torch.tensor(sub["answer_token_id"].tolist(), device=device)

        lora_hidden = lora_model(**enc, output_hidden_states=True).hidden_states
        base_top1 = gather_last(base_model(**enc).logits, enc["attention_mask"]) \
            .float().argmax(dim=-1)

        # Patch residual stream after layer l (lora_hidden[l] = output of block l-1).
        # l=0: no patch (base as-is); l=1..n_layers: patch output of base block l-1.
        flips = torch.zeros((len(sub), n_layers + 1), dtype=torch.bool)
        flips[:, 0] = (base_top1 == ans).cpu()
        for layer in range(1, n_layers + 1):
            top1 = _patched_top1(base_model, enc, lora_hidden[layer], base_layers[layer - 1])
            flips[:, layer] = (top1 == ans).cpu()

        for j in range(len(sub)):
            r = sub.iloc[j]
            hit_layers = torch.nonzero(flips[j]).flatten().tolist()
            for layer in range(n_layers + 1):
                rows.append({"variant": variant, "step": step,
                             "fact_id": r["fact_id"], "condition": r["condition"],
                             "layer": layer, "flipped": bool(flips[j, layer])})
            rows.append({"variant": variant, "step": step,
                         "fact_id": r["fact_id"], "condition": r["condition"],
                         "layer": -1,
                         "flipped": bool(hit_layers),
                         "first_flip_layer": hit_layers[0] if hit_layers else None})
    return rows


def run_patching(cfg, tokenizer, conditions: pd.DataFrame, device) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list_checkpoints(cfg)
    if not checkpoints:
        raise SystemExit("[patch] No trained adapters found — run train_lora first.")

    cap = cfg.patching.max_facts_per_condition
    sample = (conditions.groupby("condition", group_keys=False)
              .apply(lambda g: g.head(cap)).reset_index(drop=True))

    log_path = Path(cfg.output_dir) / "lora" / "training_log.csv"
    final_step = int(pd.read_csv(log_path)["step"].max()) if log_path.exists() else -1

    base_model = load_model(cfg, device=device)
    base_layers, _, _ = get_decoder_parts(base_model)
    n_layers = len(base_layers)

    all_rows = []
    for label, adapter_path in checkpoints:
        step = final_step if label == "final" else int(label.split("_")[1])
        lora_model = load_model(cfg, device=device, adapter_path=adapter_path)
        all_rows.extend(_patch_one_checkpoint(
            base_model, lora_model, base_layers, n_layers,
            sample, tokenizer, cfg, device, label, step))
        free_model(lora_model)

    free_model(base_model)

    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir / "patching.csv", index=False)

    summary = (df[df["layer"] == -1]
               .groupby(["variant", "step", "condition"])["first_flip_layer"]
               .agg(["mean", "median", "count"])
               .reset_index())
    print("\n[patch] Earliest layer at which patching flips base -> answer (per checkpoint):")
    print(summary.to_string(index=False))
