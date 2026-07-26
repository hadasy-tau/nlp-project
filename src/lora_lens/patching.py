"""Lens-free causal check: layer-wise activation patching.

Patch the residual stream after layer l from the LoRA model into the base model
and find the earliest layer at which the base model's output flips to the LoRA
answer. Causal rather than correlational, and it sidesteps the lens-validity
debate entirely.

Output: <output_dir>/results/patching.csv with one row per (fact, layer) plus a
first_flip_layer column per fact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .lenses import get_decoder_parts
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


def run_patching(cfg, tokenizer, conditions: pd.DataFrame, device) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    adapter = Path(cfg.output_dir) / "lora" / "final"
    if not adapter.exists():
        raise SystemExit("[patch] No trained adapter at lora/final — run train_lora first.")

    cap = cfg.patching.max_facts_per_condition
    sample = (conditions.groupby("condition", group_keys=False)
              .apply(lambda g: g.head(cap)).reset_index(drop=True))

    base_model = load_model(cfg, device=device)
    lora_model = load_model(cfg, device=device, adapter_path=adapter)
    base_layers, _, _ = get_decoder_parts(base_model)
    n_layers = len(base_layers)

    rows = []
    idx = list(range(len(sample)))
    for chunk in tqdm(list(batched(idx, cfg.patching.batch_size)), desc="patching"):
        sub = sample.iloc[chunk]
        enc = tokenizer(sub["prompt"].tolist(), return_tensors="pt", padding=True).to(device)
        ans = torch.tensor(sub["answer_token_id"].tolist(), device=device)

        lora_hidden = lora_model(**enc, output_hidden_states=True).hidden_states
        base_top1 = gather_last(base_model(**enc).logits, enc["attention_mask"]) \
            .float().argmax(dim=-1)

        # Patch the stream after layer l (lora_hidden[l] = output of block l-1... see
        # convention: hidden_states[l] is the stream after l blocks), i.e. replace the
        # output of base block l-1, for l = 1..n_layers.
        flips = torch.zeros((len(sub), n_layers + 1), dtype=torch.bool)
        flips[:, 0] = (base_top1 == ans).cpu()  # l=0: no patch, base as-is
        for layer in range(1, n_layers + 1):
            top1 = _patched_top1(base_model, enc, lora_hidden[layer], base_layers[layer - 1])
            flips[:, layer] = (top1 == ans).cpu()

        for j in range(len(sub)):
            r = sub.iloc[j]
            hit_layers = torch.nonzero(flips[j]).flatten().tolist()
            for layer in range(n_layers + 1):
                rows.append({"fact_id": r["fact_id"], "condition": r["condition"],
                             "layer": layer, "flipped": bool(flips[j, layer])})
            rows.append({"fact_id": r["fact_id"], "condition": r["condition"],
                         "layer": -1,  # summary row: earliest flip
                         "flipped": bool(hit_layers),
                         "first_flip_layer": hit_layers[0] if hit_layers else None})

    free_model(lora_model)
    free_model(base_model)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "patching.csv", index=False)
    summary = (df[df["layer"] == -1].groupby("condition")["first_flip_layer"]
               .agg(["mean", "median", "count"]))
    print("\n[patch] Earliest layer at which patching flips base -> answer:")
    print(summary.to_string())
