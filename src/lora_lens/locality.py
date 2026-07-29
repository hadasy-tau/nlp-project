"""Locality scoring: measure how much LoRA disrupts base-model predictions on
neighborhood prompts.

For each neighborhood prompt (a related but distinct fact from CounterFact), we
record the base model's top-1 token, then check whether the final LoRA adapter
predicts the same token. The preservation rate = fraction of prompts where the
two agree. Low preservation means LoRA has side-effects beyond the trained facts.

Output: <output_dir>/results/locality.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .utils import batched, free_model, gather_last, load_model


def run_locality_scoring(cfg, tokenizer, conditions: pd.DataFrame, device) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    adapter = Path(cfg.output_dir) / "lora" / "final"
    if not adapter.exists():
        raise SystemExit("[locality] No trained adapter at lora/final — run train_lora first.")

    # Expand neighborhood prompts — one row per prompt per fact.
    rows_exp = []
    for _, r in conditions.iterrows():
        val = r.get("neighborhood_prompts", None)
        prompts = list(val) if val is not None and hasattr(val, "__iter__") else []
        for p in prompts:
            rows_exp.append({"fact_id": r["fact_id"], "condition": r["condition"],
                             "prompt": str(p).rstrip()})

    if not rows_exp:
        print("[locality] No neighborhood prompts found — skipping locality scoring.\n"
              "           Add 'neighborhood: neighborhood_prompts' under data.fields in "
              "your config.")
        return

    expanded = pd.DataFrame(rows_exp)
    print(f"[locality] {len(expanded)} neighborhood prompts across "
          f"{conditions['condition'].value_counts().to_dict()}")

    base_model = load_model(cfg, device=device)
    lora_model = load_model(cfg, device=device, adapter_path=adapter)

    idx = list(range(len(expanded)))
    base_top1s, lora_top1s = [], []

    for chunk in tqdm(list(batched(idx, cfg.scoring.batch_size)), desc="locality"):
        sub = expanded.iloc[chunk]
        enc = tokenizer(sub["prompt"].tolist(), return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            base_logits = gather_last(
                base_model(**enc).logits, enc["attention_mask"]).float()
            lora_logits = gather_last(
                lora_model(**enc).logits, enc["attention_mask"]).float()
        base_top1s.append(base_logits.argmax(dim=-1).cpu())
        lora_top1s.append(lora_logits.argmax(dim=-1).cpu())

    free_model(lora_model)
    free_model(base_model)

    expanded = expanded.copy()
    expanded["base_top1"] = torch.cat(base_top1s).numpy()
    expanded["lora_top1"] = torch.cat(lora_top1s).numpy()
    expanded["preserved"] = expanded["base_top1"] == expanded["lora_top1"]

    expanded.to_csv(results_dir / "locality.csv", index=False)

    summary = (expanded.groupby("condition")["preserved"]
               .agg(preservation_rate="mean", n_prompts="count")
               .reset_index())
    print("\n[locality] Neighborhood preservation rate (LoRA top-1 == base top-1):")
    print(summary.to_string(index=False))
    print("           (1.0 = no side-effects on related facts; lower = more disruption)")
