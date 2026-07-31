"""Locality scoring: measure how much LoRA disrupts base-model predictions on
neighborhood prompts.

Primary metric: KL(base || lora) over the full output vocabulary — measures
representational drift regardless of whether top-1 flips. Lower is better.

Secondary metric: top-1 preservation rate (kept for reference / backward compat).
KL is the preferred metric because exact top-1 match fails spuriously when the
base model's top-two tokens are near 50/50 and LoRA merely swaps their order.

Output: <output_dir>/results/locality.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
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
    base_top1s, lora_top1s, kl_divs = [], [], []

    for chunk in tqdm(list(batched(idx, cfg.scoring.batch_size)), desc="locality"):
        sub = expanded.iloc[chunk]
        enc = tokenizer(sub["prompt"].tolist(), return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            base_logits = gather_last(
                base_model(**enc).logits, enc["attention_mask"]).float()
            lora_logits = gather_last(
                lora_model(**enc).logits, enc["attention_mask"]).float()

        base_logprobs = F.log_softmax(base_logits, dim=-1)
        lora_logprobs = F.log_softmax(lora_logits, dim=-1)
        # KL(base || lora) per example using log-space target for numerical stability.
        kl = F.kl_div(lora_logprobs, base_logprobs, reduction="none",
                      log_target=True).sum(dim=-1)

        base_top1s.append(base_logprobs.argmax(dim=-1).cpu())
        lora_top1s.append(lora_logprobs.argmax(dim=-1).cpu())
        kl_divs.append(kl.cpu())

    free_model(lora_model)
    free_model(base_model)

    expanded = expanded.copy()
    expanded["base_top1"] = torch.cat(base_top1s).numpy()
    expanded["lora_top1"] = torch.cat(lora_top1s).numpy()
    expanded["preserved"] = expanded["base_top1"] == expanded["lora_top1"]
    expanded["kl_div"] = torch.cat(kl_divs).numpy()

    expanded.to_csv(results_dir / "locality.csv", index=False)

    summary = (expanded.groupby("condition")
               .agg(kl_div_mean=("kl_div", "mean"),
                    preservation_rate=("preserved", "mean"),
                    n_prompts=("preserved", "count"))
               .reset_index())
    print("\n[locality] Neighborhood representational drift (KL(base||lora), lower = less disruption):")
    print(summary.to_string(index=False))
    print("           kl_div_mean: primary metric; preservation_rate: legacy top-1 match")
