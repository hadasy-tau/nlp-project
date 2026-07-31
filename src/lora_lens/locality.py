"""Locality scoring: measure how much LoRA disrupts base-model predictions on
neighborhood prompts.

Primary metric: KL(base || lora) restricted to the top-k tokens under the base
distribution (default k=50, configurable via scoring.locality_topk). Both
distributions are renormalized over the top-k support before computing KL, so
the result is a proper KL between two distributions over the same k-token
vocabulary. This avoids the full-vocabulary explosion caused by summing tiny
shifts across 50k long-tail tokens while preserving sensitivity to meaningful
distributional drift in the high-probability region. Lower is better.

Secondary metric: top-1 preservation rate (fraction of prompts where LoRA and
base agree on top-1 token).

Output: <output_dir>/results/locality.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import batched, free_model, gather_last, load_model


def _topk_kl(base_logprobs: torch.Tensor, lora_logprobs: torch.Tensor, k: int) -> torch.Tensor:
    """KL(base || lora) over the top-k tokens under the base distribution.

    Both log-prob vectors are sliced to the top-k base indices and renormalized
    over that support before computing KL, yielding a valid probability-simplex
    comparison on the tokens that carry the bulk of the base model's probability
    mass.
    """
    topk_vals, topk_idx = base_logprobs.topk(k, dim=-1)            # [B, k]
    lora_topk = lora_logprobs.gather(1, topk_idx)                   # [B, k]
    base_topk = topk_vals - torch.logsumexp(topk_vals, dim=-1, keepdim=True)
    lora_topk = lora_topk - torch.logsumexp(lora_topk, dim=-1, keepdim=True)
    return F.kl_div(lora_topk, base_topk, reduction="none", log_target=True).sum(dim=-1)


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
        topk = cfg.scoring.get("locality_topk", 50)
        kl = _topk_kl(base_logprobs, lora_logprobs, k=topk)

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
    topk = cfg.scoring.get("locality_topk", 50)
    print(f"\n[locality] Neighborhood drift (top-{topk} KL(base||lora), lower = less disruption):")
    print(summary.to_string(index=False))
    print("           kl_div_mean: top-k KL (primary); preservation_rate: top-1 match")
