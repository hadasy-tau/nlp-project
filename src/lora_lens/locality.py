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

Ground-truth correctness: CounterFact's neighborhood_prompts are constructed so
their correct answer is typically the *same* object as the original edited
fact's true answer (they test whether editing that fact also corrupted other
facts sharing that answer). Each expanded row carries the original fact's
answer_token_id through, so base_correct/lora_correct record whether each
model's top-1 actually matches the true answer — not just whether base and
LoRA agree with each other. This distinguishes a flip from correct to wrong
from a flip between two already-wrong guesses.

Runs across every saved checkpoint (same set used by patching.py), so
locality.csv gains a training-step dimension.

Secondary diagnostics (both computed from the same base/lora log-prob tensors
already held per batch, so cost is negligible on top of the existing pass):
  - escaped_mass: the fraction of LoRA's probability mass that falls *outside*
    the base model's top-k support. Distinguishes "LoRA became confident about
    one new token entirely outside the top-k" (high escaped mass) from "broad
    redistribution within the top-k" (low escaped mass, high KL) — both can
    otherwise produce a similarly large top-k KL after renormalization.
  - kl_div_full: KL(base || lora) over the *entire* vocabulary, with no
    renormalization needed since both are already valid distributions over the
    same support. This trades the top-k renormalization ambiguity for long-tail
    fp16 noise sensitivity, so it is reported as a secondary check that the
    top-k result isn't an artifact of renormalization, not as the primary metric.

Output: <output_dir>/results/locality.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .training import list_checkpoints
from .utils import batched, free_model, gather_last, load_model, measurement_dtype


def _topk_kl(base_logprobs: torch.Tensor, lora_logprobs: torch.Tensor,
            k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL(base || lora) over the top-k tokens under the base distribution, plus
    the top-k index tensor and LoRA's escaped probability mass (see module
    docstring for why both secondary diagnostics matter).

    Both log-prob vectors are sliced to the top-k base indices and renormalized
    over that support before computing KL, yielding a valid probability-simplex
    comparison on the tokens that carry the bulk of the base model's probability
    mass.
    """
    topk_vals, topk_idx = base_logprobs.topk(k, dim=-1)            # [B, k]
    lora_topk_raw = lora_logprobs.gather(1, topk_idx)               # [B, k]
    base_topk = topk_vals - torch.logsumexp(topk_vals, dim=-1, keepdim=True)
    lora_topk = lora_topk_raw - torch.logsumexp(lora_topk_raw, dim=-1, keepdim=True)
    kl = F.kl_div(lora_topk, base_topk, reduction="none", log_target=True).sum(dim=-1)
    escaped_mass = 1.0 - lora_topk_raw.exp().sum(dim=-1)
    return kl, topk_idx, escaped_mass.clamp(min=0.0)


def _full_kl(base_logprobs: torch.Tensor, lora_logprobs: torch.Tensor) -> torch.Tensor:
    """KL(base || lora) over the entire vocabulary — no renormalization needed
    since both inputs are already valid distributions over the same support.
    Secondary metric: substantiates that the top-k result isn't purely a
    renormalization artifact, at the cost of long-tail fp16 noise sensitivity.
    """
    return F.kl_div(lora_logprobs, base_logprobs, reduction="none", log_target=True).sum(dim=-1)


def _score_one_checkpoint(base_model, lora_model, expanded: pd.DataFrame, tokenizer,
                          cfg, device, variant: str, step: int) -> pd.DataFrame:
    """Score every expanded neighborhood prompt for one LoRA checkpoint; return a
    dataframe with base/LoRA top-1, KL, and ground-truth correctness columns."""
    idx = list(range(len(expanded)))
    topk = cfg.scoring.get("locality_topk", 50)
    base_top1s, lora_top1s, kl_divs, kl_divs_full, escaped_masses = [], [], [], [], []

    for chunk in tqdm(list(batched(idx, cfg.scoring.batch_size)), desc=f"locality[{variant}]"):
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
        kl, _, escaped_mass = _topk_kl(base_logprobs, lora_logprobs, k=topk)
        kl_full = _full_kl(base_logprobs, lora_logprobs)

        base_top1s.append(base_logprobs.argmax(dim=-1).cpu())
        lora_top1s.append(lora_logprobs.argmax(dim=-1).cpu())
        kl_divs.append(kl.cpu())
        kl_divs_full.append(kl_full.cpu())
        escaped_masses.append(escaped_mass.cpu())

    frame = expanded.copy()
    frame["variant"] = variant
    frame["step"] = step
    frame["base_top1"] = torch.cat(base_top1s).numpy()
    frame["lora_top1"] = torch.cat(lora_top1s).numpy()
    frame["preserved"] = frame["base_top1"] == frame["lora_top1"]
    ans_ids = frame["answer_token_id"].to_numpy()
    frame["base_correct"] = frame["base_top1"].to_numpy() == ans_ids
    frame["lora_correct"] = frame["lora_top1"].to_numpy() == ans_ids
    frame["kl_div"] = torch.cat(kl_divs).numpy()
    frame["kl_div_full"] = torch.cat(kl_divs_full).numpy()
    frame["escaped_mass"] = torch.cat(escaped_masses).numpy()
    return frame


def run_locality_scoring(cfg, tokenizer, conditions: pd.DataFrame, device) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list_checkpoints(cfg)
    if not checkpoints:
        raise SystemExit("[locality] No trained adapters found — run train_lora first.")

    # Expand neighborhood prompts — one row per prompt per fact. Carry the
    # original fact's answer_token_id through so ground-truth correctness can be
    # scored, not just base/LoRA agreement.
    rows_exp = []
    for _, r in conditions.iterrows():
        val = r.get("neighborhood_prompts", None)
        prompts = list(val) if val is not None and hasattr(val, "__iter__") else []
        for p in prompts:
            rows_exp.append({"fact_id": r["fact_id"], "condition": r["condition"],
                             "prompt": str(p).rstrip(),
                             "answer_token_id": r["answer_token_id"]})

    if not rows_exp:
        print("[locality] No neighborhood prompts found — skipping locality scoring.\n"
              "           Add 'neighborhood: neighborhood_prompts' under data.fields in "
              "your config.")
        return

    expanded = pd.DataFrame(rows_exp)
    print(f"[locality] {len(expanded)} neighborhood prompts across "
          f"{conditions['condition'].value_counts().to_dict()}")

    log_path = Path(cfg.output_dir) / "lora" / "training_log.csv"
    final_step = int(pd.read_csv(log_path)["step"].max()) if log_path.exists() else -1

    base_model = load_model(cfg, device=device, dtype=measurement_dtype(cfg))

    all_frames = []
    for label, adapter_path in checkpoints:
        step = final_step if label == "final" else int(label.split("_")[1])
        lora_model = load_model(cfg, device=device, adapter_path=adapter_path,
                                dtype=measurement_dtype(cfg))
        all_frames.append(_score_one_checkpoint(
            base_model, lora_model, expanded, tokenizer, cfg, device, label, step))
        free_model(lora_model)

    free_model(base_model)

    result = pd.concat(all_frames, ignore_index=True)
    result.to_csv(results_dir / "locality.csv", index=False)

    final = result[result["variant"] == "final"].copy()
    final["acc_drop"] = final["base_correct"] & ~final["lora_correct"]
    summary = (final.groupby("condition")
               .agg(kl_div_mean=("kl_div", "mean"),
                    kl_div_full_mean=("kl_div_full", "mean"),
                    escaped_mass_mean=("escaped_mass", "mean"),
                    preservation_rate=("preserved", "mean"),
                    accuracy_drop_rate=("acc_drop", "mean"),
                    n_prompts=("preserved", "count"))
               .reset_index())
    topk = cfg.scoring.get("locality_topk", 50)
    print(f"\n[locality] Neighborhood drift at final checkpoint "
          f"(top-{topk} KL(base||lora), lower = less disruption):")
    print(summary.to_string(index=False))
    print("           kl_div_mean: top-k KL (primary); kl_div_full_mean: full-vocab KL "
          "(secondary, no renormalization); escaped_mass_mean: fraction of LoRA's "
          "probability mass outside the base top-k (high = confident jump to a new "
          "token, not broad top-k scrambling); preservation_rate: top-1 match; "
          "accuracy_drop_rate: base-correct facts LoRA broke")
