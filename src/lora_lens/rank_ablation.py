"""Rank ablation: train LoRA at different ranks and compare layer-wise metrics.

For each rank r in cfg.rank_ablation.ranks, we train a LoRA adapter (keeping
alpha/r constant via alpha_multiplier) and run the full layerwise analysis on the
final checkpoint. This lets you see whether the late-layer shift is rank-sensitive
and whether lower-rank adapters move knowledge to earlier layers.

Outputs (under <output_dir>/results/):
  rank_ablation_layerwise.parquet — long-format, same schema as layerwise.parquet
                                    but with an additional 'rank' column
  rank_ablation_summary.csv       — per (rank, condition, prompt_type, lens)
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from .analysis import _check_base_final_layer, _run_variant, build_eval_table, summarize
from .lenses import load_tuned_lens
from .training import train_lora
from .utils import free_model, load_model, measurement_dtype


def run_rank_ablation(cfg, tokenizer, conditions: pd.DataFrame, device) -> None:
    if not cfg.rank_ablation.get("enabled", True):
        print("[rank_ablation] disabled in config, skipping.")
        return

    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ranks = list(cfg.rank_ablation.ranks)
    alpha_mult = int(cfg.rank_ablation.get("alpha_multiplier", 2))
    ablation_epochs = int(cfg.rank_ablation.get("epochs", cfg.lora.epochs))

    eval_df = build_eval_table(cfg, conditions).reset_index(names="prompt_idx")

    # Tuned lens is rank-independent — load once from the base model.
    base_model = load_model(cfg, device=device,
                            dtype=measurement_dtype(cfg, critical=True))
    tuned_lens = load_tuned_lens(cfg, base_model, device)
    base_frame = _run_variant(cfg, base_model, tokenizer, eval_df, tuned_lens, device,
                               "base", 0)
    base_frame = _check_base_final_layer(
        base_frame, eval_df, cfg.analysis.get("allow_reconcile", False),
        cfg.analysis.get("logprob_atol", 0.05),
        cfg.data.get("latent_logprob_threshold", -5.0))
    base_frame["rank"] = 0
    free_model(base_model)

    all_frames = [base_frame]
    for r in ranks:
        print(f"\n=== rank_ablation: r={r} (alpha={r * alpha_mult}, epochs={ablation_epochs}) ===")

        # Per-rank config: separate output dir, overridden lora params.
        r_cfg = copy.deepcopy(cfg)
        r_cfg["output_dir"] = str(Path(cfg.output_dir) / f"rank_r{r}")
        r_cfg["lora"]["r"] = r
        r_cfg["lora"]["alpha"] = r * alpha_mult
        r_cfg["lora"]["epochs"] = ablation_epochs
        # One checkpoint per rank is enough for ablation — skip intermediate saves.
        r_cfg["lora"]["checkpoint_every"] = 999999

        train_lora(r_cfg, tokenizer, conditions, device)

        adapter = Path(r_cfg.output_dir) / "lora" / "final"
        lora_model = load_model(cfg, device=device, adapter_path=adapter,
                                dtype=measurement_dtype(cfg))
        frame = _run_variant(cfg, lora_model, tokenizer, eval_df, tuned_lens, device,
                             f"r{r}", r)
        free_model(lora_model)
        frame["rank"] = r
        all_frames.append(frame)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_parquet(results_dir / "rank_ablation_layerwise.parquet", index=False)

    summary = summarize(combined)
    summary["rank"] = summary["variant"].apply(
        lambda v: 0 if v == "base" else int(v.lstrip("r")))
    summary.to_csv(results_dir / "rank_ablation_summary.csv", index=False)

    print("\n[rank_ablation] Final-layer accuracy by rank (logit lens, train prompts):")
    show = (summary[(summary["lens"] == "logit") & (summary["prompt_type"] == "train")]
            [["rank", "condition", "final_accuracy", "mean_final_logprob", "mean_first_layer"]]
            .sort_values(["condition", "rank"]))
    print(show.to_string(index=False))

    print("\n[rank_ablation] Paraphrase generalisation by rank (logit lens):")
    show_para = (summary[(summary["lens"] == "logit") & (summary["prompt_type"] == "paraphrase")]
                 [["rank", "condition", "final_accuracy", "mean_first_layer"]]
                 .sort_values(["condition", "rank"]))
    print(show_para.to_string(index=False))
    print(f"\n[rank_ablation] Full results in {results_dir}")
