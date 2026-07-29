"""Layer-wise lens analysis of base model + every LoRA checkpoint.

Outputs (under <output_dir>/results/):
  layerwise.parquet — long format: variant, step, fact_id, condition, prompt_type,
                      lens, layer, answer_logprob, answer_rank, in_top_k...
  summary.csv       — per (variant, condition, prompt_type, lens): final-layer
                      accuracy, mean answer logprob, mean first layer of appearance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .lenses import layerwise_answer_metrics, load_tuned_lens
from .training import list_checkpoints
from .utils import batched, free_model, load_model


def build_eval_table(cfg, conditions: pd.DataFrame) -> pd.DataFrame:
    """One row per (fact, prompt). Training prompts are 'train'; paraphrases are the
    held-out probes ('paraphrase') that never appeared in LoRA training (pitfall 3)."""
    rows = []
    for _, r in conditions.iterrows():
        rows.append({"fact_id": r["fact_id"], "condition": r["condition"],
                     "prompt": r["prompt"], "prompt_type": "train",
                     "answer_token_id": r["answer_token_id"]})
        if cfg.analysis.eval_paraphrases:
            for p in r["paraphrases"]:
                rows.append({"fact_id": r["fact_id"], "condition": r["condition"],
                             "prompt": str(p).rstrip(), "prompt_type": "paraphrase",
                             "answer_token_id": r["answer_token_id"]})
    return pd.DataFrame(rows)


@torch.no_grad()
def _run_variant(cfg, model, tokenizer, eval_df: pd.DataFrame, tuned_lens, device,
                 variant: str, step: int) -> pd.DataFrame:
    all_records = []
    idx = list(range(len(eval_df)))
    for chunk in tqdm(list(batched(idx, cfg.analysis.batch_size)), desc=f"lens[{variant}]"):
        sub = eval_df.iloc[chunk]
        enc = tokenizer(sub["prompt"].tolist(), return_tensors="pt", padding=True).to(device)
        ans = torch.tensor(sub["answer_token_id"].tolist(), device=device)
        recs = layerwise_answer_metrics(model, enc, ans, tuned_lens,
                                        cfg.lens.use_logit, list(cfg.analysis.top_k))
        for rec in recs:
            r = sub.iloc[rec.pop("row")]
            rec.update({"variant": variant, "step": step, "fact_id": r["fact_id"],
                        "condition": r["condition"], "prompt_type": r["prompt_type"],
                        "prompt_idx": int(r["prompt_idx"])})
            all_records.append(rec)
    return pd.DataFrame(all_records)


def summarize(layerwise: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-prompt then per-group aggregation (fast on millions of rows)."""
    n_layers = layerwise["layer"].max()
    keys = ["variant", "step", "condition", "prompt_type", "lens", "prompt_idx"]

    # Exactly one final-layer row per (variant, lens, prompt).
    final = layerwise[layerwise["layer"] == n_layers].copy()
    final["final_correct"] = final["answer_rank"] == 1
    per_prompt = final[keys + ["final_correct", "answer_logprob"]]

    # First layer of appearance = earliest layer where the answer is top-1 (NaN if never).
    first = (layerwise[layerwise["answer_rank"] == 1]
             .groupby(keys)["layer"].min().rename("first_layer").reset_index())
    per_prompt = per_prompt.merge(first, on=keys, how="left")

    return (
        per_prompt.groupby(keys[:-1])
        .agg(final_accuracy=("final_correct", "mean"),
             mean_final_logprob=("answer_logprob", "mean"),
             mean_first_layer=("first_layer", "mean"),
             frac_never_top1=("first_layer", lambda s: s.isna().mean()),
             n_prompts=("final_correct", "size"))
        .reset_index()
        .sort_values(["lens", "condition", "prompt_type", "step"])
    )


def report_highlights(summary: pd.DataFrame) -> None:
    """Print the most actionable cross-condition comparison from the final checkpoint.

    The key finding is whether 'unknown' (real facts the base model gets wrong) generalises
    to paraphrases better than 'synthetic' (invented facts). A large gap there means LoRA
    is eliciting latent knowledge rather than purely injecting new associations.
    """
    final = summary[summary["variant"] == "final"].copy()
    base = summary[summary["variant"] == "base"].copy()
    if final.empty or base.empty:
        return

    merge_keys = ["condition", "prompt_type", "lens"]
    joined = final.merge(base[merge_keys + ["final_accuracy", "mean_first_layer"]],
                         on=merge_keys, suffixes=("_final", "_base"))
    joined["acc_gain"] = joined["final_accuracy_final"] - joined["final_accuracy_base"]
    joined["layer_shift"] = joined["mean_first_layer_final"] - joined["mean_first_layer_base"]

    print("\n[analyze] === Key findings (base → final, logit lens) ===")
    logit = joined[joined["lens"] == "logit"].sort_values(["prompt_type", "condition"])
    for _, row in logit.iterrows():
        tag = f"{row['condition']:9s} / {row['prompt_type']:10s}"
        acc_b = row["final_accuracy_base"]
        acc_f = row["final_accuracy_final"]
        gain = row["acc_gain"]
        shift = row["layer_shift"]
        print(f"  {tag}  acc {acc_b:.3f} → {acc_f:.3f}  (Δ={gain:+.3f})  "
              f"first-layer shift {shift:+.1f}")

    # Highlight the unknown vs synthetic paraphrase contrast.
    para = logit[logit["prompt_type"] == "paraphrase"].set_index("condition")
    if "unknown" in para.index and "synthetic" in para.index:
        unk_gain = para.loc["unknown", "acc_gain"]
        syn_gain = para.loc["synthetic", "acc_gain"]
        ratio = unk_gain / syn_gain if syn_gain > 0 else float("inf")
        print(f"\n  *** Paraphrase generalisation: unknown Δ={unk_gain:+.3f} vs "
              f"synthetic Δ={syn_gain:+.3f} (ratio {ratio:.1f}x) ***")
        if unk_gain > syn_gain * 2:
            print("      → LoRA is primarily ELICITING latent knowledge for unknown facts,")
            print("        not just memorising prompts (unlike synthetic).")


def run_analysis(cfg, tokenizer, conditions: pd.DataFrame, device) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_table(cfg, conditions).reset_index(names="prompt_idx")
    print(f"[analyze] {len(eval_df)} eval prompts "
          f"({(eval_df['prompt_type'] == 'paraphrase').sum()} held-out paraphrases)")

    # The BASE model's tuned lens is used for every variant (see lenses.py docstring).
    base_model = load_model(cfg, device=device)
    tuned_lens = load_tuned_lens(cfg, base_model, device)

    log_path = Path(cfg.output_dir) / "lora" / "training_log.csv"
    final_step = int(pd.read_csv(log_path)["step"].max()) if log_path.exists() else -1

    variants: list[tuple[str, int, object]] = [("base", 0, None)]
    for label, path in list_checkpoints(cfg):
        step = final_step if label == "final" else int(label.split("_")[1])
        variants.append((label, step, path))
    if len(variants) == 1:
        print("[analyze] WARNING: no LoRA checkpoints found — analyzing base only.")

    frames = []
    for label, step, adapter in variants:
        if adapter is None:
            model = base_model
        else:
            model = load_model(cfg, device=device, adapter_path=adapter)
        frames.append(_run_variant(cfg, model, tokenizer, eval_df, tuned_lens, device,
                                   label, step))
        if adapter is not None:
            free_model(model)
    free_model(base_model)

    layerwise = pd.concat(frames, ignore_index=True)
    layerwise.to_parquet(results_dir / "layerwise.parquet", index=False)

    summary = summarize(layerwise)
    summary.to_csv(results_dir / "summary.csv", index=False)
    print("\n[analyze] Summary (final checkpoint + base):")
    show = summary[summary["variant"].isin(["base", "final"])]
    print(show.to_string(index=False))
    report_highlights(summary)
    print(f"\n[analyze] Full results in {results_dir}")
