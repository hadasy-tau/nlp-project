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
from .utils import batched, free_model, load_model, measurement_dtype


def build_eval_table(cfg, conditions: pd.DataFrame) -> pd.DataFrame:
    """One row per (fact, prompt). Training prompts are 'train'; paraphrases are the
    held-out probes ('paraphrase') that never appeared in LoRA training (pitfall 3).

    Train rows carry base_answer_rank/base_answer_logprob from score_base so the base
    variant can be checked against them (see _check_base_final_layer); paraphrases have
    no cached value.
    """
    rows = []
    for _, r in conditions.iterrows():
        rows.append({"fact_id": r["fact_id"], "condition": r["condition"],
                     "prompt": r["prompt"], "prompt_type": "train",
                     "answer_token_id": r["answer_token_id"],
                     "base_answer_rank": r.get("base_answer_rank", float("nan")),
                     "base_answer_logprob": r.get("base_answer_logprob", float("nan"))})
        if cfg.analysis.eval_paraphrases:
            val = r.get("paraphrases", None)
            paras = list(val) if val is not None and hasattr(val, "__iter__") else []
            for p in paras:
                rows.append({"fact_id": r["fact_id"], "condition": r["condition"],
                             "prompt": str(p).rstrip(), "prompt_type": "paraphrase",
                             "answer_token_id": r["answer_token_id"],
                             "base_answer_rank": float("nan"),
                             "base_answer_logprob": float("nan")})
    return pd.DataFrame(rows)


def _check_base_final_layer(frame: pd.DataFrame, eval_df: pd.DataFrame,
                            allow_reconcile: bool = False,
                            logprob_atol: float = 0.05,
                            latent_threshold: float = -5.0) -> pd.DataFrame:
    """Assert analyze's base re-scoring agrees with score_base on the thresholds the
    condition split is built from: rank==1, rank<=5 and the latent log-prob gate.

    Raw rank and log-prob equality is deliberately not asserted; deep-tail ranks and
    the last few log-prob digits drift with batch shape at any precision without
    moving a fact across a threshold. logprob_atol is a structural-breakage guard,
    not a precision test. Set allow_reconcile to overwrite instead of aborting.
    """
    cache = eval_df.loc[eval_df["prompt_type"] == "train",
                        ["prompt_idx", "base_answer_rank", "base_answer_logprob"]].dropna()
    if cache.empty:
        return frame
    cache = cache.set_index("prompt_idx")

    n_layers = frame["layer"].max()
    final_mask = frame["layer"] == n_layers
    has_cache = frame["prompt_idx"].isin(cache.index)
    target = final_mask & has_cache
    if not target.any():
        return frame

    measured_rank = frame.loc[target, "answer_rank"]
    measured_lp = frame.loc[target, "answer_logprob"]
    cached_rank = frame.loc[target, "prompt_idx"].map(cache["base_answer_rank"])
    cached_lp = frame.loc[target, "prompt_idx"].map(cache["base_answer_logprob"])
    n_checked = int(target.sum())

    top1_flip = (measured_rank == 1) != (cached_rank == 1)
    top5_flip = (measured_rank <= 5) != (cached_rank <= 5)
    gate_flip = (measured_lp > latent_threshold) != (cached_lp > latent_threshold)
    lp_gap = (measured_lp - cached_lp).abs()
    lp_broken = lp_gap > logprob_atol
    decisive = top1_flip | top5_flip | gate_flip | lp_broken

    # Diagnostic only; small nonzero drift is expected.
    drift = (measured_rank - cached_rank).abs()
    n_drift = int((drift > 0).sum())
    max_drift = int(drift.max()) if n_checked else 0
    max_lp_gap = float(lp_gap.max()) if n_checked else 0.0

    if not decisive.any():
        note = (f"; {n_drift} row(s) differ in exact rank by <={max_drift} deep in the "
                "tail, which no threshold depends on") if n_drift else ""
        print(f"[analyze] base final-layer check OK: {n_checked} rows agree with "
              f"score_base on rank==1, rank<=5 and the logprob>{latent_threshold:g} gate "
              f"(max |dlp|={max_lp_gap:.2e}, structural guard {logprob_atol:g}){note}.")
        return frame

    # Both lenses are checked per prompt, so report distinct prompts alongside rows.
    offenders = (frame.loc[target][decisive][["fact_id", "condition", "answer_rank"]]
                 .assign(score_base_rank=cached_rank[decisive].to_numpy(),
                         dlogprob=lp_gap[decisive].to_numpy())
                 .drop_duplicates())
    n_prompts, n_decisive = len(offenders), int(decisive.sum())
    detail = offenders.head(10).to_string(index=False)
    if not allow_reconcile:
        raise SystemExit(
            f"\n[analyze] BASE RE-SCORING DISAGREES WITH score_base ON A DECISION "
            f"THRESHOLD: {n_decisive}/{n_checked} final-layer rows "
            f"({n_prompts} distinct prompts).\n"
            f"  rank==1 flips: {int(top1_flip.sum())}   rank<=5 flips: "
            f"{int(top5_flip.sum())}   logprob>{latent_threshold:g} gate flips: "
            f"{int(gate_flip.sum())}   |dlogprob|>{logprob_atol:g} (structural): "
            f"{int(lp_broken.sum())}\n"
            "The condition split is not reproducible, so base accuracy will not match "
            "the selection rule (known != 1.000, latent != 0.000).\n"
            f"First offenders:\n{detail}\n\n"
            "Fix: --set model.inference_dtype=float32, or set "
            "analysis.allow_reconcile=true to overwrite instead."
        )

    print(f"[analyze] WARNING: reconciled {n_decisive}/{n_checked} rows "
          f"({n_prompts} prompts) that crossed a decision threshold vs score_base; "
          "the base column is now definitional rather than measured.")
    frame.loc[target, "answer_rank"] = cached_rank.astype(int).to_numpy()
    frame.loc[target, "answer_logprob"] = cached_lp.to_numpy()
    return frame


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
    """Vectorized per-prompt then per-group aggregation (fast on millions of rows).

    mean_first_layer averages only over prompts that ever reach top-1, so it is
    reported with its denominator (n_first_layer) and frac_never_top1; for a paired
    per-fact comparison use stats.py's Wilcoxon on _paired_first_layer.
    """
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
             n_first_layer=("first_layer", "count"),   # denominator of mean_first_layer
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
    joined = final.merge(
        base[merge_keys + ["final_accuracy", "mean_first_layer", "n_first_layer"]],
        on=merge_keys, suffixes=("_final", "_base"))
    joined["acc_gain"] = joined["final_accuracy_final"] - joined["final_accuracy_base"]
    joined["layer_shift"] = joined["mean_first_layer_final"] - joined["mean_first_layer_base"]

    print("\n[analyze] === Key findings (base → final, logit lens) ===")
    print("  n = prompts each mean first-layer is averaged over; unequal n means the "
          "shift is not like-for-like.")
    logit = joined[joined["lens"] == "logit"].sort_values(["prompt_type", "condition"])
    for _, row in logit.iterrows():
        tag = f"{row['condition']:9s} / {row['prompt_type']:10s}"
        acc_b = row["final_accuracy_base"]
        acc_f = row["final_accuracy_final"]
        gain = row["acc_gain"]
        shift = row["layer_shift"]
        n_b, n_f = int(row["n_first_layer_base"]), int(row["n_first_layer_final"])
        warn = "  <-- n differs >2x, shift not comparable" \
            if min(n_b, n_f) and max(n_b, n_f) > 2 * min(n_b, n_f) else ""
        print(f"  {tag}  acc {acc_b:.3f} → {acc_f:.3f}  (Δ={gain:+.3f})  "
              f"first-layer shift {shift:+.1f}  (n={n_b}→{n_f}){warn}")

    # Highlight the unknown vs known paraphrase contrast.
    # (Synthetic facts have no CounterFact paraphrase prompts, so that condition is absent here.)
    para = logit[logit["prompt_type"] == "paraphrase"].set_index("condition")
    if "unknown" in para.index and "known" in para.index:
        unk_gain = para.loc["unknown", "acc_gain"]
        kno_gain = para.loc["known", "acc_gain"]
        unk_base = para.loc["unknown", "final_accuracy_base"]
        kno_base = para.loc["known", "final_accuracy_base"]
        print(f"\n  *** Paraphrase generalisation: "
              f"unknown Δ={unk_gain:+.3f} (base {unk_base:.3f}) vs "
              f"known Δ={kno_gain:+.3f} (base {kno_base:.3f}) ***")
        if unk_gain >= kno_gain * 0.7:
            print("      → Unknown facts generalise nearly as well as known ones,")
            print("        suggesting LoRA elicits latent knowledge rather than memorising prompts.")


def run_analysis(cfg, tokenizer, conditions: pd.DataFrame, device) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_table(cfg, conditions).reset_index(names="prompt_idx")
    print(f"[analyze] {len(eval_df)} eval prompts "
          f"({(eval_df['prompt_type'] == 'paraphrase').sum()} held-out paraphrases)")

    # The BASE model's tuned lens is used for every variant (see lenses.py docstring).
    # Critical pass: must reproduce score_base, asserted below.
    base_model = load_model(cfg, device=device,
                            dtype=measurement_dtype(cfg, critical=True))
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
            model = load_model(cfg, device=device, adapter_path=adapter,
                               dtype=measurement_dtype(cfg))
        frame = _run_variant(cfg, model, tokenizer, eval_df, tuned_lens, device,
                             label, step)
        if label == "base":
            frame = _check_base_final_layer(
                frame, eval_df, cfg.analysis.get("allow_reconcile", False),
                cfg.analysis.get("logprob_atol", 0.05),
                cfg.data.get("latent_logprob_threshold", -5.0))
        frames.append(frame)
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
