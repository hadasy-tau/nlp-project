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

    Train rows carry the fact's base_answer_rank/base_answer_logprob through from
    score_base (see conditions.py) — the values that actually *defined* known/
    latent/unknown — so the base variant's final-layer numbers can be checked
    against them rather than assumed to agree (see _check_base_final_layer).
    Paraphrases have no such cached value (score_base never scored them).
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
    """Assert that analyze's base-variant re-scoring agrees with score_base.

    The base model is scored twice over the same prompts: once in score_base
    (which *defines* known/latent/unknown) and once here. Both lenses read the
    model's true final logits at the final layer (layerwise_answer_metrics uses
    `final_logits` directly at layer == n_layers, not a lens approximation), so
    in exact arithmetic the two passes must agree exactly.

    Under fp16 they didn't. score_base batches the full ~17.6k-fact pool at
    cfg.scoring.batch_size while analyze re-scores a smaller, differently-ordered
    subset at cfg.analysis.batch_size, and fp16 matmul/attention kernels are not
    bit-identical across tensor shapes. Near logit magnitude ~16 the fp16 spacing
    is ~0.016, so any fact whose top-1/top-2 margin is finer than that can flip
    between the passes. That made "known" (base top-1 correct *by definition*)
    report 98.8% base accuracy instead of 100%, and "latent" (defined as *not*
    top-1) report 0.2% instead of 0.0% — and every offender was rank 2 exactly,
    never rank 3+, which is the signature of a near-tie flip rather than a bug.

    Running the measurement passes in fp32 (model.inference_dtype: float32, plus
    disable_tf32) widens the spacing to ~1e-6 and the decisive disagreement
    disappears. So this is now an assertion, not a repair: it proves on every run
    that the two passes agree, instead of papering over the fact that they don't.
    Set analysis.allow_reconcile: true to fall back to the old overwrite behaviour
    when deliberately running the bulk passes in fp16.

    What "agree" means here is deliberately narrow. Exact rank equality across the
    whole 50k vocabulary is NOT the invariant and must not be asserted: deep in the
    tail thousands of tokens sit within ~1e-7 log-prob of each other, so a rank of
    2739 vs 2740 differs by whether one single token happened to land above or below
    the answer. That is irreducible at any precision and irrelevant — nothing in this
    pipeline thresholds on rank 2739. (An fp32 dev run produced exactly that: two
    facts off by one at ranks 371 and 2739, both harmless.)

    The invariants that DO matter are the thresholds the pipeline actually applies:
      * rank == 1              — defines `known`, and drives final_accuracy and
                                 every first-layer-of-appearance
      * rank <= 5              — defines the latent/unknown boundary
      * logprob > latent_threshold — the secondary latent gate (default -5)
    A disagreement on any of those is fatal.

    The same reasoning applies to the log-prob as to the rank, and for the same
    reason: assert the threshold, not the raw value. Forward-pass noise through a
    deep stack amplifies fp32's ~1e-7 relative error to ~1e-3 absolute on a
    final log-prob (observed: 1.6e-3 on a 12-layer CPU dev run), which is
    invisible at the 2 decimal places these values are ever reported to and
    cannot move a fact across the -5 gate unless it was already sitting on it.
    `logprob_atol` is therefore NOT a precision assertion — it is a loose guard
    (default 0.05) against structural breakage: a gap that large means the two
    passes scored different prompts or different answer tokens, not that the
    arithmetic drifted.

    Exact-rank drift is reported as a diagnostic only, because a *large* drift
    would indicate something worse than tail noise even though ±1 is expected.
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

    # Decisive: the thresholds the pipeline actually applies (see docstring).
    top1_flip = (measured_rank == 1) != (cached_rank == 1)
    top5_flip = (measured_rank <= 5) != (cached_rank <= 5)
    gate_flip = (measured_lp > latent_threshold) != (cached_lp > latent_threshold)
    lp_gap = (measured_lp - cached_lp).abs()
    lp_broken = lp_gap > logprob_atol           # structural guard, not a precision test
    decisive = top1_flip | top5_flip | gate_flip | lp_broken

    # Diagnostic only: exact-rank drift, which is expected to be small and nonzero.
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

    # One row per (prompt, lens) is checked — both lenses read the model's true final
    # logits at layer n_layers, so a flipped prompt shows up once per lens. Report
    # distinct prompts too, otherwise the counts look inflated.
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
            "The same model over the same prompts crossed a threshold the condition split "
            "is built from, so the split is not reproducible and base-column accuracy will "
            "not match the selection rule (known != 1.000, latent != 0.000).\n"
            f"First offenders:\n{detail}\n\n"
            "Fix: run the measurement passes in fp32 —\n"
            "    --set model.inference_dtype=float32\n"
            "(and make sure disable_tf32() runs; it is called from run.py:main).\n"
            "To accept fp16 and overwrite the disagreeing rows with score_base's values "
            "instead, set analysis.allow_reconcile=true — but then base and LoRA columns "
            "are measured at different effective precisions, which must be disclosed."
        )

    print(f"[analyze] WARNING: reconciled {n_decisive}/{n_checked} base final-layer rows "
          f"({n_prompts} distinct prompts) that crossed a decision threshold vs score_base "
          "(analysis.allow_reconcile=true). The base column "
          "is now definitional rather than measured; LoRA variants are unaffected and "
          "remain measured at the bulk precision.")
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

    `mean_first_layer` is an average over *only* the prompts that ever reach top-1;
    prompts that never do are NaN and dropped by `.mean()`. That denominator is not
    optional context — for base/unknown/train it was 7 prompts against the LoRA
    variant's 396, which is what made the paper's "+1.2 layers later" shift a
    comparison between non-comparable populations. `n_first_layer` (the actual
    denominator) and `frac_never_top1` are therefore reported alongside it, and any
    cross-variant layer delta must be read against them. For the correct paired
    per-fact comparison, use stats.py's Wilcoxon on `_paired_first_layer`.
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
    print("  n= is the population each side's mean first-layer is averaged over; a layer "
          "shift\n  between wildly unequal n is not a like-for-like comparison "
          "(see summarize docstring).")
    logit = joined[joined["lens"] == "logit"].sort_values(["prompt_type", "condition"])
    for _, row in logit.iterrows():
        tag = f"{row['condition']:9s} / {row['prompt_type']:10s}"
        acc_b = row["final_accuracy_base"]
        acc_f = row["final_accuracy_final"]
        gain = row["acc_gain"]
        shift = row["layer_shift"]
        n_b, n_f = int(row["n_first_layer_base"]), int(row["n_first_layer_final"])
        warn = "  <-- populations differ >2x, shift not comparable" \
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
    # The base variant is a "critical" pass: it must reproduce score_base exactly
    # (asserted by _check_base_final_layer below).
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
