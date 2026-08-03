"""Statistical rigor pass on top of existing pipeline outputs.

Pure post-processing of results already on disk (patching.csv, layerwise.parquet)
-- no new experiments, no retraining. Adds the significance/uncertainty
quantification the raw summary tables lack:

  - Bootstrap 95% CI (percentile method) on the median causal first-flip layer
    per condition, among facts that flip at all ("conditional on flipping" --
    see the McNemar section below for the separate flip-rate/appearance-rate
    comparison).
  - Mann-Whitney U test + rank-biserial effect size (with bootstrap CI) between
    adjacent conditions (latent vs. unknown, unknown vs. synthetic) on
    first-flip-layer distributions, among facts that flip at all. This tests
    whether one condition's first-flip layers tend to be stochastically
    earlier/later than the other's -- not specifically a difference in medians.
  - Fisher's exact test (with conditional-MLE odds-ratio 95% CI) on the
    flip-rate 2x2 contingency tables (e.g. how many of 100 facts per condition
    flip at all), since flip *rate* and flip *depth* are separate claims.
  - McNemar's test on the paired base-vs-final "ever reaches top-1" indicator
    (logit lens), per condition -- the appearance-*rate* analogue of the
    Wilcoxon depth test below, answering "does LoRA make more facts reach
    top-1 at all" rather than "how much earlier do they reach it".
  - Wilcoxon signed-rank test (paired, per-fact) on the logit-lens
    first-layer-of-appearance shift (base vs. final variant), per condition,
    in two variants:
      * "both_reached": restricted to facts that reach top-1 in both base and
        final (real overlap only, no censoring) -- the primary appearance-depth
        analysis.
      * "censored_sensitivity": every fact included, with never-reached facts
        censored at n_layers + 1 (worse than any real layer) -- a sensitivity
        check on the primary analysis, not a headline result on its own.
  - Holm correction of p-values within each test family (Mann-Whitney, Fisher,
    the two Wilcoxon variants, McNemar), reported alongside the raw p-value.

Statistical unit: fact_id (one training prompt per fact). Paraphrases
(prompt_type == "paraphrase") are never included in any test here; only
prompt_type == "train" rows are used, and prompt_idx is a 1:1 relabeling of
fact_id within that subset (asserted in _paired_first_layer).

Note on the rank-biserial sign convention: `_rank_biserial(u, n1, n2)` returns
positive when sample 1 (a) tends to have LARGER (later) values than sample 2
(b). This is the opposite of an earlier, buggy version of this function --
any effect size computed before this fix has the wrong sign (e.g. a previously
reported r=+0.49 for latent-vs-unknown should read r=-0.49, correctly
reflecting that latent flips earlier than unknown).

Output: <output_dir>/results/stats_summary.csv, a printed report, and two
LaTeX table snippets (depth-ordering table, appearance-rate/depth table).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.stats.contingency import odds_ratio as _odds_ratio
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
from statsmodels.stats.multitest import multipletests

COND_ORDER = ["known", "latent", "unknown", "synthetic"]
CONDITION_LABELS = {
    "known": "Known", "latent": "Latent (top-5)",
    "unknown": "Unknown", "synthetic": "Synthetic",
}
# Adjacent pairs in the hypothesized depth spectrum (known excluded -- it's not
# part of the causal patching comparison; see patching.py).
ADJACENT_PAIRS = [("latent", "unknown"), ("unknown", "synthetic")]


def _bootstrap_median_ci(values: np.ndarray, n_boot: int = 10_000, ci: float = 0.95,
                         seed: int = 0) -> tuple[float, float, float]:
    """Percentile-method bootstrap CI on the median of `values`."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    boot_medians = np.median(samples, axis=1)
    lo, hi = np.percentile(boot_medians, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(np.median(values)), float(lo), float(hi)


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation effect size from a Mann-Whitney U statistic
    (U computed for sample 1 (a) vs sample 2 (b), i.e. scipy's
    `mannwhitneyu(a, b).statistic`, which counts pairs where a_i > b_j).

    Positive r means sample a tends to have LARGER (later) first-flip layers
    than sample b; negative means a tends to be earlier. r in [-1, 1], 0 = no
    tendency. (See module docstring: this is the corrected sign convention.)
    """
    if n1 == 0 or n2 == 0:
        return float("nan")
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def _rank_biserial_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                      ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on the rank-biserial effect size: resample `a`
    and `b` independently (with replacement, at their observed sizes) and
    recompute Mann-Whitney U + the effect size on each draw."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    effects = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        u = sstats.mannwhitneyu(ra, rb, alternative="two-sided").statistic
        effects[i] = _rank_biserial(u, len(ra), len(rb))
    lo, hi = np.percentile(effects, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def _holm_adjust(rows: list[dict], p_key: str = "p_value") -> None:
    """In-place: add 'p_holm' to each row dict, Holm-corrected within this
    family (`rows`) only. NaN p-values pass through as NaN."""
    for r in rows:
        r["p_holm"] = float("nan")
    valid = [(i, r[p_key]) for i, r in enumerate(rows) if not np.isnan(r[p_key])]
    if not valid:
        return
    idx, pvals = zip(*valid)
    _, p_holm, _, _ = multipletests(pvals, method="holm")
    for i, p in zip(idx, p_holm):
        rows[i]["p_holm"] = float(p)


def _validate_flipped_column(df: pd.DataFrame) -> pd.DataFrame:
    """Explicitly parse patching.csv's 'flipped' column to bool, rejecting the
    `astype(bool)`-on-strings footgun (every non-empty string, including
    "False", is truthy) and failing loudly on missing values instead of
    silently coercing them."""
    col = df["flipped"]
    if col.isna().any():
        raise ValueError(f"patching.csv has {col.isna().sum()} missing 'flipped' values -- "
                         "cannot treat as a valid boolean without an explicit policy.")
    if col.dtype == bool:
        return df
    mapped = col.astype(str).str.strip().map({"True": True, "False": False})
    if mapped.isna().any():
        bad = col[mapped.isna()].unique()
        raise ValueError(f"patching.csv 'flipped' column has unrecognized values: {bad!r}")
    df = df.copy()
    df["flipped"] = mapped
    return df


def _assert_no_duplicates(df: pd.DataFrame, subset: list[str], label: str) -> None:
    dupes = df[df.duplicated(subset=subset, keep=False)]
    if not dupes.empty:
        raise ValueError(f"{label}: {len(dupes)} duplicated rows on {subset}; "
                         f"example:\n{dupes.head()}")


def _patching_first_flip_by_condition(patch_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Final-checkpoint first_flip_layer values per condition (NaN = never flipped,
    dropped for the layer-depth comparisons but kept for flip-rate ones)."""
    final = patch_df[(patch_df["layer"] == -1) & (patch_df["variant"] == "final")]
    return {c: final[final["condition"] == c]["first_flip_layer"].to_numpy()
            for c in final["condition"].unique()}


def _flip_contingency(patch_df: pd.DataFrame, cond_a: str, cond_b: str) -> np.ndarray | None:
    """2x2 [[flipped_a, not_flipped_a], [flipped_b, not_flipped_b]] table."""
    final = patch_df[(patch_df["layer"] == -1) & (patch_df["variant"] == "final")]
    a = final[final["condition"] == cond_a]["flipped"]
    b = final[final["condition"] == cond_b]["flipped"]
    if a.empty or b.empty:
        return None
    return np.array([[int(a.sum()), int((~a).sum())],
                     [int(b.sum()), int((~b).sum())]])


def _paired_first_layer(layerwise: pd.DataFrame, condition: str,
                        prompt_type: str = "train", censor: bool = True) -> pd.DataFrame:
    """Per-prompt first-layer-of-appearance (logit lens) for base vs. final,
    matched on prompt_idx so paired tests are genuine paired comparisons.

    Statistical unit is fact_id (see module docstring); the assertion below
    confirms prompt_idx <-> fact_id is a 1:1 mapping within this subset, so no
    paraphrase (or other) inflation of the paired sample can occur silently.

    Facts present for only one of the two variants (shouldn't happen given
    `build_eval_table` runs every variant over the same fact set, but guarded
    here anyway) are dropped before pivoting, so a NaN in the returned table
    always means "present but never reached top-1", never "no row at all".

    If censor=True (sensitivity analysis, see module docstring): facts that
    never reach top-1 in a variant are filled with n_layers + 1 (worse than
    any real layer), and the returned table has no NaNs.
    If censor=False (primary analysis): non-appearances are left as NaN.
    Callers wanting "first-appearance depth among facts reaching top-1 in
    both variants" should follow with `.dropna()`; callers wanting the
    "ever reaches top-1 at all" indicator should use `_paired_ever_reached`.
    """
    n_layers = layerwise["layer"].max()
    sub = layerwise[(layerwise["lens"] == "logit") & (layerwise["condition"] == condition) &
                    (layerwise["prompt_type"] == prompt_type) &
                    (layerwise["variant"].isin(["base", "final"]))]
    if sub.empty:
        return pd.DataFrame()

    # fact_id is the statistical unit here: build_eval_table emits exactly one
    # "train" row per fact_id (paraphrases are prompt_type == "paraphrase" and
    # excluded by the filter above), so prompt_idx <-> fact_id is already a
    # 1:1 mapping in this subset -- no paraphrase-inflation of the paired
    # sample. Deduplicate the long (one-row-per-layer) frame down to the
    # (variant, fact_id, prompt_idx) level before checking, since fact_id
    # legitimately repeats across layers within a variant.
    id_map = sub[["variant", "fact_id", "prompt_idx"]].drop_duplicates()
    assert id_map.groupby("variant")["fact_id"].apply(lambda s: s.is_unique).all(), (
        "expected a 1:1 fact_id <-> prompt_idx mapping per variant in the "
        f"prompt_type == {prompt_type!r} subset")

    all_idx = sub[["variant", "prompt_idx"]].drop_duplicates()
    by_variant = all_idx.groupby("variant")["prompt_idx"].apply(set)
    if not {"base", "final"}.issubset(by_variant.index):
        return pd.DataFrame()
    common = by_variant["base"] & by_variant["final"]
    all_idx = all_idx[all_idx["prompt_idx"].isin(common)]

    first = (sub[sub["answer_rank"] == 1]
            .groupby(["variant", "prompt_idx"])["layer"].min()
            .rename("first_layer").reset_index())
    first = all_idx.merge(first, on=["variant", "prompt_idx"], how="left")
    if censor:
        first["first_layer"] = first["first_layer"].fillna(n_layers + 1)
        return first.pivot(index="prompt_idx", columns="variant", values="first_layer").dropna()
    return first.pivot(index="prompt_idx", columns="variant", values="first_layer")


def _paired_ever_reached(layerwise: pd.DataFrame, condition: str,
                         prompt_type: str = "train") -> pd.DataFrame:
    """Per-prompt boolean 'ever reaches top-1' (logit lens) for base vs. final,
    for the McNemar paired appearance-rate test. Shares the fact_id/prompt_idx
    unit and base/final overlap guard with `_paired_first_layer`."""
    wide = _paired_first_layer(layerwise, condition, prompt_type, censor=False)
    if wide.empty or "base" not in wide.columns or "final" not in wide.columns:
        return pd.DataFrame()
    return ~wide.isna()


def _mcnemar_table(wide_bool: pd.DataFrame) -> np.ndarray:
    """Paired 2x2 table [[n11, n10], [n01, n00]] where n_ij = count with
    base ever-reached == (i == 0) and final ever-reached == (j == 0)."""
    n11 = int((wide_bool["base"] & wide_bool["final"]).sum())
    n10 = int((wide_bool["base"] & ~wide_bool["final"]).sum())
    n01 = int((~wide_bool["base"] & wide_bool["final"]).sum())
    n00 = int((~wide_bool["base"] & ~wide_bool["final"]).sum())
    return np.array([[n11, n10], [n01, n00]])


def _wilcoxon_report(diffs: np.ndarray, seed: int) -> dict | None:
    """Run Wilcoxon signed-rank on `diffs` (= final - base; negative means the
    answer appeared earlier after LoRA) and return the reporting fields:
    median diff (primary statistic, with bootstrap CI), mean diff (secondary),
    n_pairs, n_nonzero. Returns None if all diffs are zero or the test can't
    run (e.g. too few nonzero diffs for scipy's exact/normal approximation)."""
    diffs = np.asarray(diffs, dtype=float)
    n_pairs = len(diffs)
    n_nonzero = int(np.sum(diffs != 0))
    if n_nonzero == 0:
        return None
    try:
        res = sstats.wilcoxon(diffs)
    except ValueError:
        return None
    median_diff, ci_lo, ci_hi = _bootstrap_median_ci(diffs, seed=seed)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "n_pairs": n_pairs, "n_nonzero": n_nonzero,
            "median_diff": median_diff, "mean_diff": float(diffs.mean()),
            "ci_lo": ci_lo, "ci_hi": ci_hi}


def run_stats(cfg) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    patch_path = results_dir / "patching.csv"
    layerwise_path = results_dir / "layerwise.parquet"

    rows: list[dict] = []

    # -- Bootstrap CIs + Mann-Whitney + Fisher on patching.csv -----------------
    if patch_path.exists():
        patch_df = pd.read_csv(patch_path)
        patch_df = _validate_flipped_column(patch_df)
        _assert_no_duplicates(patch_df[patch_df["layer"] == -1],
                              ["variant", "condition", "fact_id"], "patching.csv (final rows)")
        by_cond = _patching_first_flip_by_condition(patch_df)

        print("\n[stats] Bootstrap 95% CI on median causal first-flip layer (final checkpoint),")
        print("        among examples that flipped at all (see McNemar section below for")
        print("        flip-rate/appearance-rate comparisons):")
        for cond in [c for c in COND_ORDER if c in by_cond]:
            median, lo, hi = _bootstrap_median_ci(by_cond[cond], seed=0)
            n = int((~np.isnan(by_cond[cond])).sum())
            print(f"  {CONDITION_LABELS[cond]:16s} median={median:.1f}  "
                  f"95% CI=[{lo:.1f}, {hi:.1f}]  (n_flipped={n})")
            rows.append({"type": "bootstrap_ci", "condition": cond,
                        "metric": "first_flip_layer_median_given_flipped", "value": median,
                        "ci_lo": lo, "ci_hi": hi, "n": n})

        # -- Mann-Whitney U + rank-biserial effect size, adjacent pairs --------
        print("\n[stats] Mann-Whitney U test on first-flip-layer distributions, among examples")
        print("        that flipped at all (adjacent conditions in the hypothesized depth")
        print("        spectrum). Tests whether one condition's first-flip layers tend to be")
        print("        stochastically earlier/later than the other's -- not specifically a")
        print("        difference in medians:")
        mw_rows = []
        for cond_a, cond_b in ADJACENT_PAIRS:
            a = by_cond.get(cond_a, np.array([]))
            a = a[~np.isnan(a)]
            b = by_cond.get(cond_b, np.array([]))
            b = b[~np.isnan(b)]
            if len(a) < 1 or len(b) < 1:
                continue
            res = sstats.mannwhitneyu(a, b, alternative="two-sided")
            effect = _rank_biserial(res.statistic, len(a), len(b))
            eff_lo, eff_hi = _rank_biserial_ci(a, b, seed=1)
            print(f"  {cond_a} (n={len(a)}) vs {cond_b} (n={len(b)}): "
                  f"U={res.statistic:.1f}  p={res.pvalue:.4g}  "
                  f"rank-biserial r={effect:.3f} [{eff_lo:.3f}, {eff_hi:.3f}]")
            mw_rows.append({"type": "mannwhitney", "comparison": f"{cond_a}_vs_{cond_b}",
                        "statistic": float(res.statistic), "p_value": float(res.pvalue),
                        "effect_size_rank_biserial": effect,
                        "effect_ci_lo": eff_lo, "effect_ci_hi": eff_hi,
                        "n1": len(a), "n2": len(b)})
        _holm_adjust(mw_rows)
        for r in mw_rows:
            print(f"    Holm-adjusted p ({r['comparison']}): {r['p_holm']:.4g}")
        rows.extend(mw_rows)

        # -- Fisher's exact test on flip-rate contingency tables ---------------
        print("\n[stats] Fisher's exact test on flip-rate (flipped-at-all) contingency tables")
        print("        (adjacent conditions), with conditional-MLE odds-ratio 95% CI:")
        fe_rows = []
        for cond_a, cond_b in ADJACENT_PAIRS:
            table = _flip_contingency(patch_df, cond_a, cond_b)
            if table is None:
                continue
            odds_ratio, p_value = sstats.fisher_exact(table)
            try:
                or_ci = _odds_ratio(table).confidence_interval(confidence_level=0.95)
                or_ci_lo, or_ci_hi = float(or_ci.low), float(or_ci.high)
            except (ValueError, ZeroDivisionError):
                or_ci_lo, or_ci_hi = float("nan"), float("nan")
            print(f"  {cond_a} {table[0].tolist()} vs {cond_b} {table[1].tolist()}: "
                  f"odds_ratio={odds_ratio:.3f} [{or_ci_lo:.3f}, {or_ci_hi:.3f}]  "
                  f"p={p_value:.4g}")
            fe_rows.append({"type": "fisher_exact", "comparison": f"{cond_a}_vs_{cond_b}",
                        "odds_ratio": odds_ratio, "p_value": p_value,
                        "odds_ratio_ci_lo": or_ci_lo, "odds_ratio_ci_hi": or_ci_hi,
                        "table": str(table.tolist())})
        _holm_adjust(fe_rows)
        for r in fe_rows:
            print(f"    Holm-adjusted p ({r['comparison']}): {r['p_holm']:.4g}")
        rows.extend(fe_rows)
    else:
        print(f"[stats] {patch_path} not found -- skipping patching-based tests.")

    # -- Wilcoxon (both variants) + McNemar on layerwise.parquet ---------------
    if layerwise_path.exists():
        layerwise = pd.read_parquet(layerwise_path)
        _assert_no_duplicates(layerwise, ["variant", "lens", "prompt_idx", "layer"],
                              "layerwise.parquet")

        # -- Primary B: Wilcoxon among facts reaching top-1 in both variants ---
        print("\n[stats] Wilcoxon signed-rank test on paired first-layer-of-appearance shift")
        print("        (base vs. final, logit lens, train prompts), restricted to facts that")
        print("        reach top-1 in BOTH variants (real overlap only, no censoring).")
        print("        final - base < 0 means the answer appeared earlier after LoRA:")
        both_rows = []
        for cond in COND_ORDER:
            wide = _paired_first_layer(layerwise, cond, censor=False)
            if wide.empty or "base" not in wide.columns or "final" not in wide.columns:
                continue
            wide = wide.dropna()
            if wide.empty:
                print(f"  {CONDITION_LABELS[cond]:16s}  no facts reach top-1 in both variants "
                      "-- skipped")
                continue
            diffs = (wide["final"] - wide["base"]).to_numpy()
            result = _wilcoxon_report(diffs, seed=3)
            if result is None:
                print(f"  {CONDITION_LABELS[cond]:16s}  all differences zero or test failed "
                      "-- skipped")
                continue
            print(f"  {CONDITION_LABELS[cond]:16s}  n_pairs={result['n_pairs']}  "
                  f"n_nonzero={result['n_nonzero']}  "
                  f"median_diff={result['median_diff']:+.1f} "
                  f"[{result['ci_lo']:+.1f}, {result['ci_hi']:+.1f}]  "
                  f"mean_diff={result['mean_diff']:+.2f}  "
                  f"W={result['statistic']:.1f}  p={result['p_value']:.4g}")
            both_rows.append({"type": "wilcoxon_both_reached", "condition": cond, **result})
        _holm_adjust(both_rows)
        for r in both_rows:
            print(f"    Holm-adjusted p ({r['condition']}): {r['p_holm']:.4g}")
        rows.extend(both_rows)

        # -- Sensitivity: Wilcoxon with never-reached censored -----------------
        print("\n[stats] [sensitivity analysis] Wilcoxon signed-rank test on the same paired")
        print("        shift, but censoring never-reached facts at n_layers + 1 instead of")
        print("        dropping them. final - base < 0 means earlier after LoRA:")
        sens_rows = []
        for cond in COND_ORDER:
            wide = _paired_first_layer(layerwise, cond, censor=True)
            if wide.empty or "base" not in wide.columns or "final" not in wide.columns:
                continue
            diffs = (wide["final"] - wide["base"]).to_numpy()
            result = _wilcoxon_report(diffs, seed=2)
            if result is None:
                print(f"  {CONDITION_LABELS[cond]:16s}  all differences zero or test failed "
                      "-- skipped")
                continue
            print(f"  {CONDITION_LABELS[cond]:16s}  n_pairs={result['n_pairs']}  "
                  f"n_nonzero={result['n_nonzero']}  "
                  f"median_diff={result['median_diff']:+.1f} "
                  f"[{result['ci_lo']:+.1f}, {result['ci_hi']:+.1f}]  "
                  f"mean_diff={result['mean_diff']:+.2f}  "
                  f"W={result['statistic']:.1f}  p={result['p_value']:.4g}")
            sens_rows.append({"type": "wilcoxon_censored_sensitivity", "condition": cond,
                             **result})
        _holm_adjust(sens_rows)
        for r in sens_rows:
            print(f"    Holm-adjusted p ({r['condition']}): {r['p_holm']:.4g}")
        rows.extend(sens_rows)

        # -- Primary A: McNemar on paired "ever reaches top-1" -----------------
        print("\n[stats] McNemar's test on paired appearance rate (does the fact EVER reach")
        print("        top-1 at any layer, base vs. final; logit lens, train prompts):")
        mcnemar_rows = []
        for cond in COND_ORDER:
            wide_bool = _paired_ever_reached(layerwise, cond)
            if wide_bool.empty:
                continue
            table = _mcnemar_table(wide_bool)
            n11, n10, n01, n00 = int(table[0, 0]), int(table[0, 1]), int(table[1, 0]), \
                int(table[1, 1])
            if n10 == 0 and n01 == 0:
                print(f"  {CONDITION_LABELS[cond]:16s}  table={table.tolist()}  "
                      "no discordant pairs -- McNemar undefined")
                mcnemar_rows.append({"type": "mcnemar", "condition": cond,
                            "statistic": float("nan"), "p_value": float("nan"),
                            "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                            "direction": "no discordant pairs"})
                continue
            res = _sm_mcnemar(table, exact=True)
            if n01 > n10:
                direction = "more facts newly reach top-1 after LoRA"
            elif n10 > n01:
                direction = "more facts stop reaching top-1 after LoRA"
            else:
                direction = "no net direction (n10 == n01)"
            print(f"  {CONDITION_LABELS[cond]:16s}  table={table.tolist()}  "
                  f"statistic={res.statistic:.1f}  p={res.pvalue:.4g}  ({direction})")
            mcnemar_rows.append({"type": "mcnemar", "condition": cond,
                        "statistic": float(res.statistic), "p_value": float(res.pvalue),
                        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                        "direction": direction})
        _holm_adjust(mcnemar_rows)
        for r in mcnemar_rows:
            print(f"    Holm-adjusted p ({r['condition']}): {r['p_holm']:.4g}")
        rows.extend(mcnemar_rows)
    else:
        print(f"[stats] {layerwise_path} not found -- skipping Wilcoxon/McNemar tests.")

    if not rows:
        print("[stats] No inputs found -- nothing to write.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "stats_summary.csv", index=False)
    print(f"\n[stats] Full results in {results_dir / 'stats_summary.csv'}")
    print_stats_latex_table(out)
    print_appearance_latex_table(out)


def print_stats_latex_table(stats_df: pd.DataFrame) -> None:
    """LaTeX snippet: bootstrap CIs + adjacent-pair significance tests (depth,
    conditional on flipping), ready to paste into the paper's statistics
    subsection."""
    ci = stats_df[stats_df["type"] == "bootstrap_ci"]
    mw = stats_df[stats_df["type"] == "mannwhitney"].set_index("comparison")
    fe = stats_df[stats_df["type"] == "fisher_exact"].set_index("comparison")

    # Plain ASCII in the printed header: Windows consoles default to cp1252,
    # which cannot encode box-drawing characters and would crash the print.
    print("\n% -- Table: Statistical significance of the depth ordering --------")
    print(r"""\begin{table}[t]
\centering\small
\caption{Bootstrap 95\% CIs (10,000 resamples) on the median causal first-flip
layer, conditional on the fact flipping at all (see
Table~\ref{tab:appearance} for the flip-rate/appearance-rate view), and
Holm-corrected significance tests between adjacent conditions in the
hypothesized depth spectrum. MW = Mann-Whitney $U$ (rank-biserial effect
size $r$ with bootstrap 95\% CI; $r>0$ means the first-listed condition's
first-flip layers tend to be later than the second's -- not specifically a
difference in medians); Fisher = Fisher's exact test on the flip-rate
2$\times$2 table. $p$ values shown are Holm-adjusted within each test family
(raw $p$ in \texttt{stats\_summary.csv}).}
\label{tab:stats}
\begin{tabular}{lccc}
\toprule
\textbf{Condition} & \textbf{Median [95\% CI]} & \textbf{MW $p_{\text{holm}}$ ($r$)} & \textbf{Fisher $p_{\text{holm}}$} \\
\midrule""")
    for cond_a, cond_b in ADJACENT_PAIRS:
        row_a = ci[ci["condition"] == cond_a]
        if row_a.empty:
            continue
        row_a = row_a.iloc[0]
        key = f"{cond_a}_vs_{cond_b}"
        mw_str = f"{mw.loc[key, 'p_holm']:.3g} ({mw.loc[key, 'effect_size_rank_biserial']:.2f})" \
            if key in mw.index else "--"
        fe_str = f"{fe.loc[key, 'p_holm']:.3g}" if key in fe.index else "--"
        print(f"{CONDITION_LABELS[cond_a]} & {row_a['value']:.1f} "
              f"[{row_a['ci_lo']:.1f}, {row_a['ci_hi']:.1f}] & {mw_str} & {fe_str} \\\\")
    last_cond = ADJACENT_PAIRS[-1][1]
    row_last = ci[ci["condition"] == last_cond]
    if not row_last.empty:
        row_last = row_last.iloc[0]
        print(f"{CONDITION_LABELS[last_cond]} & {row_last['value']:.1f} "
              f"[{row_last['ci_lo']:.1f}, {row_last['ci_hi']:.1f}] & -- & -- \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def print_appearance_latex_table(stats_df: pd.DataFrame) -> None:
    """LaTeX snippet: base-vs-final appearance-rate (McNemar) and, among facts
    that reach top-1 in both variants, first-appearance-depth shift
    (Wilcoxon both-reached), per condition. Complements
    Table~\\ref{tab:stats}'s depth-among-those-that-flip view with the
    appearance-rate view (items 5/6)."""
    mc = stats_df[stats_df["type"] == "mcnemar"].set_index("condition")
    wb = stats_df[stats_df["type"] == "wilcoxon_both_reached"].set_index("condition")
    if mc.empty and wb.empty:
        return

    print("\n% -- Table: Base vs. final appearance rate and depth (paired) ------")
    print(r"""\begin{table}[t]
\centering\small
\caption{Paired base-vs-final comparisons, logit lens, train prompts.
McNemar tests whether a fact's probability of ever reaching top-1 changes
after LoRA ($n_{01}$ = newly reaches top-1, $n_{10}$ = stops reaching
top-1 -- the discordant pairs). Wilcoxon (both-reached) tests the shift in
first-appearance layer among facts that reach top-1 in \emph{both} variants
(median diff $<0$ means earlier after LoRA). $p$ values shown are
Holm-adjusted within each test family (raw $p$ in
\texttt{stats\_summary.csv}).}
\label{tab:appearance}
\begin{tabular}{lcccc}
\toprule
\textbf{Condition} & \textbf{McNemar $n_{01}$/$n_{10}$} & \textbf{McNemar $p_{\text{holm}}$} & \textbf{Median diff [95\% CI]} & \textbf{Wilcoxon $p_{\text{holm}}$} \\
\midrule""")
    for cond in COND_ORDER:
        mc_pair, mc_p = "--", "--"
        if cond in mc.index:
            row = mc.loc[cond]
            mc_pair = f"{int(row['n01'])}/{int(row['n10'])}"
            mc_p = f"{row['p_holm']:.3g}" if not np.isnan(row["p_holm"]) else "n/a"
        w_diff, w_p = "--", "--"
        if cond in wb.index:
            row = wb.loc[cond]
            w_diff = f"{row['median_diff']:+.1f} [{row['ci_lo']:+.1f}, {row['ci_hi']:+.1f}]"
            w_p = f"{row['p_holm']:.3g}"
        print(f"{CONDITION_LABELS[cond]} & {mc_pair} & {mc_p} & {w_diff} & {w_p} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
