"""Per-fact trajectory taxonomy over layerwise.parquet (post-processing, no GPU).

first_layer cannot tell an answer that surfaces and holds from one that surfaces and
is then suppressed, so this adds the settle layer, a trajectory class, and the paired
base->LoRA transition between classes.

Writes trajectory.parquet, trajectory_summary.csv, trajectory_transitions.csv,
trajectory_moderator.csv and trajectory_vs_patching.csv under <output_dir>/results/.

Standalone (flat results dir, mirroring visualize.py):
    python -m lora_lens.trajectory results/run31_07_02
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["variant", "step", "lens", "prompt_idx"]
COND_ORDER = ["known", "latent", "unknown", "synthetic"]
CLASS_ORDER = ["never", "transient", "late_only", "persistent"]
LENSES = ("logit", "tuned")


def per_prompt_trajectory(layerwise: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, step, lens, prompt): first_layer (earliest top-1),
    settle_layer (earliest layer top-1 through to the output), traj_class, and how
    many layers hold top-1 / how often it toggles."""
    n_layers = int(layerwise["layer"].max())
    df = layerwise[KEYS + ["layer", "answer_rank"]].copy()
    df["top1"] = df["answer_rank"] == 1
    df = df.sort_values(KEYS + ["layer"], kind="stable")

    grouped = df.groupby(KEYS, sort=False)
    out = grouped.agg(n_layers_top1=("top1", "sum"))

    out = out.join(df[df["top1"]].groupby(KEYS, sort=False)["layer"].min()
                   .rename("first_layer"))
    # The settle layer is one past the last layer that was NOT top-1.
    out = out.join(df[~df["top1"]].groupby(KEYS, sort=False)["layer"].max()
                   .rename("_last_non_top1"))
    out = out.join(df[df["layer"] == n_layers].set_index(KEYS)["top1"]
                   .rename("final_top1"))

    prev = grouped["top1"].shift()
    df["_changed"] = prev.notna() & (df["top1"] != prev)
    out = out.join(df.groupby(KEYS, sort=False)["_changed"].sum().rename("n_toggles"))

    out["settle_layer"] = np.where(out["final_top1"],
                                   out["_last_non_top1"].fillna(-1) + 1, np.nan)
    out["traj_class"] = np.select(
        [out["n_layers_top1"] == 0,
         ~out["final_top1"].astype(bool),
         out["settle_layer"] >= n_layers],
        ["never", "transient", "late_only"], default="persistent")
    out = out.drop(columns="_last_non_top1").reset_index()

    meta = layerwise[KEYS + ["fact_id", "condition", "prompt_type"]].drop_duplicates(KEYS)
    return out.merge(meta, on=KEYS, how="left")


def summarize_trajectory(traj: pd.DataFrame) -> pd.DataFrame:
    """Class shares and mean depths per (variant, step, lens, condition, prompt_type)."""
    keys = ["variant", "step", "lens", "condition", "prompt_type"]
    counts = (traj.groupby(keys + ["traj_class"], sort=False).size()
              .unstack("traj_class", fill_value=0))
    for cls in CLASS_ORDER:
        if cls not in counts.columns:
            counts[cls] = 0
    counts = counts[CLASS_ORDER]
    counts["n_prompts"] = counts.sum(axis=1)

    depths = traj.groupby(keys, sort=False).agg(
        mean_first_layer=("first_layer", "mean"),
        n_first_layer=("first_layer", "count"),
        mean_settle_layer=("settle_layer", "mean"),
        n_settle_layer=("settle_layer", "count"),
        mean_toggles=("n_toggles", "mean"))
    return counts.join(depths).reset_index()


def _paired_classes(traj: pd.DataFrame, condition: str, lens: str,
                    prompt_type: str) -> pd.DataFrame:
    """Wide base-vs-final traj_class per fact, restricted to facts present in both."""
    sub = traj[(traj["lens"] == lens) & (traj["condition"] == condition) &
               (traj["prompt_type"] == prompt_type) &
               (traj["variant"].isin(["base", "final"]))]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index="prompt_idx", columns="variant",
                           values="traj_class", aggfunc="first")
    if not {"base", "final"}.issubset(wide.columns):
        return pd.DataFrame()
    return wide.dropna()


def transitions(traj: pd.DataFrame, lens: str = "logit",
                prompt_type: str = "train") -> pd.DataFrame:
    """Base->final traj_class transition counts per condition, plus McNemar on whether
    the answer survives to the output (paired, so n01 = LoRA fixed it, n10 = broke it)."""
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    survives = {"persistent", "late_only"}
    rows = []
    for cond in [c for c in COND_ORDER if c in set(traj["condition"])]:
        wide = _paired_classes(traj, cond, lens, prompt_type)
        if wide.empty:
            continue
        for base_cls in CLASS_ORDER:
            for final_cls in CLASS_ORDER:
                n = int(((wide["base"] == base_cls) & (wide["final"] == final_cls)).sum())
                if n:
                    rows.append({"lens": lens, "condition": cond, "base_class": base_cls,
                                 "final_class": final_cls, "n": n})

        b, f = wide["base"].isin(survives), wide["final"].isin(survives)
        n11, n10 = int((b & f).sum()), int((b & ~f).sum())
        n01, n00 = int((~b & f).sum()), int((~b & ~f).sum())
        stat, pval = float("nan"), float("nan")
        if n10 or n01:
            res = sm_mcnemar(np.array([[n11, n10], [n01, n00]]), exact=True)
            stat, pval = float(res.statistic), float(res.pvalue)
        rows.append({"lens": lens, "condition": cond, "base_class": "ANY",
                     "final_class": "SURVIVES",
                     "n": len(wide), "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                     "mcnemar_stat": stat, "p_value": pval})
    return pd.DataFrame(rows)


def _baseline_logprob(layerwise: pd.DataFrame, lens: str, prompt_type: str) -> pd.Series:
    """Base-variant final-layer answer log-prob per prompt, the continuous moderator."""
    n_layers = int(layerwise["layer"].max())
    base = layerwise[(layerwise["variant"] == "base") & (layerwise["lens"] == lens) &
                     (layerwise["prompt_type"] == prompt_type) &
                     (layerwise["layer"] == n_layers)]
    return base.set_index("prompt_idx")["answer_logprob"].rename("baseline_logprob")


def moderator_data(traj: pd.DataFrame, layerwise: pd.DataFrame, lens: str = "logit",
                   prompt_type: str = "train") -> pd.DataFrame:
    """Per-prompt join of baseline log-prob with base/final settle layers (and shift).

    Shared by moderator_regression (OLS) and fig_moderator_scatter (points).
    """
    sub = traj[(traj["lens"] == lens) & (traj["prompt_type"] == prompt_type) &
               (traj["variant"].isin(["base", "final"]))]
    wide = sub.pivot_table(index="prompt_idx", columns="variant", values="settle_layer")
    if wide.empty or not {"base", "final"}.issubset(wide.columns):
        return pd.DataFrame()
    meta = (sub[sub["variant"] == "final"][["prompt_idx", "fact_id", "condition"]]
            .drop_duplicates("prompt_idx")
            .set_index("prompt_idx"))
    wide = wide.join(meta, how="left")
    wide = wide.join(_baseline_logprob(layerwise, lens, prompt_type), how="inner")
    wide = wide.rename(columns={"base": "base_settle", "final": "final_settle"})
    wide["shift"] = wide["final_settle"] - wide["base_settle"]
    wide["lens"] = lens
    wide["prompt_type"] = prompt_type
    return wide.reset_index()


def moderator_regression(traj: pd.DataFrame, layerwise: pd.DataFrame, lens: str = "logit",
                         prompt_type: str = "train") -> pd.DataFrame:
    """Regress post-LoRA settle depth on continuous baseline log-prob, replacing the
    four bins as the inferential unit; depth_final is primary because `shift` needs the
    fact to settle in both variants, which in base is almost only `known`."""
    import statsmodels.api as sm

    wide = moderator_data(traj, layerwise, lens=lens, prompt_type=prompt_type)
    if wide.empty:
        return pd.DataFrame()

    rows = []
    for model, outcome in (("depth_final", "final_settle"), ("shift", "shift")):
        data = wide[[outcome, "baseline_logprob"]].dropna()
        if len(data) < 10:
            continue
        fit = sm.OLS(data[outcome],
                     sm.add_constant(data[["baseline_logprob"]])).fit(cov_type="HC1")
        ci = fit.conf_int()
        for name in fit.params.index:
            rows.append({"lens": lens, "model": model, "term": name,
                         "coef": float(fit.params[name]),
                         "se": float(fit.bse[name]), "p_value": float(fit.pvalues[name]),
                         "ci_lo": float(ci.loc[name, 0]), "ci_hi": float(ci.loc[name, 1]),
                         "n": int(fit.nobs), "r_squared": float(fit.rsquared)})
    return pd.DataFrame(rows)


def vs_patching(traj: pd.DataFrame, patching: pd.DataFrame, lens: str = "logit",
                prompt_type: str = "train") -> pd.DataFrame:
    """Join the LoRA settle layer to the causal first-flip layer per fact, to test
    whether settle_layer rather than first_layer is what patching tracks."""
    final_traj = traj[(traj["variant"] == "final") & (traj["lens"] == lens) &
                      (traj["prompt_type"] == prompt_type)]
    flips = patching[(patching["layer"] == -1) & (patching["variant"] == "final")]
    if final_traj.empty or flips.empty:
        return pd.DataFrame()
    out = final_traj[["fact_id", "condition", "first_layer", "settle_layer",
                      "traj_class"]].merge(
        flips[["fact_id", "first_flip_layer", "flipped"]], on="fact_id", how="inner")
    out["lens"] = lens
    return out


def depth_agreement(joined: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation and median level gap between lens depth and causal first-flip,
    since the two can rank facts alike while disagreeing badly on absolute depth."""
    from scipy import stats as sstats

    rows = []
    for cond, g in joined.groupby("condition"):
        for col in ("first_layer", "settle_layer"):
            pair = g[[col, "first_flip_layer"]].dropna()
            if len(pair) < 5:
                continue
            rho, p = sstats.spearmanr(pair[col], pair["first_flip_layer"])
            rows.append({"condition": cond, "metric": col, "spearman_rho": float(rho),
                         "p_value": float(p), "n": len(pair),
                         "median_lens": float(pair[col].median()),
                         "median_first_flip": float(pair["first_flip_layer"].median()),
                         "median_gap": float((pair[col] - pair["first_flip_layer"]).median())})
    return pd.DataFrame(rows)


def run_trajectory(cfg) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    _run_trajectory(results_dir)


def _run_trajectory(results_dir: Path) -> None:
    """Core trajectory logic against a flat results directory."""
    layerwise_path = results_dir / "layerwise.parquet"
    if not layerwise_path.exists():
        raise SystemExit(f"[trajectory] {layerwise_path} not found -- run analyze first.")

    layerwise = pd.read_parquet(layerwise_path)
    traj = per_prompt_trajectory(layerwise)
    traj.to_parquet(results_dir / "trajectory.parquet", index=False)

    summary = summarize_trajectory(traj)
    summary.to_csv(results_dir / "trajectory_summary.csv", index=False)
    print("\n[trajectory] Class counts (logit lens, train prompts, base vs final):")
    show = summary[(summary["lens"] == "logit") & (summary["prompt_type"] == "train") &
                   (summary["variant"].isin(["base", "final"]))]
    print(show[["variant", "condition"] + CLASS_ORDER +
               ["mean_settle_layer", "n_settle_layer"]].to_string(index=False))

    # Both lenses: tag rows with lens column and concatenate.
    trans_parts = [transitions(traj, lens=lens) for lens in LENSES]
    trans = pd.concat([t for t in trans_parts if not t.empty], ignore_index=True)
    if not trans.empty:
        trans.to_csv(results_dir / "trajectory_transitions.csv", index=False)
        surv = trans[(trans["base_class"] == "ANY") & (trans["lens"] == "logit")]
        if not surv.empty:
            print("\n[trajectory] Does the answer survive to the output? "
                  "(McNemar, base vs final, logit)")
            print(surv[["condition", "n11", "n10", "n01", "n00", "p_value"]]
                  .to_string(index=False))

    mod_parts = [moderator_regression(traj, layerwise, lens=lens) for lens in LENSES]
    mod = pd.concat([m for m in mod_parts if not m.empty], ignore_index=True)
    if not mod.empty:
        mod.to_csv(results_dir / "trajectory_moderator.csv", index=False)
        print("\n[trajectory] Settle depth ~ baseline log-prob (OLS, HC1); depth_final is "
              "primary, shift is range-restricted:")
        print(mod[mod["lens"] == "logit"].to_string(index=False))

    patch_path = results_dir / "patching.csv"
    if patch_path.exists():
        patch_df = pd.read_csv(patch_path)
        join_parts = [vs_patching(traj, patch_df, lens=lens) for lens in LENSES]
        joined = pd.concat([j for j in join_parts if not j.empty], ignore_index=True)
        if not joined.empty:
            joined.to_csv(results_dir / "trajectory_vs_patching.csv", index=False)
            print("\n[trajectory] Lens depth vs causal first-flip layer (logit):")
            print(depth_agreement(joined[joined["lens"] == "logit"]).to_string(index=False))

    print(f"\n[trajectory] Full results in {results_dir}")


if __name__ == "__main__":
    from .utils import configure_stdout

    configure_stdout()
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/results")
    _run_trajectory(results)
