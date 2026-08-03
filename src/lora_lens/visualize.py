"""Generate publication-ready figures and LaTeX table snippets from pipeline results.

Three figures, each designed to answer one specific question:
  fig_delta_logprob.pdf  — WHERE does LoRA's gain appear across layers?
  fig_patching.pdf       — WHERE is the update causally located, and how does it grow?
  fig_rank_ablation.pdf  — Does more capacity (rank) change WHERE or just HOW MANY?
  fig_layer_top1_*.pdf   — At which layers is the correct answer rank-1? (spike plot)
  fig_layer_top1_*_comparison.pdf — same, but before vs after LoRA on one figure

Usage as a pipeline stage:
    python -m lora_lens.run --config configs/default.yaml --stages visualize

Standalone (pass results directory):
    python -m lora_lens.visualize /content/drive/MyDrive/nlp_outputs/results
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color-blind-friendly palette (Okabe-Ito) — ordered by increasing knowledge absence.
COLORS = {
    "known":     "#0072B2",  # blue        — model already knows it
    "latent":    "#56B4E9",  # sky-blue    — in top-5, but suppressed
    "unknown":   "#E69F00",  # amber       — not in top-5, real fact
    "synthetic": "#D55E00",  # vermillion  — invented entity, zero prior
}
CONDITION_LABELS = {
    "known":     "Known",
    "latent":    "Latent (top-5)",
    "unknown":   "Unknown",
    "synthetic": "Synthetic",
}
LINE_STYLES = {"known": "solid", "latent": (0, (5, 1)), "unknown": "dashed", "synthetic": "dotted"}
MARKERS     = {"known": "o",     "latent": "D",          "unknown": "s",      "synthetic": "^"}
COND_ORDER  = ["known", "latent", "unknown", "synthetic"]
MARKER_EVERY = 4


# ── Figure 1: Layer-wise Δ log-probability ─────────────────────────────────────
#
# Answers: WHERE across layers does LoRA add log-probability mass?
# Key story: Known = small, early; Unknown = large, mid-layer; Synthetic = large, final layers.

def fig_delta_logprob(results_dir: Path, figures_dir: Path) -> None:
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping delta logprob figure.")
        return

    df = pd.read_parquet(path)
    df = df[(df["lens"] == "logit") & (df["variant"].isin(["base", "final"]))]

    # Training prompts for all three conditions (paraphrases exist only for known/unknown,
    # and using train is consistent and shows the maximum LoRA effect).
    df = df[df["prompt_type"] == "train"]

    mean_lp = (df.groupby(["variant", "condition", "layer"])["answer_logprob"]
                 .mean().reset_index())

    base_lp  = (mean_lp[mean_lp["variant"] == "base"]
                .rename(columns={"answer_logprob": "base_lp"}).drop(columns="variant"))
    final_lp = (mean_lp[mean_lp["variant"] == "final"]
                .rename(columns={"answer_logprob": "final_lp"}).drop(columns="variant"))
    delta = base_lp.merge(final_lp, on=["condition", "layer"])
    delta["delta_lp"] = delta["final_lp"] - delta["base_lp"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for cond in COND_ORDER:
        sub = delta[delta["condition"] == cond].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["delta_lp"],
                color=COLORS[cond],
                linestyle=LINE_STYLES[cond],
                marker=MARKERS[cond], markevery=MARKER_EVERY, markersize=4,
                label=CONDITION_LABELS[cond])

    # Vertical markers at median causal first-flip layers, read live from patching.csv
    # (final checkpoint) so the figure always matches whatever the data currently says.
    patch_path = results_dir / "patching.csv"
    if patch_path.exists():
        patch_df = pd.read_csv(patch_path)
        final_patch = patch_df[(patch_df["layer"] == -1) & (patch_df["variant"] == "final")]
        for cond in COND_ORDER:
            medians = final_patch[final_patch["condition"] == cond]["first_flip_layer"].dropna()
            if medians.empty:
                continue
            ax.axvline(medians.median(), color=COLORS[cond], linewidth=0.8,
                      linestyle=":", alpha=0.7)
    else:
        print(f"[viz] {patch_path} not found — skipping first-flip-layer markers.")

    ax.axhline(0, color="black", linewidth=0.6, linestyle="-", alpha=0.4)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Δ mean log-probability\n(LoRA − base)")
    ax.set_title("Where does LoRA add log-probability?")
    ax.set_xlim(0, 24)
    ax.legend(loc="upper left")

    fig.tight_layout(pad=0.5)
    out = figures_dir / "fig_delta_logprob.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── Figure 2: Causal patching — two panels ─────────────────────────────────────
#
# Left: WHERE is the update causally located? (distribution of first-flip layers)
# Right: HOW does it accumulate during training? (count vs step)

def fig_patching(results_dir: Path, figures_dir: Path) -> None:
    path = results_dir / "patching.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping patching figure.")
        return

    df = pd.read_csv(path)
    summary = df[df["layer"] == -1].copy()

    fig, (ax_box, ax_dyn) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # ── Left panel: distribution of first-flip-layer at final checkpoint ──────
    final = summary[summary["variant"] == "final"].copy()
    present = [c for c in COND_ORDER if c in final["condition"].values]

    # Fix the y-range up front so the "n/N" annotations all land at the same
    # height; otherwise each boxplot's autoscale (as points are added one
    # condition at a time) leaves earlier low-range conditions (e.g. "known",
    # which rarely needs a flip) with labels far below the rest.
    ax_box.set_ylim(-1, 26)

    for i, cond in enumerate(present):
        sub = final[final["condition"] == cond]["first_flip_layer"].dropna()
        n_total = len(final[final["condition"] == cond])
        n_flip = len(sub)

        bp = ax_box.boxplot(sub, positions=[i], widths=0.45, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2),
                            flierprops=dict(marker="o", markersize=2.5,
                                            markerfacecolor=COLORS[cond], alpha=0.5))
        bp["boxes"][0].set(facecolor=COLORS[cond], alpha=0.75)

        # Jitter individual points.
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(sub))
        ax_box.scatter(i + jitter, sub, color=COLORS[cond],
                       alpha=0.35, s=8, zorder=3)

        # Annotate flip count above the box.
        ax_box.text(i, ax_box.get_ylim()[1] if ax_box.get_ylim()[1] > 0 else 25,
                    f"{n_flip}/{n_total}", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS[cond], fontweight="bold")

    ax_box.set_xticks(list(range(len(present))))
    ax_box.set_xticklabels([CONDITION_LABELS[c] for c in present])
    ax_box.set_ylabel("First-flip layer")
    ax_box.set_title("Causal locus of LoRA update")

    # ── Right panel: count of flipped facts vs training step ──────────────────
    # Known is trivially always 100 (already correct); omit it.
    for cond in [c for c in COND_ORDER if c != "known"]:
        sub = (summary[(summary["condition"] == cond) & (summary["variant"] != "final")]
               .groupby("step")["first_flip_layer"]
               .count().reset_index(name="flipped"))
        # also add the final checkpoint step
        final_count = summary[(summary["condition"] == cond) &
                               (summary["variant"] == "final")]["first_flip_layer"].count()
        final_step_val = summary[(summary["variant"] == "final")]["step"].max()
        sub = pd.concat([sub, pd.DataFrame([{"step": final_step_val,
                                              "flipped": final_count}])],
                        ignore_index=True).sort_values("step")
        ax_dyn.plot(sub["step"], sub["flipped"],
                    color=COLORS[cond], linestyle=LINE_STYLES[cond],
                    marker=MARKERS[cond], markersize=5,
                    label=CONDITION_LABELS[cond])

    ax_dyn.set_xlabel("Training step")
    ax_dyn.set_ylabel("Facts where patching flips\nprediction (out of 100)")
    ax_dyn.set_title("Learning dynamics")
    ax_dyn.set_ylim(0, 100)
    ax_dyn.legend()

    fig.tight_layout(pad=0.4, w_pad=1.5)
    out = figures_dir / "fig_patching.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── Figure 3: Rank ablation — accuracy and depth ──────────────────────────────
#
# Left: Does more rank → better accuracy? (yes, especially for novel knowledge)
# Right: Does more rank → earlier emergence? (NO for synthetic — key insight)

def fig_rank_ablation(results_dir: Path, figures_dir: Path) -> None:
    path = results_dir / "rank_ablation_summary.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping rank ablation figure.")
        return

    df = pd.read_csv(path)
    df = df[df["lens"] == "logit"]

    fig, (ax_acc, ax_layer) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ranks_ordered = sorted(df["rank"].unique())

    # ── Left: accuracy vs rank ────────────────────────────────────────────────
    # Solid lines = train prompts; dashed = paraphrase.
    for cond in COND_ORDER:
        for ptype, lstyle, suffix in [("train", "solid", ""), ("paraphrase", "dashed", " (para.)")]:
            sub = (df[(df["condition"] == cond) & (df["prompt_type"] == ptype)]
                   .sort_values("rank"))
            if sub.empty:
                continue
            label = CONDITION_LABELS[cond] + suffix if ptype == "paraphrase" else CONDITION_LABELS[cond]
            ax_acc.plot(sub["rank"], sub["final_accuracy"],
                        color=COLORS[cond], linestyle=lstyle,
                        marker=MARKERS[cond], markersize=4,
                        label=label)

    ax_acc.set_xlabel("LoRA rank $r$  (0 = base model)")
    ax_acc.set_ylabel("Acc@1")
    ax_acc.set_title("Accuracy vs rank")
    ax_acc.set_xticks(ranks_ordered)
    ax_acc.set_ylim(-0.03, 1.03)
    ax_acc.legend(fontsize=7, ncol=2)

    # ── Right: first-layer-of-appearance vs rank ──────────────────────────────
    for cond in COND_ORDER:
        sub = (df[(df["condition"] == cond) & (df["prompt_type"] == "train")]
               .sort_values("rank"))
        if sub.empty:
            continue
        ax_layer.plot(sub["rank"], sub["mean_first_layer"],
                      color=COLORS[cond], linestyle=LINE_STYLES[cond],
                      marker=MARKERS[cond], markersize=4,
                      label=CONDITION_LABELS[cond])

    ax_layer.set_xlabel("LoRA rank $r$  (0 = base model)")
    ax_layer.set_ylabel("Mean first-appearance layer")
    ax_layer.set_title("Emergence depth vs rank")
    ax_layer.set_xticks(ranks_ordered)
    ax_layer.set_ylim(16, 24)
    ax_layer.legend()

    fig.tight_layout(pad=0.4, w_pad=1.5)
    out = figures_dir / "fig_rank_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── Figure 3b: Locality drift across training ──────────────────────────────────
#
# Mirrors the right panel of fig_patching: KL divergence (top-k, base||LoRA) on
# neighborhood prompts vs. training step, one line per condition. Substantiates
# the paper's own recommendation to "monitor neighborhood KL-divergence during
# training as an early warning signal" with an actual curve.

def fig_locality(results_dir: Path, figures_dir: Path) -> None:
    path = results_dir / "locality.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping locality-vs-step figure.")
        return

    df = pd.read_csv(path)
    if "step" not in df.columns or df["variant"].nunique() <= 1:
        print("[viz] locality.csv has only one checkpoint — skipping locality-vs-step figure "
              "(re-run score_locality after this update to get per-checkpoint tracking).")
        return

    by_step = (df.groupby(["condition", "variant", "step"])["kl_div"]
              .mean().reset_index())

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for cond in COND_ORDER:
        sub = by_step[by_step["condition"] == cond].sort_values("step")
        if sub.empty:
            continue
        ax.plot(sub["step"], sub["kl_div"],
                color=COLORS[cond], linestyle=LINE_STYLES[cond],
                marker=MARKERS[cond], markersize=5,
                label=CONDITION_LABELS[cond])

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean top-$k$ KL(base $\\|$ LoRA)\non neighborhood prompts")
    ax.set_title("Locality drift during training")
    ax.legend(fontsize=7)

    fig.tight_layout(pad=0.5)
    out = figures_dir / "fig_locality.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── Figure 4: Layer-wise top-1 accuracy (histogram-style) ─────────────────────
#
# Answers: at which layers does the correct answer briefly become rank-1?
# Combined panel pools all conditions; by-condition panel shows the four trends.
# Uses layerwise.parquet (in_top_1 per layer) — no extra pipeline stage needed.

def _layer_top1_curve(df: pd.DataFrame, variant: str, lens: str,
                      prompt_type: str = "train") -> pd.DataFrame:
    """Per-layer fraction of prompts where the answer is rank-1 (0–100%)."""
    sub = df[(df["variant"] == variant) & (df["lens"] == lens)
             & (df["prompt_type"] == prompt_type)]
    if sub.empty:
        return pd.DataFrame(columns=["condition", "layer", "pct_top1", "n_prompts"])
    pct = (sub.groupby(["condition", "layer"], as_index=False)
             .agg(pct_top1=("in_top_1", "mean"), n_prompts=("in_top_1", "size")))
    pct["pct_top1"] *= 100.0
    return pct


def _layer_top1_pooled(df: pd.DataFrame, variant: str, lens: str,
                       prompt_type: str = "train") -> pd.DataFrame:
    """Per-layer % top-1 across all conditions for one variant."""
    sub = df[(df["variant"] == variant) & (df["lens"] == lens)
             & (df["prompt_type"] == prompt_type)]
    if sub.empty:
        return pd.DataFrame(columns=["layer", "pct_top1"])
    pooled = (sub.groupby("layer", as_index=False)
              .agg(pct_top1=("in_top_1", "mean")))
    pooled["pct_top1"] *= 100.0
    return pooled


def fig_layer_top1_combined(results_dir: Path, figures_dir: Path,
                            variant: str = "base", lens: str = "logit",
                            prompt_type: str = "train") -> None:
    """Single-variant combined curve (base or final only)."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping layer top-1 combined figure.")
        return

    df = pd.read_parquet(path)
    pooled = _layer_top1_pooled(df, variant, lens, prompt_type)
    if pooled.empty:
        print("[viz] no rows for layer top-1 combined figure — skipping.")
        return

    n_layers = int(pooled["layer"].max())
    layers = np.arange(n_layers + 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(pooled["layer"], pooled["pct_top1"],
           width=0.85, color="#666666", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("% of facts with correct answer\n(rank-1 at this layer)")
    ax.set_title(f"Layer-wise accuracy — all conditions ({variant}, {lens} lens)")
    ax.set_xlim(-0.5, n_layers + 0.5)
    ax.set_xticks(layers[::max(1, len(layers) // 8)])
    ax.set_ylim(0, min(105, pooled["pct_top1"].max() * 1.15 + 1))

    fig.tight_layout(pad=0.5)
    suffix = "" if (variant == "base" and lens == "logit") else f"_{variant}_{lens}"
    out = figures_dir / f"fig_layer_top1_combined{suffix}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


def fig_layer_top1_combined_comparison(results_dir: Path, figures_dir: Path,
                                       lens: str = "logit",
                                       prompt_type: str = "train") -> None:
    """Grouped bars: before vs after LoRA, all conditions pooled."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping layer top-1 comparison figure.")
        return

    df = pd.read_parquet(path)
    base = _layer_top1_pooled(df, "base", lens, prompt_type)
    final = _layer_top1_pooled(df, "final", lens, prompt_type)
    if base.empty or final.empty:
        print("[viz] missing base or final rows — skipping layer top-1 comparison figure.")
        return

    merged = base.merge(final, on="layer", suffixes=("_base", "_final"))
    n_layers = int(merged["layer"].max())
    layers = np.arange(n_layers + 1)
    width = 0.38

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(merged["layer"] - width / 2, merged["pct_top1_base"], width,
           color="#BBBBBB", alpha=0.9, edgecolor="white", linewidth=0.4,
           label="Before LoRA (base)")
    ax.bar(merged["layer"] + width / 2, merged["pct_top1_final"], width,
           color="#0072B2", alpha=0.9, edgecolor="white", linewidth=0.4,
           label="After LoRA (final)")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("% of facts with correct answer\n(rank-1 at this layer)")
    ax.set_title(f"Layer-wise accuracy — before vs after LoRA ({lens} lens)")
    ax.set_xlim(-0.5, n_layers + 0.5)
    ax.set_xticks(layers[::max(1, len(layers) // 8)])
    ymax = max(merged["pct_top1_base"].max(), merged["pct_top1_final"].max())
    ax.set_ylim(0, min(105, ymax * 1.15 + 1))
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout(pad=0.5)
    suffix = "" if lens == "logit" else f"_{lens}"
    out = figures_dir / f"fig_layer_top1_combined_comparison{suffix}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


def fig_layer_top1_by_condition(results_dir: Path, figures_dir: Path,
                                variant: str = "base", lens: str = "logit",
                                prompt_type: str = "train") -> None:
    """Single-variant curves by condition (base or final only)."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping layer top-1 by-condition figure.")
        return

    df = pd.read_parquet(path)
    curves = _layer_top1_curve(df, variant, lens, prompt_type)
    if curves.empty:
        print("[viz] no rows for layer top-1 by-condition figure — skipping.")
        return

    n_layers = int(curves["layer"].max())
    ymax = min(105, curves["pct_top1"].max() * 1.15 + 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for cond in COND_ORDER:
        sub = curves[curves["condition"] == cond].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["pct_top1"],
                color=COLORS[cond], linestyle=LINE_STYLES[cond],
                marker=MARKERS[cond], markevery=MARKER_EVERY, markersize=4,
                linewidth=1.8, label=CONDITION_LABELS[cond])

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("% of facts with correct answer\n(rank-1 at this layer)")
    ax.set_title(f"Layer-wise accuracy by condition ({variant}, {lens} lens)")
    ax.set_xlim(0, n_layers)
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout(pad=0.5)
    suffix = "" if (variant == "base" and lens == "logit") else f"_{variant}_{lens}"
    out = figures_dir / f"fig_layer_top1_by_condition{suffix}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


def fig_layer_top1_by_condition_comparison(results_dir: Path, figures_dir: Path,
                                           lens: str = "logit",
                                           prompt_type: str = "train") -> None:
    """Four condition curves × two variants: dashed = before LoRA, solid = after."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping layer top-1 by-condition comparison.")
        return

    df = pd.read_parquet(path)
    base = _layer_top1_curve(df, "base", lens, prompt_type)
    final = _layer_top1_curve(df, "final", lens, prompt_type)
    if base.empty or final.empty:
        print("[viz] missing base or final rows — skipping by-condition comparison.")
        return

    n_layers = int(max(base["layer"].max(), final["layer"].max()))
    ymax = min(105, max(base["pct_top1"].max(), final["pct_top1"].max()) * 1.15 + 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for cond in COND_ORDER:
        b = base[base["condition"] == cond].sort_values("layer")
        f = final[final["condition"] == cond].sort_values("layer")
        if b.empty and f.empty:
            continue
        if not b.empty:
            ax.plot(b["layer"], b["pct_top1"],
                    color=COLORS[cond], linestyle="--", linewidth=1.4,
                    alpha=0.75, label="_nolegend_")
        if not f.empty:
            ax.plot(f["layer"], f["pct_top1"],
                    color=COLORS[cond], linestyle=LINE_STYLES[cond],
                    marker=MARKERS[cond], markevery=MARKER_EVERY, markersize=4,
                    linewidth=1.8, label=CONDITION_LABELS[cond])

    # Condition legend (solid lines) + variant style legend (dashed vs solid).
    cond_handles, cond_labels = ax.get_legend_handles_labels()
    style_handles = [
        plt.Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.4,
                   label="Before LoRA (base)"),
        plt.Line2D([0], [0], color="#333333", linestyle="-", linewidth=1.8,
                   label="After LoRA (final)"),
    ]
    leg1 = ax.legend(cond_handles, cond_labels, loc="upper left", fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="center left", fontsize=7,
              bbox_to_anchor=(0.0, 0.35))

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("% of facts with correct answer\n(rank-1 at this layer)")
    ax.set_title(f"Layer-wise accuracy by condition — before vs after ({lens} lens)")
    ax.set_xlim(0, n_layers)
    ax.set_ylim(0, ymax)

    fig.tight_layout(pad=0.5)
    suffix = "" if lens == "logit" else f"_{lens}"
    out = figures_dir / f"fig_layer_top1_by_condition_comparison{suffix}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── Figure 5: Baseline fact-pool rank/log-prob distribution ────────────────────
#
# Answers: why these specific cut points (rank 1 / rank 5 / logprob threshold)?
# Pure visualization of facts_scored.parquet — no new computation. Rank 5 is a
# safe hardcoded boundary (top5_correct is computed with a fixed k=5 in
# scoring.py, not configurable), but the log-prob cut is
# data.latent_logprob_threshold in the config (default -5.0, see conditions.py)
# and must be read from there so this figure can't silently go stale if that
# threshold is ever tuned (e.g. the sensitivity sweep noted in the paper).

_DEFAULT_LATENT_THRESHOLD = -5.0


def _load_latent_threshold(output_dir: Path) -> float:
    """Read data.latent_logprob_threshold from <output_dir>/config_resolved.yaml
    (written by every pipeline run), falling back to the conditions.py default
    if the file is missing or the key was never overridden."""
    path = output_dir / "config_resolved.yaml"
    if not path.exists():
        return _DEFAULT_LATENT_THRESHOLD
    import yaml
    with open(path, encoding="utf-8") as f:
        resolved = yaml.safe_load(f) or {}
    return resolved.get("data", {}).get("latent_logprob_threshold", _DEFAULT_LATENT_THRESHOLD)


def fig_baseline_distribution(output_dir: Path, figures_dir: Path,
                              latent_threshold: float = _DEFAULT_LATENT_THRESHOLD) -> None:
    path = output_dir / "facts_scored.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping baseline distribution figure.")
        return

    df = pd.read_parquet(path)

    fig, (ax_rank, ax_lp) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # ── Left: bucketed answer_rank histogram ──────────────────────────────────
    bins = [1, 2, 6, 11, 51, np.inf]
    labels = ["1\n(known)", "2\u20135\n(latent cand.)", "6\u201310", "11\u201350", "51+"]
    bucket = pd.cut(df["answer_rank"], bins=bins, right=False, labels=labels)
    counts = bucket.value_counts().reindex(labels)
    bar_colors = [COLORS["known"], COLORS["latent"], COLORS["unknown"],
                  COLORS["unknown"], COLORS["unknown"]]
    ax_rank.bar(range(len(counts)), counts.values, color=bar_colors, alpha=0.85)
    ax_rank.set_xticks(range(len(counts)))
    ax_rank.set_xticklabels(labels, fontsize=7)
    ax_rank.set_xlabel("Base-model answer rank")
    ax_rank.set_ylabel("Number of facts")
    ax_rank.set_title("Rank distribution (n={:,})".format(len(df)))

    # ── Right: answer_logprob histogram with the latent threshold marked ──────
    finite_lp = df["answer_logprob"].replace([np.inf, -np.inf], np.nan).dropna()
    ax_lp.hist(finite_lp.clip(lower=-20), bins=40, color=COLORS["unknown"], alpha=0.75)
    ax_lp.axvline(latent_threshold, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
    ax_lp.text(latent_threshold, ax_lp.get_ylim()[1] * 0.92, f" logprob = {latent_threshold:g}",
              fontsize=7, ha="left", va="top")
    ax_lp.set_xlabel("Base-model answer log-probability")
    ax_lp.set_ylabel("Number of facts")
    ax_lp.set_title("Log-probability distribution")

    fig.suptitle(f"Why rank\u22125 / logprob>{latent_threshold:g} define \u2018latent\u2019",
                fontsize=9, y=1.03)
    fig.tight_layout(pad=0.4, w_pad=1.8)
    out = figures_dir / "fig_baseline_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── LaTeX table snippets ───────────────────────────────────────────────────────

def print_latex_tables(results_dir: Path) -> None:

    # ── Table 1: Main results ──────────────────────────────────────────────────
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        s_all = pd.read_csv(summary_path)
        s = s_all[(s_all["lens"] == "logit") & (s_all["variant"].isin(["base", "final"]))]

        rows = [
            ("Known",          "train", "known",     "train"),
            ("Known",          "para.", "known",     "paraphrase"),
            ("Latent (top-5)", "train", "latent",    "train"),
            ("Latent (top-5)", "para.", "latent",    "paraphrase"),
            ("Unknown",        "train", "unknown",   "train"),
            ("Unknown",        "para.", "unknown",   "paraphrase"),
            ("Synthetic",      "train", "synthetic", "train"),
            # Present only once synthetic facts carry generated paraphrases
            # (make_synthetic() in data.py); the row is silently skipped below
            # via the b.empty/f.empty guard on older runs without this data.
            ("Synthetic",      "para.", "synthetic", "paraphrase"),
        ]

        # Plain ASCII: Windows consoles default to cp1252, which cannot encode
        # box-drawing characters and would crash the print.
        print("\n% -- Table 1: Main results --------------------------------------")
        print(r"""\begin{table}[t]
\centering\small
\caption{Logit-lens results before and after LoRA fine-tuning (Pythia-410m-deduped, $r=16$).
\emph{First layer $\Delta$} = shift in mean first-appearance layer (negative = earlier).
Para.\ = held-out paraphrase prompts.}
\label{tab:main_results}
\begin{tabular}{llccccc}
\toprule
 & & \multicolumn{2}{c}{\textbf{Acc@1}} & \multicolumn{2}{c}{\textbf{Mean log-prob}} & \textbf{First layer} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\textbf{Condition} & \textbf{Prompts} & Base & LoRA & Base & LoRA & $\Delta$ \\
\midrule""")
        for label_cond, label_ptype, cond, ptype in rows:
            b = s[(s["variant"] == "base") & (s["condition"] == cond) & (s["prompt_type"] == ptype)]
            f = s[(s["variant"] == "final") & (s["condition"] == cond) & (s["prompt_type"] == ptype)]
            if b.empty or f.empty:
                continue
            b, f = b.iloc[0], f.iloc[0]
            shift = f["mean_first_layer"] - b["mean_first_layer"]
            sign = "+" if shift >= 0 else ""
            print(f"{label_cond} & {label_ptype} & "
                  f"{b['final_accuracy']:.3f} & {f['final_accuracy']:.3f} & "
                  f"{b['mean_final_logprob']:.2f} & {f['mean_final_logprob']:.2f} & "
                  f"{sign}{shift:.1f} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")

        # ── Table 1b: Logit-lens vs. tuned-lens comparison ────────────────────
        # Substantiates the "convergent evidence" claim with a number instead of
        # an assertion: both lenses are computed for every checkpoint, but the
        # tuned-lens rows were never surfaced anywhere until now.
        cmp_df = s_all[(s_all["variant"] == "final") & (s_all["prompt_type"] == "train")]
        if "tuned" in cmp_df["lens"].unique():
            print("\n% -- Table 1b: Logit-lens vs. tuned-lens comparison -------------")
            print(r"""\begin{table}[t]
\centering\small
\caption{Final-checkpoint logit-lens vs.\ tuned-lens agreement on training prompts.
Convergence between an untuned and a trained unembedding-correction lens indicates
the layer-wise trajectories are not an artifact of the base model's raw unembedding.}
\label{tab:lens_comparison}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{\textbf{Acc@1}} & \multicolumn{2}{c}{\textbf{Mean first layer}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Condition} & Logit & Tuned & Logit & Tuned \\
\midrule""")
            for cond in COND_ORDER:
                lg = cmp_df[(cmp_df["condition"] == cond) & (cmp_df["lens"] == "logit")]
                tn = cmp_df[(cmp_df["condition"] == cond) & (cmp_df["lens"] == "tuned")]
                if lg.empty or tn.empty:
                    continue
                lg, tn = lg.iloc[0], tn.iloc[0]
                print(f"{CONDITION_LABELS[cond]} & "
                      f"{lg['final_accuracy']:.3f} & {tn['final_accuracy']:.3f} & "
                      f"{lg['mean_first_layer']:.1f} & {tn['mean_first_layer']:.1f} \\\\")
            print(r"""\bottomrule
\end{tabular}
\end{table}""")
        else:
            print("\n[viz] no tuned-lens rows in summary.csv — skipping lens comparison table.")

    # ── Table 2: Causal patching ───────────────────────────────────────────────
    patch_path = results_dir / "patching.csv"
    if patch_path.exists():
        p = pd.read_csv(patch_path)
        p = p[(p["layer"] == -1) & (p["variant"] == "final")]
        pat = (p.groupby("condition")["first_flip_layer"]
                .agg(mean="mean", median="median", count="count")
                .reindex(COND_ORDER))

        print("\n% -- Table 2: Causal patching -----------------------------------")
        print(r"""\begin{table}[t]
\centering\small
\caption{Activation patching at the final LoRA checkpoint ($r=16$).
\emph{Flipped} = facts (out of 100) where any single-layer patch causes the base
model to predict the target answer.}
\label{tab:patching}
\begin{tabular}{lccc}
\toprule
\textbf{Condition} & \textbf{Flipped} & \textbf{Mean first-flip layer} & \textbf{Median} \\
\midrule""")
        for cond, label in [(c, CONDITION_LABELS[c]) for c in COND_ORDER]:
            if cond not in pat.index:
                continue
            r = pat.loc[cond]
            if pd.isna(r["count"]):
                continue
            print(f"{label} & {int(r['count'])}/100 & {r['mean']:.2f} & {r['median']:.0f} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # ── Table 3: Locality ──────────────────────────────────────────────────────
    loc_path = results_dir / "locality.csv"
    if loc_path.exists():
        loc = pd.read_csv(loc_path)
        # locality.csv now has one row per checkpoint (variant/step); restrict to
        # the final checkpoint for this summary table.
        if "variant" in loc.columns:
            loc = loc[loc["variant"] == "final"]
        agg_kwargs = dict(rate=("preserved", "mean"), n=("preserved", "count"))
        has_kl = "kl_div" in loc.columns
        has_secondary = "kl_div_full" in loc.columns and "escaped_mass" in loc.columns
        if has_kl:
            agg_kwargs["kl"] = ("kl_div", "mean")
        if has_secondary:
            agg_kwargs["kl_full"] = ("kl_div_full", "mean")
            agg_kwargs["escaped"] = ("escaped_mass", "mean")
        loc_sum = loc.groupby("condition").agg(**agg_kwargs).reset_index()

        print("\n% -- Table 3: Locality -------------------------------------------")
        if has_secondary:
            print(r"""\begin{table}[t]
\centering\small
\caption{Neighborhood locality at the final checkpoint.
\emph{Pres.} = fraction of prompts where LoRA top-1 agrees with base top-1
(1.0 = no collateral damage). \emph{KL top-$k$} (primary) and \emph{KL full}
(secondary, no renormalization) are mean KL$(P_\text{base}\|P_\text{LoRA})$;
\emph{Esc.\ mass} is the fraction of LoRA's probability mass falling outside
the base model's top-$k$ support (high = confident jump to a new token, not
broad top-$k$ scrambling).}
\label{tab:locality}
\begin{tabular}{lccccc}
\toprule
\textbf{Condition} & \textbf{Pres.} & \textbf{KL top-$k$} & \textbf{KL full} &
\textbf{Esc.\ mass} & \textbf{$N$} \\
\midrule""")
            for cond in [c for c in COND_ORDER if c != "synthetic"]:
                label = CONDITION_LABELS[cond]
                row = loc_sum[loc_sum["condition"] == cond]
                if row.empty:
                    continue
                row = row.iloc[0]
                print(f"{label} & {row['rate']:.3f} & {row['kl']:.2f} & "
                      f"{row['kl_full']:.2f} & {row['escaped']:.3f} & {int(row['n']):,} \\\\")
        else:
            print(r"""\begin{table}[t]
\centering\small
\caption{Neighborhood preservation rate: fraction of related-fact prompts where
LoRA top-1 agrees with the base model's top-1 (1.0 = no collateral damage).}
\label{tab:locality}
\begin{tabular}{lcc}
\toprule
\textbf{Condition} & \textbf{Preservation rate} & \textbf{$N$ prompts} \\
\midrule""")
            for cond in [c for c in COND_ORDER if c != "synthetic"]:
                label = CONDITION_LABELS[cond]
                row = loc_sum[loc_sum["condition"] == cond]
                if row.empty:
                    continue
                row = row.iloc[0]
                print(f"{label} & {row['rate']:.3f} & {int(row['n']):,} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n[viz] LaTeX snippets above — paste into your .tex file.")


# ── Entry points ───────────────────────────────────────────────────────────────

def run_visualize(cfg, *_args) -> None:
    output_dir = Path(cfg.output_dir)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    latent_threshold = cfg.data.get("latent_logprob_threshold", _DEFAULT_LATENT_THRESHOLD)
    _run(results_dir, figures_dir, output_dir, latent_threshold)


def _run(results_dir: Path, figures_dir: Path, output_dir: Path | None = None,
        latent_threshold: float = _DEFAULT_LATENT_THRESHOLD) -> None:
    output_dir = output_dir if output_dir is not None else results_dir.parent
    fig_delta_logprob(results_dir, figures_dir)
    fig_patching(results_dir, figures_dir)
    fig_rank_ablation(results_dir, figures_dir)
    fig_locality(results_dir, figures_dir)
    fig_layer_top1_combined_comparison(results_dir, figures_dir)
    fig_layer_top1_by_condition_comparison(results_dir, figures_dir)
    fig_layer_top1_combined_comparison(results_dir, figures_dir, lens="tuned")
    fig_layer_top1_by_condition_comparison(results_dir, figures_dir, lens="tuned")
    fig_baseline_distribution(output_dir, figures_dir, latent_threshold)
    print_latex_tables(results_dir)


if __name__ == "__main__":
    from .utils import configure_stdout

    configure_stdout()
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/results")
    figures = results.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _run(results, figures, results.parent, _load_latent_threshold(results.parent))
