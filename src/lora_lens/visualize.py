"""Generate publication-ready figures and LaTeX table snippets from pipeline results.

Three figures, each designed to answer one specific question:
  fig_delta_logprob.pdf  — WHERE does LoRA's gain appear across layers?
  fig_patching.pdf       — WHERE is the update causally located, and how does it grow?
  fig_rank_ablation.pdf  — Does more capacity (rank) change WHERE or just HOW MANY?

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

    # Vertical markers at median causal first-flip layers (filled in after patching runs).
    for x, cond in [(14, "unknown"), (19, "synthetic")]:
        ax.axvline(x, color=COLORS[cond], linewidth=0.8, linestyle=":", alpha=0.7)

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
    ax_box.set_ylim(-1, 26)
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


# ── LaTeX table snippets ───────────────────────────────────────────────────────

def print_latex_tables(results_dir: Path) -> None:

    # ── Table 1: Main results ──────────────────────────────────────────────────
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        s = s[(s["lens"] == "logit") & (s["variant"].isin(["base", "final"]))]

        rows = [
            ("Known",          "train", "known",     "train"),
            ("Known",          "para.", "known",     "paraphrase"),
            ("Latent (top-5)", "train", "latent",    "train"),
            ("Latent (top-5)", "para.", "latent",    "paraphrase"),
            ("Unknown",        "train", "unknown",   "train"),
            ("Unknown",        "para.", "unknown",   "paraphrase"),
            ("Synthetic",      "train", "synthetic", "train"),
        ]

        print("\n% ── Table 1: Main results ─────────────────────────────────────")
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

    # ── Table 2: Causal patching ───────────────────────────────────────────────
    patch_path = results_dir / "patching.csv"
    if patch_path.exists():
        p = pd.read_csv(patch_path)
        p = p[(p["layer"] == -1) & (p["variant"] == "final")]
        pat = (p.groupby("condition")["first_flip_layer"]
                .agg(mean="mean", median="median", count="count")
                .reindex(COND_ORDER))

        print("\n% ── Table 2: Causal patching ──────────────────────────────────")
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
            print(f"{label} & {int(r['count'])}/100 & {r['mean']:.2f} & {r['median']:.0f} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # ── Table 3: Locality ──────────────────────────────────────────────────────
    loc_path = results_dir / "locality.csv"
    if loc_path.exists():
        loc = pd.read_csv(loc_path)
        loc_sum = (loc.groupby("condition")["preserved"]
                   .agg(rate="mean", n="count").reset_index())

        print("\n% ── Table 3: Locality ─────────────────────────────────────────")
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
    results_dir = Path(cfg.output_dir) / "results"
    figures_dir = Path(cfg.output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _run(results_dir, figures_dir)


def _run(results_dir: Path, figures_dir: Path) -> None:
    fig_delta_logprob(results_dir, figures_dir)
    fig_patching(results_dir, figures_dir)
    fig_rank_ablation(results_dir, figures_dir)
    print_latex_tables(results_dir)


if __name__ == "__main__":
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/results")
    figures = results.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _run(results, figures)
