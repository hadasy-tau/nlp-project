"""Generate publication-ready figures and LaTeX table snippets from pipeline results.

Reads from <output_dir>/results/ — run after all pipeline stages complete.

Outputs written to <output_dir>/figures/:
  fig_layer_accuracy.pdf   — layer-wise acc@1 for base vs LoRA (3 conditions)
  fig_patching_dynamics.pdf — patching flip-count evolution across training steps
  fig_rank_ablation.pdf    — accuracy vs LoRA rank (train + paraphrase)

LaTeX snippets for tables are printed to stdout.

Usage as a pipeline stage:
    python -m lora_lens.run --config configs/default.yaml --stages visualize

Or standalone (pass results directory as argument):
    python -m lora_lens.visualize /content/outputs/results
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
})

COLORS = {"known": "#1f77b4", "unknown": "#ff7f0e", "synthetic": "#2ca02c"}
CONDITION_LABELS = {"known": "Known", "unknown": "Unknown", "synthetic": "Synthetic"}
LINE_STYLES = {"base": (0, (4, 2)), "final": "solid"}  # dashed for base, solid for LoRA


# ── Figure 1: Layer-wise accuracy ─────────────────────────────────────────────

def fig_layer_accuracy(results_dir: Path, figures_dir: Path) -> None:
    """3-panel plot: acc@1 across layers for base vs final LoRA, logit lens."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping layer accuracy figure.")
        return

    df = pd.read_parquet(path)
    df = df[(df["lens"] == "logit") & (df["variant"].isin(["base", "final"]))]

    # Compute acc@1 = fraction where answer is rank 1, per layer.
    acc = (df.assign(correct=df["answer_rank"] == 1)
             .groupby(["variant", "condition", "prompt_type", "layer"])["correct"]
             .mean().reset_index(name="acc1"))

    # Panel layout: Known (para), Unknown (para), Synthetic (train)
    panels = [
        ("known",     "paraphrase", "Known — paraphrase prompts"),
        ("unknown",   "paraphrase", "Unknown — paraphrase prompts"),
        ("synthetic", "train",      "Synthetic — training prompts"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), sharey=True)
    for ax, (cond, ptype, title) in zip(axes, panels):
        sub = acc[(acc["condition"] == cond) & (acc["prompt_type"] == ptype)]
        for variant in ["base", "final"]:
            row = sub[sub["variant"] == variant].sort_values("layer")
            if row.empty:
                continue
            label = "Base" if variant == "base" else "LoRA"
            ax.plot(row["layer"], row["acc1"],
                    color=COLORS[cond],
                    linestyle=LINE_STYLES[variant],
                    label=label)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend()
    axes[0].set_ylabel("Acc@1 (logit lens)")

    fig.tight_layout(pad=0.4, w_pad=1.2)
    out = figures_dir / "fig_layer_accuracy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── Figure 2: Patching dynamics ───────────────────────────────────────────────

def fig_patching_dynamics(results_dir: Path, figures_dir: Path) -> None:
    """Line chart: how many facts flip at each training step, by condition."""
    path = results_dir / "patching.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping patching dynamics figure.")
        return

    df = pd.read_csv(path)
    summary = (df[df["layer"] == -1]
               .groupby(["step", "condition"])
               .agg(count=("first_flip_layer", "count"))
               .reset_index())

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for cond in ["known", "unknown", "synthetic"]:
        sub = summary[summary["condition"] == cond].sort_values("step")
        if sub.empty:
            continue
        ax.plot(sub["step"], sub["count"],
                color=COLORS[cond],
                marker="o", markersize=3,
                label=CONDITION_LABELS[cond])

    ax.set_xlabel("Training step")
    ax.set_ylabel("Facts where patching flips\nprediction (out of 100)")
    ax.set_title("Causal flip count vs training step")
    ax.legend()
    ax.set_ylim(0, 105)

    fig.tight_layout(pad=0.4)
    out = figures_dir / "fig_patching_dynamics.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── Figure 3: Rank ablation ───────────────────────────────────────────────────

def fig_rank_ablation(results_dir: Path, figures_dir: Path) -> None:
    """Two-panel: train accuracy and paraphrase accuracy vs LoRA rank."""
    path = results_dir / "rank_ablation_summary.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping rank ablation figure.")
        return

    df = pd.read_csv(path)
    df = df[df["lens"] == "logit"]

    fig, (ax_train, ax_para) = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=False)

    for cond in ["known", "unknown", "synthetic"]:
        train_sub = (df[(df["condition"] == cond) & (df["prompt_type"] == "train")]
                     .sort_values("rank"))
        if not train_sub.empty:
            ax_train.plot(train_sub["rank"], train_sub["final_accuracy"],
                          color=COLORS[cond], marker="o", markersize=4,
                          label=CONDITION_LABELS[cond])

    for cond in ["known", "unknown"]:  # synthetic has no paraphrases
        para_sub = (df[(df["condition"] == cond) & (df["prompt_type"] == "paraphrase")]
                    .sort_values("rank"))
        if not para_sub.empty:
            ax_para.plot(para_sub["rank"], para_sub["final_accuracy"],
                         color=COLORS[cond], marker="o", markersize=4,
                         label=CONDITION_LABELS[cond])

    for ax, title in [(ax_train, "Training prompts"), (ax_para, "Paraphrase prompts")]:
        ax.set_xlabel("LoRA rank $r$")
        ax.set_ylabel("Acc@1 (logit lens)")
        ax.set_title(title)
        ax.set_xticks([0, 4, 8, 16, 32])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(-0.03, 1.03)
        ax.legend()

    fig.tight_layout(pad=0.4, w_pad=1.2)
    out = figures_dir / "fig_rank_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")


# ── LaTeX table snippets ───────────────────────────────────────────────────────

def print_latex_tables(results_dir: Path) -> None:
    """Print ready-to-paste LaTeX table code for the main results."""

    # ── Table 1: Main results ──────────────────────────────────────────────────
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        s = s[(s["lens"] == "logit") & (s["variant"].isin(["base", "final"]))]

        rows = [
            ("Known",     "train",      "known",     "train"),
            ("Known",     "paraphrase", "known",     "paraphrase"),
            ("Unknown",   "train",      "unknown",   "train"),
            ("Unknown",   "paraphrase", "unknown",   "paraphrase"),
            ("Synthetic", "train",      "synthetic", "train"),
        ]

        print("\n% ── Table 1: Main results ────────────────────────────────────────")
        print(r"""\begin{table}[t]
\centering
\small
\caption{Layer-wise logit-lens results before and after LoRA fine-tuning (Pythia-410m-deduped, $r=16$).
\emph{First layer} = mean layer at which the correct answer first appears as top-1.
Para.\ = held-out paraphrase prompts never seen in training.}
\label{tab:main_results}
\begin{tabular}{llccccc}
\toprule
 & & \multicolumn{2}{c}{\textbf{Acc@1}} & \multicolumn{2}{c}{\textbf{Mean logprob}} & \textbf{First layer} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\textbf{Condition} & \textbf{Prompts} & Base & LoRA & Base & LoRA & $\Delta$ \\
\midrule""")
        for label_cond, label_ptype, cond, ptype in rows:
            b = s[(s["variant"] == "base") & (s["condition"] == cond) &
                  (s["prompt_type"] == ptype)]
            f = s[(s["variant"] == "final") & (s["condition"] == cond) &
                  (s["prompt_type"] == ptype)]
            if b.empty or f.empty:
                continue
            b, f = b.iloc[0], f.iloc[0]
            fl_shift = f["mean_first_layer"] - b["mean_first_layer"]
            sign = "+" if fl_shift >= 0 else ""
            print(
                f"{label_cond} & {label_ptype} & "
                f"{b['final_accuracy']:.3f} & {f['final_accuracy']:.3f} & "
                f"{b['mean_final_logprob']:.2f} & {f['mean_final_logprob']:.2f} & "
                f"{sign}{fl_shift:.1f} \\\\"
            )
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
                .reindex(["known", "unknown", "synthetic"]))

        print("\n% ── Table 2: Causal patching ─────────────────────────────────────")
        print(r"""\begin{table}[t]
\centering
\small
\caption{Activation patching results at the final LoRA checkpoint.
\emph{Count} = number of facts (out of 100) where patching any layer flips the
base model's prediction to the target answer.
\emph{First-flip layer} = earliest layer at which the flip occurs.}
\label{tab:patching}
\begin{tabular}{lccc}
\toprule
\textbf{Condition} & \textbf{Count flipped} & \textbf{Mean first-flip layer} & \textbf{Median} \\
\midrule""")
        for cond, label in [("known", "Known"), ("unknown", "Unknown"), ("synthetic", "Synthetic")]:
            if cond not in pat.index or pd.isna(pat.loc[cond, "mean"]):
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

        print("\n% ── Table 3: Locality ────────────────────────────────────────────")
        print(r"""\begin{table}[t]
\centering
\small
\caption{Neighborhood preservation rate after LoRA fine-tuning.
A rate of 1.0 means no side effects on related facts; lower values indicate
collateral disruption.}
\label{tab:locality}
\begin{tabular}{lcc}
\toprule
\textbf{Condition} & \textbf{Preservation rate} & \textbf{N prompts} \\
\midrule""")
        for _, row in loc_sum.iterrows():
            label = CONDITION_LABELS.get(row["condition"], row["condition"])
            print(f"{label} & {row['rate']:.3f} & {int(row['n'])} \\\\")
        print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n[viz] LaTeX snippets printed above — paste into your .tex file.")


# ── Entry points ───────────────────────────────────────────────────────────────

def run_visualize(cfg, *_args) -> None:
    """Pipeline stage entry point."""
    results_dir = Path(cfg.output_dir) / "results"
    figures_dir = Path(cfg.output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _run(results_dir, figures_dir)


def _run(results_dir: Path, figures_dir: Path) -> None:
    fig_layer_accuracy(results_dir, figures_dir)
    fig_patching_dynamics(results_dir, figures_dir)
    fig_rank_ablation(results_dir, figures_dir)
    print_latex_tables(results_dir)


if __name__ == "__main__":
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/results")
    figures = results.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _run(results, figures)
