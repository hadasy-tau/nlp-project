"""Generate publication-ready figures and LaTeX table snippets from pipeline results.

H1/H2 figures, Graphs 1–10, and synthetic S1–S9. Lens-based figures are written
twice under figures/{logit,tuned}/; patching figures are lens-free under figures/.

Usage as a pipeline stage:
    python -m lora_lens.run --config configs/default.yaml --stages visualize

Standalone (pass results directory):
    python -m lora_lens.visualize results/run31_07_02
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
from scipy import stats as sstats

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

COLORS = {
    "known":     "#0072B2",
    "latent":    "#56B4E9",
    "unknown":   "#E69F00",
    "synthetic": "#D55E00",
    "existing":  "#0072B2",
    "base":      "#BBBBBB",
    "final":     "#0072B2",
}
CONDITION_LABELS = {
    "known":     "Known",
    "latent":    "Latent (top-5)",
    "unknown":   "Unknown",
    "synthetic": "Synthetic",
    "existing":  "Existing",
}
LINE_STYLES = {"known": "solid", "latent": (0, (5, 1)), "unknown": "dashed",
               "synthetic": "dotted", "existing": "solid"}
MARKERS     = {"known": "o", "latent": "D", "unknown": "s", "synthetic": "^",
               "existing": "o"}
COND_ORDER  = ["known", "latent", "unknown", "synthetic"]
EXISTING_CONDS = ["known", "latent", "unknown"]
MARKER_EVERY = 4
LENSES = ("logit", "tuned")
THREE_STATE = {"never": "never", "transient": "transient",
               "late_only": "survives", "persistent": "survives"}
THREE_STATE_ORDER = ["never", "transient", "survives"]


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_fig(fig, figures_dir: Path, stem: str, *,
              prefix: str = "new_version", lens: str | None = None) -> Path:
    """Save fig to figures_dir/{lens}/{prefix}_{stem}.pdf when lens is set,
    else figures_dir/{prefix}_{stem}.pdf."""
    if lens is not None:
        out_dir = figures_dir / lens
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = figures_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{prefix}_{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out}")
    return out


def _write_tex(figures_dir: Path, name: str, body: str, *,
               lens: str | None = None) -> Path:
    if lens is not None:
        out_dir = figures_dir / lens
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = figures_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.tex"
    out.write_text(body, encoding="utf-8")
    print(f"[viz] wrote {out}")
    return out


# ── Shared helpers (lens-agnostic) ────────────────────────────────────────────

def existing_vs_synthetic(condition: str | pd.Series) -> str | pd.Series:
    """Map condition → 'existing' (known∪latent∪unknown) or 'synthetic'."""
    if isinstance(condition, pd.Series):
        return condition.map(
            lambda c: "synthetic" if c == "synthetic" else "existing")
    return "synthetic" if condition == "synthetic" else "existing"


def three_state(traj_class: str | pd.Series) -> str | pd.Series:
    """Collapse traj_class to never / transient / survives (late_only∪persistent)."""
    if isinstance(traj_class, pd.Series):
        return traj_class.map(THREE_STATE)
    return THREE_STATE.get(traj_class, traj_class)


def detectable_prompt_idxs(traj: pd.DataFrame, lens: str,
                           prompt_type: str = "train") -> set:
    """Prompt idxs where the base model is top-1 at any layer (lens-detectable)."""
    base = traj[(traj["variant"] == "base") & (traj["lens"] == lens) &
                (traj["prompt_type"] == prompt_type)]
    return set(base.loc[base["first_layer"].notna(), "prompt_idx"])


def final_correct_prompt_idxs(traj: pd.DataFrame, variant: str, lens: str,
                              prompt_type: str = "train") -> set:
    """Prompt idxs where traj_class is late_only or persistent (final-layer correct)."""
    sub = traj[(traj["variant"] == variant) & (traj["lens"] == lens) &
               (traj["prompt_type"] == prompt_type)]
    return set(sub.loc[sub["traj_class"].isin(["late_only", "persistent"]), "prompt_idx"])


def _load_traj(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "trajectory.parquet"
    if path.exists():
        return pd.read_parquet(path)
    layerwise_path = results_dir / "layerwise.parquet"
    if not layerwise_path.exists():
        return None
    from .trajectory import per_prompt_trajectory
    print("[viz] trajectory.parquet missing — deriving from layerwise.parquet")
    return per_prompt_trajectory(pd.read_parquet(layerwise_path))


def _load_conditions(results_dir: Path, output_dir: Path | None = None
                     ) -> pd.DataFrame | None:
    """Locate conditions.parquet next to results or under output_dir."""
    candidates = [
        results_dir / "conditions.parquet",
        results_dir.parent / "conditions.parquet",
    ]
    if output_dir is not None:
        candidates.append(Path(output_dir) / "conditions.parquet")
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    return None


def _pooled_median_first_flip(results_dir: Path) -> float | None:
    path = results_dir / "patching.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    final = df[(df["layer"] == -1) & (df["variant"] == "final")]
    vals = final["first_flip_layer"].dropna()
    return float(vals.median()) if len(vals) else None


def _persistent_flip_layer(patch_df: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, fact) earliest L such that flipped on all layers L..N."""
    rows = patch_df[patch_df["layer"] >= 0].copy()
    if rows.empty:
        return pd.DataFrame(columns=["variant", "fact_id", "condition",
                                     "persistent_flip_layer"])
    n_layers = int(rows["layer"].max())
    out_rows = []
    for (variant, fact_id), g in rows.groupby(["variant", "fact_id"]):
        flipped = g.set_index("layer")["flipped"].reindex(range(n_layers + 1),
                                                          fill_value=False)
        persistent = None
        for L in range(n_layers + 1):
            if flipped.loc[L:].all():
                persistent = L
                break
        cond = g["condition"].iloc[0]
        out_rows.append({"variant": variant, "fact_id": fact_id,
                         "condition": cond, "persistent_flip_layer": persistent})
    return pd.DataFrame(out_rows)


def _layer_top1_curve(df: pd.DataFrame, variant: str, lens: str,
                      prompt_type: str = "train",
                      prompt_idxs: set | None = None,
                      conditions: list[str] | None = None) -> pd.DataFrame:
    sub = df[(df["variant"] == variant) & (df["lens"] == lens)
             & (df["prompt_type"] == prompt_type)]
    if conditions is not None:
        sub = sub[sub["condition"].isin(conditions)]
    if prompt_idxs is not None:
        sub = sub[sub["prompt_idx"].isin(prompt_idxs)]
    if sub.empty:
        return pd.DataFrame(columns=["condition", "layer", "pct_top1"])
    pct = (sub.groupby(["condition", "layer"], as_index=False)
             .agg(pct_top1=("in_top_1", "mean")))
    pct["pct_top1"] *= 100.0
    return pct


def _layer_top1_pooled(df: pd.DataFrame, variant: str, lens: str,
                       prompt_type: str = "train",
                       prompt_idxs: set | None = None,
                       conditions: list[str] | None = None) -> pd.DataFrame:
    sub = df[(df["variant"] == variant) & (df["lens"] == lens)
             & (df["prompt_type"] == prompt_type)]
    if conditions is not None:
        sub = sub[sub["condition"].isin(conditions)]
    if prompt_idxs is not None:
        sub = sub[sub["prompt_idx"].isin(prompt_idxs)]
    if sub.empty:
        return pd.DataFrame(columns=["layer", "pct_top1"])
    pooled = (sub.groupby("layer", as_index=False)
              .agg(pct_top1=("in_top_1", "mean")))
    pooled["pct_top1"] *= 100.0
    return pooled


def _draw_sankey(ax, flows: dict[tuple[str, str], int],
                 left_label: str = "base", right_label: str = "final") -> None:
    """Simple 3-state alluvial between base (left) and final (right)."""
    states = THREE_STATE_ORDER
    colors = {"never": "#999999", "transient": "#E69F00", "survives": "#0072B2"}
    left_tot = {s: sum(n for (a, b), n in flows.items() if a == s) for s in states}
    right_tot = {s: sum(n for (a, b), n in flows.items() if b == s) for s in states}
    total = max(sum(left_tot.values()), 1)

    def _ys(tots):
        y, out = 1.0, {}
        for s in states:
            h = tots[s] / total
            out[s] = (y - h, y)  # (bottom, top)
            y -= h + 0.04
        return out

    ly, ry = _ys(left_tot), _ys(right_tot)
    left_cursor = {s: ly[s][1] for s in states}
    right_cursor = {s: ry[s][1] for s in states}

    for (a, b), n in flows.items():
        if n <= 0:
            continue
        h = n / total
        y0_top, y1_top = left_cursor[a], right_cursor[b]
        y0_bot, y1_bot = y0_top - h, y1_top - h
        left_cursor[a] = y0_bot
        right_cursor[b] = y1_bot
        xs = np.linspace(0.15, 0.85, 40)
        top = y0_top + (y1_top - y0_top) * (xs - 0.15) / 0.7
        bot = y0_bot + (y1_bot - y0_bot) * (xs - 0.15) / 0.7
        ax.fill_between(xs, bot, top, color=colors[a], alpha=0.35, linewidth=0)

    for s in states:
        if left_tot[s]:
            b, t = ly[s]
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.02, b), 0.12, t - b, boxstyle="round,pad=0.01",
                facecolor=colors[s], edgecolor="white", alpha=0.9))
            ax.text(0.08, (b + t) / 2, f"{s}\n{left_tot[s]}", ha="center",
                    va="center", fontsize=6, color="white", fontweight="bold")
        if right_tot[s]:
            b, t = ry[s]
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.86, b), 0.12, t - b, boxstyle="round,pad=0.01",
                facecolor=colors[s], edgecolor="white", alpha=0.9))
            ax.text(0.92, (b + t) / 2, f"{s}\n{right_tot[s]}", ha="center",
                    va="center", fontsize=6, color="white", fontweight="bold")

    ax.text(0.08, 1.05, left_label, ha="center", fontsize=8, fontweight="bold")
    ax.text(0.92, 1.05, right_label, ha="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.axis("off")


# =============================================================================
# H1 / H2 FIGURES
# =============================================================================

def fig_delta_layer_hist(results_dir: Path, figures_dir: Path,
                         lens: str = "logit") -> None:
    """1×2 histogram of base−final shift for first_layer and settle_layer (pooled)."""
    traj = _load_traj(results_dir)
    if traj is None:
        print("[viz] no trajectory data — skipping delta_layer_hist.")
        return

    sub = traj[(traj["lens"] == lens) & (traj["prompt_type"] == "train") &
               (traj["variant"].isin(["base", "final"]))]
    # Pool existing conditions for H1/H2 (exclude synthetic from the pooled H1/H2
    # story? Plan says pool known/latent/unknown/synthetic everywhere in H1/H2
    # except fig_patching and tab_trajectory_transitions. So include all.)
    wide_first = sub.pivot_table(index="prompt_idx", columns="variant",
                                 values="first_layer")
    wide_settle = sub.pivot_table(index="prompt_idx", columns="variant",
                                  values="settle_layer")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for ax, wide, title in (
        (axes[0], wide_first, "first_layer"),
        (axes[1], wide_settle, "settle_layer"),
    ):
        if not {"base", "final"}.issubset(wide.columns):
            ax.set_visible(False)
            continue
        both = wide[["base", "final"]].dropna()
        if both.empty:
            ax.set_title(f"{title} (no data)")
            continue
        delta = both["base"] - both["final"]  # positive = earlier
        ax.hist(delta, bins=20, color="#0072B2", alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
        mean_d, med_d = float(delta.mean()), float(delta.median())
        pct_pos = 100.0 * (delta > 0).mean()
        ax.axvline(mean_d, color="#D55E00", linewidth=1.2, linestyle="--",
                   label=f"mean={mean_d:.1f}")
        ax.axvline(med_d, color="#E69F00", linewidth=1.2, linestyle=":",
                   label=f"median={med_d:.1f}")
        ax.set_xlabel(r"$\Delta$ = base $-$ final (positive = earlier)")
        ax.set_ylabel("Prompts")
        ax.set_title(title)
        ax.legend(fontsize=7, title=f"{pct_pos:.0f}% $\\Delta$>0  n={len(delta)}",
                  title_fontsize=7)

    fig.suptitle(f"Base→LoRA layer shift ({lens} lens, pooled)", fontsize=10, y=1.02)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    _save_fig(fig, figures_dir, "delta_layer_hist", lens=lens)


def fig_layer_hist(results_dir: Path, figures_dir: Path,
                   lens: str = "logit") -> None:
    """1×2 raw layer distributions: first_layer | settle_layer, base vs final overlay."""
    traj = _load_traj(results_dir)
    if traj is None:
        print("[viz] no trajectory data — skipping layer_hist.")
        return

    sub = traj[(traj["lens"] == lens) & (traj["prompt_type"] == "train") &
               (traj["variant"].isin(["base", "final"]))]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for ax, col, title in (
        (axes[0], "first_layer", "first_layer"),
        (axes[1], "settle_layer", "settle_layer"),
    ):
        for variant, color, label in (("base", "#BBBBBB", "Base"),
                                      ("final", "#0072B2", "LoRA")):
            vals = sub.loc[sub["variant"] == variant, col].dropna()
            if vals.empty:
                continue
            ax.hist(vals, bins=20, alpha=0.55, color=color, label=label,
                    edgecolor="white")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Prompts")
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle(f"Layer distributions ({lens} lens, pooled)", fontsize=10, y=1.02)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    _save_fig(fig, figures_dir, "layer_hist", lens=lens)


def fig_delta_logprob(results_dir: Path, figures_dir: Path,
                      lens: str = "logit") -> None:
    """Single pooled line: mean Δ log-prob (LoRA − base) per layer."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping delta logprob figure.")
        return

    df = pd.read_parquet(path)
    df = df[(df["lens"] == lens) & (df["variant"].isin(["base", "final"])) &
            (df["prompt_type"] == "train")]

    mean_lp = (df.groupby(["variant", "layer"])["answer_logprob"]
                 .mean().reset_index())
    base_lp = (mean_lp[mean_lp["variant"] == "base"]
               .rename(columns={"answer_logprob": "base_lp"}).drop(columns="variant"))
    final_lp = (mean_lp[mean_lp["variant"] == "final"]
                .rename(columns={"answer_logprob": "final_lp"}).drop(columns="variant"))
    delta = base_lp.merge(final_lp, on="layer")
    delta["delta_lp"] = delta["final_lp"] - delta["base_lp"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(delta["layer"], delta["delta_lp"], color="#0072B2",
            marker="o", markevery=MARKER_EVERY, markersize=4,
            label="Pooled Δ log-prob")

    med = _pooled_median_first_flip(results_dir)
    if med is not None:
        ax.axvline(med, color="#D55E00", linewidth=1.0, linestyle=":",
                   label=f"median first-flip={med:.0f}")

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Δ mean log-probability\n(LoRA − base)")
    ax.set_title(f"Where does LoRA add log-probability? ({lens})")
    ax.set_xlim(0, int(delta["layer"].max()) if not delta.empty else 24)
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "delta_logprob", lens=lens)


def fig_patching(results_dir: Path, figures_dir: Path) -> None:
    """Per-condition patching: left box of first-flip layers, right flip-count vs step."""
    path = results_dir / "patching.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping patching figure.")
        return

    df = pd.read_csv(path)
    summary = df[df["layer"] == -1].copy()

    fig, (ax_box, ax_dyn) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    final = summary[summary["variant"] == "final"].copy()
    present = [c for c in COND_ORDER if c in final["condition"].values]
    n_layers_patch = int(df["layer"][df["layer"] >= 0].max()) if (df["layer"] >= 0).any() else 24
    ax_box.set_ylim(-1, n_layers_patch + 2)

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
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(sub))
        ax_box.scatter(i + jitter, sub, color=COLORS[cond], alpha=0.35, s=8, zorder=3)
        ax_box.text(i, ax_box.get_ylim()[1],
                    f"{n_flip}/{n_total}", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS[cond], fontweight="bold")

    ax_box.set_xticks(list(range(len(present))))
    ax_box.set_xticklabels([CONDITION_LABELS[c] for c in present])
    ax_box.set_ylabel("First-flip layer")
    ax_box.set_title("Causal locus of LoRA update")

    n_cap = int(final[final["condition"] != "known"].groupby("condition").size().max()
               or 100)
    for cond in [c for c in COND_ORDER if c != "known"]:
        sub = (summary[(summary["condition"] == cond) & (summary["variant"] != "final")]
               .groupby("step")["first_flip_layer"]
               .count().reset_index(name="flipped"))
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
    ax_dyn.set_ylabel(f"Facts where patching flips\nprediction (out of {n_cap})")
    ax_dyn.set_title("Learning dynamics")
    ax_dyn.set_ylim(0, n_cap)
    ax_dyn.legend()

    fig.tight_layout(pad=0.4, w_pad=1.5)
    _save_fig(fig, figures_dir, "patching", lens=None)


def fig_patching_pooled(results_dir: Path, figures_dir: Path) -> None:
    """Same two panels as fig_patching, but pooled across conditions."""
    path = results_dir / "patching.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping patching_pooled.")
        return

    df = pd.read_csv(path)
    summary = df[df["layer"] == -1].copy()
    final = summary[summary["variant"] == "final"]
    flips = final["first_flip_layer"].dropna()
    n_total, n_flip = len(final), len(flips)

    fig, (ax_box, ax_dyn) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    n_layers_patch = int(df["layer"][df["layer"] >= 0].max()) if (df["layer"] >= 0).any() else 24
    ax_box.set_ylim(-1, n_layers_patch + 2)

    bp = ax_box.boxplot(flips, positions=[0], widths=0.45, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set(facecolor="#0072B2", alpha=0.75)
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(flips))
    ax_box.scatter(jitter, flips, color="#0072B2", alpha=0.35, s=8, zorder=3)
    ax_box.text(0, ax_box.get_ylim()[1], f"{n_flip}/{n_total}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_box.set_xticks([0])
    ax_box.set_xticklabels(["Pooled"])
    ax_box.set_ylabel("First-flip layer")
    ax_box.set_title("Causal locus (pooled)")

    by_step = (summary[summary["variant"] != "final"]
               .groupby("step")["first_flip_layer"].count()
               .reset_index(name="flipped"))
    final_count = final["first_flip_layer"].count()
    final_step = summary[summary["variant"] == "final"]["step"].max()
    by_step = pd.concat([by_step, pd.DataFrame([{"step": final_step,
                                                  "flipped": final_count}])],
                        ignore_index=True).sort_values("step")
    ax_dyn.plot(by_step["step"], by_step["flipped"], color="#0072B2",
                marker="o", markersize=5)
    ax_dyn.set_xlabel("Training step")
    ax_dyn.set_ylabel("Facts flipped (pooled)")
    ax_dyn.set_title("Learning dynamics (pooled)")

    fig.tight_layout(pad=0.4, w_pad=1.5)
    _save_fig(fig, figures_dir, "patching_pooled", lens=None)


def fig_patching_layer_hist(results_dir: Path, figures_dir: Path) -> None:
    """Pooled: (a) first_flip_layer; (b) persistent_flip_layer derived in viz."""
    path = results_dir / "patching.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping patching_layer_hist.")
        return

    df = pd.read_csv(path)
    final_sum = df[(df["layer"] == -1) & (df["variant"] == "final")]
    first = final_sum["first_flip_layer"].dropna()

    persist = _persistent_flip_layer(df[df["variant"] == "final"])
    persist_vals = persist["persistent_flip_layer"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    axes[0].hist(first, bins=20, color="#0072B2", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("first_flip_layer")
    axes[0].set_ylabel("Facts")
    axes[0].set_title(f"First flip (n={len(first)})")

    axes[1].hist(persist_vals, bins=20, color="#D55E00", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("persistent_flip_layer")
    axes[1].set_ylabel("Facts")
    axes[1].set_title(f"Persistent flip (n={len(persist_vals)})")

    fig.suptitle("Patching depth distributions (pooled)", fontsize=10, y=1.02)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    _save_fig(fig, figures_dir, "patching_layer_hist", lens=None)


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
        print("[viz] missing base or final rows — skipping layer top-1 comparison.")
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
    ax.set_title(f"Layer-wise accuracy — before vs after ({lens})")
    ax.set_xlim(-0.5, n_layers + 0.5)
    ax.set_xticks(layers[::max(1, len(layers) // 8)])
    ymax = max(merged["pct_top1_base"].max(), merged["pct_top1_final"].max())
    ax.set_ylim(0, min(105, ymax * 1.15 + 1))
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "layer_top1_combined_comparison", lens=lens)


def fig_moderator_scatter(results_dir: Path, figures_dir: Path,
                          lens: str = "logit") -> None:
    """x=baseline_logprob, y=base_settle−final_settle; regression + Spearman."""
    traj = _load_traj(results_dir)
    path = results_dir / "layerwise.parquet"
    if traj is None or not path.exists():
        print("[viz] missing trajectory/layerwise — skipping moderator_scatter.")
        return

    from .trajectory import moderator_data
    data = moderator_data(traj, pd.read_parquet(path), lens=lens)
    if data.empty:
        print("[viz] empty moderator_data — skipping moderator_scatter.")
        return

    pts = data[["baseline_logprob", "base_settle", "final_settle"]].dropna()
    if len(pts) < 5:
        print("[viz] too few settle pairs — skipping moderator_scatter.")
        return
    pts = pts.copy()
    pts["y"] = pts["base_settle"] - pts["final_settle"]  # positive = earlier

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.scatter(pts["baseline_logprob"], pts["y"], s=12, alpha=0.45, color="#0072B2")

    slope, intercept, r, p, _ = sstats.linregress(pts["baseline_logprob"], pts["y"])
    xs = np.linspace(pts["baseline_logprob"].min(), pts["baseline_logprob"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color="#D55E00", linewidth=1.5)
    rho, p_sp = sstats.spearmanr(pts["baseline_logprob"], pts["y"])
    r2 = r ** 2
    ax.text(0.05, 0.95,
            f"slope={slope:.2f}\nSpearman ρ={rho:.2f}\np={p_sp:.2g}\nR²={r2:.2f}",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Caption note: also report depth_agreement gap if available.
    vs_path = results_dir / "trajectory_vs_patching.csv"
    note = ""
    if vs_path.exists():
        vs = pd.read_csv(vs_path)
        if "lens" in vs.columns:
            vs = vs[vs["lens"] == lens]
        if not vs.empty:
            pair = vs[["settle_layer", "first_flip_layer"]].dropna()
            if len(pair) >= 5:
                rho2, _ = sstats.spearmanr(pair["settle_layer"], pair["first_flip_layer"])
                gap = float((pair["settle_layer"] - pair["first_flip_layer"]).median())
                note = f"\n(vs patching: ρ={rho2:.2f}, med gap={gap:.1f})"

    ax.set_xlabel("Baseline answer log-prob")
    ax.set_ylabel(r"Settle shift (base $-$ final)")
    ax.set_title(f"Moderator: prior signal vs settle shift ({lens}){note}")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "moderator_scatter", lens=lens)


# =============================================================================
# GRAPHS 1–10
# =============================================================================

def fig_layer_top1_detectable(results_dir: Path, figures_dir: Path,
                              lens: str = "logit") -> None:
    """Graph 1: layer-wise top-1 on previously detectable facts, base vs LoRA."""
    path = results_dir / "layerwise.parquet"
    traj = _load_traj(results_dir)
    if not path.exists() or traj is None:
        print("[viz] missing data — skipping layer_top1_detectable.")
        return

    idxs = detectable_prompt_idxs(traj, lens)
    if not idxs:
        print("[viz] no detectable prompts — skipping layer_top1_detectable.")
        return

    df = pd.read_parquet(path)
    base = _layer_top1_pooled(df, "base", lens, prompt_idxs=idxs)
    final = _layer_top1_pooled(df, "final", lens, prompt_idxs=idxs)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(base["layer"], base["pct_top1"], color="#BBBBBB", linestyle="--",
            label="Base", marker="o", markevery=MARKER_EVERY, markersize=3)
    ax.plot(final["layer"], final["pct_top1"], color="#0072B2",
            label="LoRA", marker="o", markevery=MARKER_EVERY, markersize=3)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("% top-1 correct")
    ax.set_title(f"Detectable facts (n={len(idxs)}, {lens})")
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "layer_top1_detectable", lens=lens)


def fig_layer_top1_all_facts(results_dir: Path, figures_dir: Path,
                             lens: str = "logit") -> None:
    """Graph 2: all-facts layer top-1 + by-condition companion."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping layer_top1_all_facts.")
        return
    df = pd.read_parquet(path)

    # Pooled base vs final
    base = _layer_top1_pooled(df, "base", lens)
    final = _layer_top1_pooled(df, "final", lens)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(base["layer"], base["pct_top1"], color="#BBBBBB", linestyle="--",
            label="Base", marker="o", markevery=MARKER_EVERY, markersize=3)
    ax.plot(final["layer"], final["pct_top1"], color="#0072B2",
            label="LoRA", marker="o", markevery=MARKER_EVERY, markersize=3)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("% top-1 correct")
    ax.set_title(f"All facts ({lens})")
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "layer_top1_all_facts", lens=lens)

    # By condition (final solid, base dashed)
    base_c = _layer_top1_curve(df, "base", lens)
    final_c = _layer_top1_curve(df, "final", lens)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for cond in COND_ORDER:
        b = base_c[base_c["condition"] == cond].sort_values("layer")
        f = final_c[final_c["condition"] == cond].sort_values("layer")
        if not b.empty:
            ax.plot(b["layer"], b["pct_top1"], color=COLORS[cond], linestyle="--",
                    linewidth=1.2, alpha=0.7)
        if not f.empty:
            ax.plot(f["layer"], f["pct_top1"], color=COLORS[cond],
                    linestyle=LINE_STYLES[cond], marker=MARKERS[cond],
                    markevery=MARKER_EVERY, markersize=3, label=CONDITION_LABELS[cond])
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("% top-1 correct")
    ax.set_title(f"All facts by condition ({lens})")
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "layer_top1_all_facts_by_condition", lens=lens)


def fig_final_vs_paraphrase(results_dir: Path, figures_dir: Path,
                            lens: str = "logit") -> None:
    """Graph 3: train vs paraphrase final-layer accuracy (+ by-condition)."""
    path = results_dir / "summary.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping final_vs_paraphrase.")
        return
    s = pd.read_csv(path)
    s = s[(s["lens"] == lens) & (s["variant"] == "final")]

    # Pooled-ish bar: mean across conditions for train vs para
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for i, ptype in enumerate(["train", "paraphrase"]):
        vals = s[s["prompt_type"] == ptype]["final_accuracy"]
        if vals.empty:
            continue
        ax.bar(i, vals.mean(), color="#0072B2" if ptype == "train" else "#56B4E9",
               alpha=0.85, label=ptype)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Train", "Paraphrase"])
    ax.set_ylabel("Acc@1 (mean over conditions)")
    ax.set_title(f"Train vs paraphrase Acc@1 ({lens})")
    ax.set_ylim(0, 1.05)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "final_vs_paraphrase", lens=lens)

    # By condition grouped bars
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(len(COND_ORDER))
    width = 0.35
    for offset, ptype, color in ((-width / 2, "train", "#0072B2"),
                                  (width / 2, "paraphrase", "#56B4E9")):
        vals = []
        for cond in COND_ORDER:
            row = s[(s["condition"] == cond) & (s["prompt_type"] == ptype)]
            vals.append(float(row["final_accuracy"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + offset, vals, width, color=color, alpha=0.85, label=ptype)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in COND_ORDER], fontsize=7)
    ax.set_ylabel("Acc@1")
    ax.set_title(f"Train vs paraphrase by condition ({lens})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "final_vs_paraphrase_by_condition", lens=lens)


def fig_paraphrase_layers_after_lora(results_dir: Path, figures_dir: Path,
                                     lens: str = "logit") -> None:
    """Graph 4: pooled only. Filter: base train correct; ≥1 base para incorrect;
    final para correct; require final train correct."""
    traj = _load_traj(results_dir)
    path = results_dir / "layerwise.parquet"
    if traj is None or not path.exists():
        print("[viz] missing data — skipping paraphrase_layers_after_lora.")
        return

    t = traj[(traj["lens"] == lens) & (traj["variant"].isin(["base", "final"]))]
    # Per fact: train and paraphrase classes
    train = t[t["prompt_type"] == "train"]
    para = t[t["prompt_type"] == "paraphrase"]
    if train.empty or para.empty:
        print("[viz] no paraphrase rows — skipping paraphrase_layers_after_lora.")
        return

    # Join on fact_id × variant
    tr = train.set_index(["fact_id", "variant"])
    pr = para.set_index(["fact_id", "variant"])
    # Base train correct = survives
    base_train_ok = set(tr.loc[(slice(None), "base"), :].query(
        "traj_class in ['late_only','persistent']").index.get_level_values(0))
    final_train_ok = set(tr.loc[(slice(None), "final"), :].query(
        "traj_class in ['late_only','persistent']").index.get_level_values(0))
    base_para = pr.loc[(slice(None), "base"), :]
    # ≥1 base para incorrect: fact has at least one paraphrase that does not survive
    # (or: fact's (aggregated) paraphrase is incorrect — we treat each para prompt;
    # use fact_ids where ANY paraphrase prompt is not final-correct in base)
    base_para_bad = set(base_para.loc[
        ~base_para["traj_class"].isin(["late_only", "persistent"])]
        .index.get_level_values(0))
    final_para_ok = set(pr.loc[(slice(None), "final"), :].query(
        "traj_class in ['late_only','persistent']").index.get_level_values(0))

    facts = base_train_ok & base_para_bad & final_para_ok & final_train_ok
    if not facts:
        print("[viz] Graph 4 filter matched 0 facts — skipping.")
        return

    # Map to paraphrase prompt_idxs for the final variant layer curve
    para_final = para[(para["variant"] == "final") & (para["fact_id"].isin(facts))]
    idxs = set(para_final["prompt_idx"])
    df = pd.read_parquet(path)
    curve = _layer_top1_pooled(df, "final", lens, prompt_type="paraphrase",
                               prompt_idxs=idxs)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(curve["layer"], curve["pct_top1"], color="#0072B2",
            marker="o", markevery=MARKER_EVERY, markersize=3)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("% top-1 (paraphrase, post-LoRA)")
    ax.set_title(f"Paraphrase layers after LoRA (n_facts={len(facts)}, {lens})")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "paraphrase_layers_after_lora", lens=lens)


def fig_paraphrase_layer_delay(results_dir: Path, figures_dir: Path,
                               lens: str = "logit") -> None:
    """Graph 5: Δ = L_para − L_orig; never-correct reported separately."""
    traj = _load_traj(results_dir)
    if traj is None:
        print("[viz] no trajectory — skipping paraphrase_layer_delay.")
        return

    sub = traj[(traj["lens"] == lens) & (traj["variant"] == "final")]
    train = sub[sub["prompt_type"] == "train"][["fact_id", "condition", "first_layer",
                                                 "traj_class"]].rename(
        columns={"first_layer": "L_orig", "traj_class": "cls_orig"})
    # One para row per fact: take min first_layer across paraphrases (earliest)
    para = (sub[sub["prompt_type"] == "paraphrase"]
            .groupby(["fact_id", "condition"], as_index=False)
            .agg(L_para=("first_layer", "min"),
                 cls_para=("traj_class",
                           lambda s: "never" if (s == "never").all() else "ok")))
    joined = train.merge(para, on=["fact_id", "condition"], how="inner")
    never = joined[(joined["L_orig"].isna()) | (joined["L_para"].isna())]
    both = joined[joined["L_orig"].notna() & joined["L_para"].notna()].copy()
    both["delta"] = both["L_para"] - both["L_orig"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    if not both.empty:
        ax.hist(both["delta"], bins=20, color="#0072B2", alpha=0.8, edgecolor="white")
        ax.axvline(both["delta"].mean(), color="#D55E00", linestyle="--",
                   label=f"mean={both['delta'].mean():.1f}")
    ax.set_xlabel(r"$\Delta = L_{para} - L_{orig}$")
    ax.set_ylabel("Facts")
    ax.set_title(f"Paraphrase layer delay ({lens})\n"
                 f"n={len(both)}, never-correct={len(never)}")
    if not both.empty:
        ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "paraphrase_layer_delay", lens=lens)

    # By condition
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    data, labels = [], []
    for cond in COND_ORDER:
        vals = both.loc[both["condition"] == cond, "delta"]
        if len(vals):
            data.append(vals)
            labels.append(CONDITION_LABELS[cond])
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, cond in zip(bp["boxes"],
                               [c for c in COND_ORDER
                                if len(both[both["condition"] == c])]):
            patch.set_facecolor(COLORS[cond])
            patch.set_alpha(0.75)
    ax.set_ylabel(r"$\Delta = L_{para} - L_{orig}$")
    ax.set_title(f"Paraphrase delay by condition ({lens})")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "paraphrase_layer_delay_by_condition", lens=lens)


def fig_fact_state_sankey(results_dir: Path, figures_dir: Path,
                          lens: str = "logit") -> None:
    """Graph 6: 3-state Sankey base→final (+ by-condition)."""
    traj = _load_traj(results_dir)
    if traj is None:
        print("[viz] no trajectory — skipping fact_state_sankey.")
        return

    sub = traj[(traj["lens"] == lens) & (traj["prompt_type"] == "train") &
               (traj["variant"].isin(["base", "final"]))].copy()
    sub["state"] = three_state(sub["traj_class"])

    def _flows(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
        wide = frame.pivot_table(index="prompt_idx", columns="variant",
                                 values="state", aggfunc="first").dropna()
        if wide.empty or not {"base", "final"}.issubset(wide.columns):
            return {}
        counts = wide.groupby(["base", "final"]).size()
        return {(a, b): int(n) for (a, b), n in counts.items()}

    # Pooled
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    _draw_sankey(ax, _flows(sub))
    ax.set_title(f"Fact-state transitions ({lens}, pooled)")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "fact_state_sankey", lens=lens)

    # By condition
    present = [c for c in COND_ORDER if c in sub["condition"].values]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3.0))
    if len(present) == 1:
        axes = [axes]
    for ax, cond in zip(axes, present):
        _draw_sankey(ax, _flows(sub[sub["condition"] == cond]))
        ax.set_title(CONDITION_LABELS[cond])
    fig.suptitle(f"Fact-state transitions by condition ({lens})", fontsize=10, y=1.02)
    fig.tight_layout(pad=0.4)
    _save_fig(fig, figures_dir, "fact_state_sankey_by_condition", lens=lens)


def fig_rank_final_accuracy(results_dir: Path, figures_dir: Path,
                            lens: str = "logit") -> None:
    """Graph 7: Acc@1 vs rank by condition (from rank_ablation_summary.csv)."""
    path = results_dir / "rank_ablation_summary.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping rank_final_accuracy.")
        return
    df = pd.read_csv(path)
    df = df[(df["lens"] == lens) & (df["prompt_type"] == "train")]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for cond in COND_ORDER:
        sub = df[df["condition"] == cond].sort_values("rank")
        if sub.empty:
            continue
        ax.plot(sub["rank"], sub["final_accuracy"], color=COLORS[cond],
                linestyle=LINE_STYLES[cond], marker=MARKERS[cond], markersize=4,
                label=CONDITION_LABELS[cond])
    ax.set_xlabel("LoRA rank $r$ (0 = base)")
    ax.set_ylabel("Acc@1")
    ax.set_title(f"Accuracy vs rank ({lens})")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "rank_final_accuracy", lens=lens)


def fig_rank_layer_accuracy(results_dir: Path, figures_dir: Path,
                            lens: str = "logit") -> None:
    """Graph 8: layer-wise accuracy faceted by condition from rank ablation."""
    path = results_dir / "rank_ablation_layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping rank_layer_accuracy.")
        return
    df = pd.read_parquet(path)
    df = df[(df["lens"] == lens) & (df["prompt_type"] == "train")]
    # Prefer final-ish variants: rank adapters are labeled r4/r8/... plus base
    present = [c for c in COND_ORDER if c in df["condition"].values]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(3.0 * len(present), 2.8),
                             sharey=True)
    if len(present) == 1:
        axes = [axes]
    ranks = sorted(df["rank"].unique()) if "rank" in df.columns else sorted(
        df["variant"].unique())
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(ranks)))

    for ax, cond in zip(axes, present):
        sub = df[df["condition"] == cond]
        for color, rank in zip(cmap, ranks):
            if "rank" in sub.columns:
                rsub = sub[sub["rank"] == rank]
            else:
                rsub = sub[sub["variant"] == rank]
            curve = (rsub.groupby("layer")["in_top_1"].mean() * 100).reset_index()
            if curve.empty:
                continue
            ax.plot(curve["layer"], curve["in_top_1"], color=color,
                    label=str(rank), linewidth=1.2)
        ax.set_title(CONDITION_LABELS[cond])
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("% top-1")
    axes[-1].legend(fontsize=6, title="rank", title_fontsize=6)
    fig.suptitle(f"Rank ablation layer accuracy ({lens})", fontsize=10, y=1.02)
    fig.tight_layout(pad=0.4)
    _save_fig(fig, figures_dir, "rank_layer_accuracy", lens=lens)


def fig_delta_detectable(results_dir: Path, figures_dir: Path,
                         lens: str = "logit") -> None:
    """Graph 9: ΔAcc (final−base) per layer on Graph 1 detectable subset."""
    path = results_dir / "layerwise.parquet"
    traj = _load_traj(results_dir)
    if not path.exists() or traj is None:
        print("[viz] missing data — skipping delta_detectable.")
        return
    idxs = detectable_prompt_idxs(traj, lens)
    if not idxs:
        print("[viz] no detectable prompts — skipping delta_detectable.")
        return
    df = pd.read_parquet(path)
    base = _layer_top1_pooled(df, "base", lens, prompt_idxs=idxs)
    final = _layer_top1_pooled(df, "final", lens, prompt_idxs=idxs)
    merged = base.merge(final, on="layer", suffixes=("_base", "_final"))
    merged["delta"] = merged["pct_top1_final"] - merged["pct_top1_base"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(merged["layer"], merged["delta"], color="#0072B2",
            marker="o", markevery=MARKER_EVERY, markersize=3)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(r"$\Delta$ Acc@1 (pp)")
    ax.set_title(f"ΔAcc on detectable subset (n={len(idxs)}, {lens})")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "delta_detectable", lens=lens)


def fig_delta_by_relation(results_dir: Path, figures_dir: Path,
                          lens: str = "logit",
                          output_dir: Path | None = None) -> None:
    """Graph 10: ΔAcc by relation (+ heatmap). Needs conditions.parquet."""
    conds = _load_conditions(results_dir, output_dir)
    path = results_dir / "layerwise.parquet"
    if conds is None:
        print("[viz] conditions.parquet not found — skipping delta_by_relation.")
        return
    if not path.exists():
        print(f"[viz] {path} not found — skipping delta_by_relation.")
        return

    df = pd.read_parquet(path)
    n_layers = int(df["layer"].max())
    final_layer = df[(df["lens"] == lens) & (df["prompt_type"] == "train") &
                     (df["layer"] == n_layers) &
                     (df["variant"].isin(["base", "final"]))]
    joined = final_layer.merge(conds[["fact_id", "relation"]], on="fact_id",
                               how="inner")
    if joined.empty:
        print("[viz] relation join empty — skipping delta_by_relation.")
        return

    acc = (joined.groupby(["relation", "variant"])["in_top_1"]
           .mean().unstack("variant"))
    if not {"base", "final"}.issubset(acc.columns):
        print("[viz] missing base/final for relations — skipping.")
        return
    acc["delta"] = acc["final"] - acc["base"]
    acc = acc.sort_values("delta", ascending=False)

    # Bar chart of Δ by relation
    fig, ax = plt.subplots(figsize=(max(4.0, 0.25 * len(acc)), 3.0))
    ax.bar(range(len(acc)), acc["delta"] * 100, color="#0072B2", alpha=0.85)
    ax.set_xticks(range(len(acc)))
    ax.set_xticklabels(acc.index, rotation=90, fontsize=6)
    ax.set_ylabel(r"$\Delta$ Acc@1 (pp)")
    ax.set_title(f"ΔAcc by relation ({lens})")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "delta_by_relation", lens=lens)

    # Heatmap: relations × layer of Δ top-1
    lw = df[(df["lens"] == lens) & (df["prompt_type"] == "train") &
            (df["variant"].isin(["base", "final"]))]
    lw = lw.merge(conds[["fact_id", "relation"]], on="fact_id", how="inner")
    pivot = (lw.groupby(["relation", "variant", "layer"])["in_top_1"]
             .mean().unstack("variant"))
    if not {"base", "final"}.issubset(pivot.columns):
        return
    pivot["delta"] = (pivot["final"] - pivot["base"]) * 100
    heat = pivot["delta"].unstack("layer")
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.22 * len(heat))))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(heat.values)),
                   vmax=np.nanmax(np.abs(heat.values)))
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat.index, fontsize=6)
    ax.set_xlabel("Layer")
    ax.set_title(f"ΔAcc heatmap by relation × layer ({lens})")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Δ pp")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "delta_by_relation_heatmap", lens=lens)


# =============================================================================
# SYNTHETIC FACTS FIGURES (S1–S9)
# Existing = known+latent+unknown vs synthetic. Filenames: new_version_synthetic_*.pdf
# =============================================================================

def _tag_existing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group"] = existing_vs_synthetic(out["condition"])
    return out


def fig_synthetic_layer_top1_existing_vs_synthetic(
        results_dir: Path, figures_dir: Path, lens: str = "logit") -> None:
    """S1."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping S1.")
        return
    df = _tag_existing(pd.read_parquet(path))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for group, color in (("existing", COLORS["existing"]),
                         ("synthetic", COLORS["synthetic"])):
        for variant, ls in (("base", "--"), ("final", "-")):
            sub = df[(df["group"] == group) & (df["variant"] == variant) &
                     (df["lens"] == lens) & (df["prompt_type"] == "train")]
            if sub.empty:
                continue
            curve = sub.groupby("layer")["in_top_1"].mean() * 100
            ax.plot(curve.index, curve.values, color=color, linestyle=ls,
                    label=f"{group}/{variant}", linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("% top-1")
    ax.set_title(f"S1: existing vs synthetic ({lens})")
    ax.legend(fontsize=6)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "layer_top1_existing_vs_synthetic",
              prefix="new_version_synthetic", lens=lens)


def fig_synthetic_layer_delta_existing_vs_synthetic(
        results_dir: Path, figures_dir: Path, lens: str = "logit") -> None:
    """S2."""
    path = results_dir / "layerwise.parquet"
    if not path.exists():
        print(f"[viz] {path} not found — skipping S2.")
        return
    df = _tag_existing(pd.read_parquet(path))
    df = df[(df["lens"] == lens) & (df["prompt_type"] == "train") &
            (df["variant"].isin(["base", "final"]))]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for group, color in (("existing", COLORS["existing"]),
                         ("synthetic", COLORS["synthetic"])):
        sub = df[df["group"] == group]
        mean_lp = sub.groupby(["variant", "layer"])["answer_logprob"].mean().unstack(0)
        if not {"base", "final"}.issubset(mean_lp.columns):
            continue
        ax.plot(mean_lp.index, mean_lp["final"] - mean_lp["base"],
                color=color, label=group, marker="o", markevery=MARKER_EVERY,
                markersize=3)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ mean log-prob")
    ax.set_title(f"S2: Δ log-prob existing vs synthetic ({lens})")
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "layer_delta_existing_vs_synthetic",
              prefix="new_version_synthetic", lens=lens)


def fig_synthetic_first_layer_existing_vs_synthetic(
        results_dir: Path, figures_dir: Path, lens: str = "logit") -> None:
    """S3: first_layer distributions; never-correct separate."""
    traj = _load_traj(results_dir)
    if traj is None:
        print("[viz] no trajectory — skipping S3.")
        return
    sub = traj[(traj["lens"] == lens) & (traj["prompt_type"] == "train") &
               (traj["variant"] == "final")].copy()
    sub["group"] = existing_vs_synthetic(sub["condition"])
    never_frac = sub.groupby("group")["traj_class"].apply(
        lambda s: (s == "never").mean())

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    data, labels = [], []
    for group in ("existing", "synthetic"):
        vals = sub.loc[sub["group"] == group, "first_layer"].dropna()
        if len(vals):
            data.append(vals)
            nf = float(never_frac.get(group, 0))
            labels.append(f"{group}\n(never={nf:.0%})")
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, group in zip(bp["boxes"],
                                [g for g in ("existing", "synthetic")
                                 if len(sub.loc[sub["group"] == g, "first_layer"].dropna())]):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.75)
    ax.set_ylabel("first_layer (final)")
    ax.set_title(f"S3: first-layer existing vs synthetic ({lens})")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "first_layer_existing_vs_synthetic",
              prefix="new_version_synthetic", lens=lens)


def fig_synthetic_acc_over_checkpoints(
        results_dir: Path, figures_dir: Path, lens: str = "logit") -> None:
    """S4: Acc vs step; skip if no intermediate steps."""
    path = results_dir / "summary.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping S4.")
        return
    s = pd.read_csv(path)
    s = s[(s["lens"] == lens) & (s["prompt_type"] == "train")].copy()
    steps = s[s["variant"].str.startswith("step_", na=False)]
    if steps.empty:
        print("[viz] no intermediate checkpoints — skipping S4.")
        return
    s["group"] = existing_vs_synthetic(s["condition"])
    # Include final
    use = s[s["variant"].str.startswith("step_", na=False) | (s["variant"] == "final")]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for group, color in (("existing", COLORS["existing"]),
                         ("synthetic", COLORS["synthetic"])):
        g = (use[use["group"] == group]
             .groupby("step")["final_accuracy"].mean().reset_index())
        ax.plot(g["step"], g["final_accuracy"], color=color, marker="o",
                markersize=4, label=group)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Acc@1")
    ax.set_title(f"S4: Acc over checkpoints ({lens})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "acc_over_checkpoints",
              prefix="new_version_synthetic", lens=lens)


def fig_synthetic_first_layer_over_checkpoints(
        results_dir: Path, figures_dir: Path, lens: str = "logit") -> None:
    """S5: mean first_layer vs step; annotate frac_never."""
    path = results_dir / "summary.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping S5.")
        return
    s = pd.read_csv(path)
    s = s[(s["lens"] == lens) & (s["prompt_type"] == "train")].copy()
    if not s["variant"].str.startswith("step_", na=False).any():
        print("[viz] no intermediate checkpoints — skipping S5.")
        return
    s["group"] = existing_vs_synthetic(s["condition"])
    use = s[s["variant"].str.startswith("step_", na=False) | (s["variant"] == "final")]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for group, color in (("existing", COLORS["existing"]),
                         ("synthetic", COLORS["synthetic"])):
        g = (use[use["group"] == group]
             .groupby("step")
             .agg(mean_first=("mean_first_layer", "mean"),
                  frac_never=("frac_never_top1", "mean"))
             .reset_index())
        ax.plot(g["step"], g["mean_first"], color=color, marker="o",
                markersize=4, label=group)
        if not g.empty:
            last = g.iloc[-1]
            ax.annotate(f"never={last['frac_never']:.0%}",
                        (last["step"], last["mean_first"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color=color)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean first_layer")
    ax.set_title(f"S5: first-layer over checkpoints ({lens})")
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "first_layer_over_checkpoints",
              prefix="new_version_synthetic", lens=lens)


def fig_synthetic_paraphrase_existing_vs_synthetic(
        results_dir: Path, figures_dir: Path, lens: str = "logit") -> None:
    """S6: paraphrase Acc existing vs synthetic."""
    path = results_dir / "summary.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping S6.")
        return
    s = pd.read_csv(path)
    s = s[(s["lens"] == lens) & (s["variant"] == "final") &
          (s["prompt_type"] == "paraphrase")].copy()
    if s.empty:
        print("[viz] no paraphrase rows — skipping S6.")
        return
    s["group"] = existing_vs_synthetic(s["condition"])
    g = s.groupby("group")["final_accuracy"].mean()
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    groups = [x for x in ("existing", "synthetic") if x in g.index]
    ax.bar(range(len(groups)), [g[x] for x in groups],
           color=[COLORS[x] for x in groups], alpha=0.85)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Paraphrase Acc@1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"S6: paraphrase Acc ({lens})")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "paraphrase_existing_vs_synthetic",
              prefix="new_version_synthetic", lens=lens)


def fig_synthetic_sankey_existing_vs_synthetic(
        results_dir: Path, figures_dir: Path, lens: str = "logit") -> None:
    """S7: 3-state Sankey, two panels."""
    traj = _load_traj(results_dir)
    if traj is None:
        print("[viz] no trajectory — skipping S7.")
        return
    sub = traj[(traj["lens"] == lens) & (traj["prompt_type"] == "train") &
               (traj["variant"].isin(["base", "final"]))].copy()
    sub["state"] = three_state(sub["traj_class"])
    sub["group"] = existing_vs_synthetic(sub["condition"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, group in zip(axes, ("existing", "synthetic")):
        g = sub[sub["group"] == group]
        wide = g.pivot_table(index="prompt_idx", columns="variant",
                             values="state", aggfunc="first").dropna()
        flows = {}
        if not wide.empty and {"base", "final"}.issubset(wide.columns):
            counts = wide.groupby(["base", "final"]).size()
            flows = {(a, b): int(n) for (a, b), n in counts.items()}
        _draw_sankey(ax, flows)
        ax.set_title(group)
    fig.suptitle(f"S7: state transitions ({lens})", fontsize=10, y=1.02)
    fig.tight_layout(pad=0.4)
    _save_fig(fig, figures_dir, "sankey_existing_vs_synthetic",
              prefix="new_version_synthetic", lens=lens)


def fig_synthetic_patching_existing_vs_synthetic(
        results_dir: Path, figures_dir: Path) -> None:
    """S8: flip rate vs layer — lens-free."""
    path = results_dir / "patching.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping S8.")
        return
    df = pd.read_csv(path)
    df = df[(df["variant"] == "final") & (df["layer"] >= 0)].copy()
    df["group"] = existing_vs_synthetic(df["condition"])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for group, color in (("existing", COLORS["existing"]),
                         ("synthetic", COLORS["synthetic"])):
        sub = df[df["group"] == group]
        rate = sub.groupby("layer")["flipped"].mean()
        ax.plot(rate.index, rate.values, color=color, marker="o",
                markevery=MARKER_EVERY, markersize=3, label=group)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Flip rate")
    ax.set_title("S8: patching flip rate existing vs synthetic")
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "patching_existing_vs_synthetic",
              prefix="new_version_synthetic", lens=None)


def fig_synthetic_layer_by_relation(
        results_dir: Path, figures_dir: Path, lens: str = "logit",
        output_dir: Path | None = None) -> None:
    """S9: synthetic-only × relation (+ delta)."""
    conds = _load_conditions(results_dir, output_dir)
    path = results_dir / "layerwise.parquet"
    if conds is None:
        print("[viz] conditions.parquet not found — skipping S9.")
        return
    if not path.exists():
        print(f"[viz] {path} not found — skipping S9.")
        return

    syn_conds = conds[conds["condition"] == "synthetic"][["fact_id", "relation"]]
    df = pd.read_parquet(path)
    df = df[(df["condition"] == "synthetic") & (df["lens"] == lens) &
            (df["prompt_type"] == "train") &
            (df["variant"].isin(["base", "final"]))]
    joined = df.merge(syn_conds, on="fact_id", how="inner")
    if joined.empty:
        print("[viz] synthetic×relation join empty — skipping S9.")
        return

    # Mean first-layer-ish: final-layer Acc by relation for final variant
    n_layers = int(joined["layer"].max())
    final_acc = (joined[(joined["variant"] == "final") & (joined["layer"] == n_layers)]
                 .groupby("relation")["in_top_1"].mean().sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(max(4.0, 0.25 * len(final_acc)), 3.0))
    ax.bar(range(len(final_acc)), final_acc.values * 100,
           color=COLORS["synthetic"], alpha=0.85)
    ax.set_xticks(range(len(final_acc)))
    ax.set_xticklabels(final_acc.index, rotation=90, fontsize=6)
    ax.set_ylabel("Acc@1 (%)")
    ax.set_title(f"S9: synthetic Acc by relation ({lens})")
    fig.tight_layout(pad=0.5)
    _save_fig(fig, figures_dir, "layer_by_relation",
              prefix="new_version_synthetic", lens=lens)

    # Delta Acc by relation
    acc = (joined[joined["layer"] == n_layers]
           .groupby(["relation", "variant"])["in_top_1"].mean().unstack("variant"))
    if {"base", "final"}.issubset(acc.columns):
        acc["delta"] = (acc["final"] - acc["base"]) * 100
        acc = acc.sort_values("delta", ascending=False)
        fig, ax = plt.subplots(figsize=(max(4.0, 0.25 * len(acc)), 3.0))
        ax.bar(range(len(acc)), acc["delta"], color=COLORS["synthetic"], alpha=0.85)
        ax.set_xticks(range(len(acc)))
        ax.set_xticklabels(acc.index, rotation=90, fontsize=6)
        ax.set_ylabel(r"$\Delta$ Acc@1 (pp)")
        ax.set_title(f"S9: synthetic ΔAcc by relation ({lens})")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        fig.tight_layout(pad=0.5)
        _save_fig(fig, figures_dir, "layer_by_relation_delta",
                  prefix="new_version_synthetic", lens=lens)


def _run_synthetic_figures(results_dir: Path, figures_dir: Path,
                           output_dir: Path | None = None) -> None:
    for lens in LENSES:
        fig_synthetic_layer_top1_existing_vs_synthetic(results_dir, figures_dir, lens)
        fig_synthetic_layer_delta_existing_vs_synthetic(results_dir, figures_dir, lens)
        fig_synthetic_first_layer_existing_vs_synthetic(results_dir, figures_dir, lens)
        fig_synthetic_acc_over_checkpoints(results_dir, figures_dir, lens)
        fig_synthetic_first_layer_over_checkpoints(results_dir, figures_dir, lens)
        fig_synthetic_paraphrase_existing_vs_synthetic(results_dir, figures_dir, lens)
        fig_synthetic_sankey_existing_vs_synthetic(results_dir, figures_dir, lens)
        fig_synthetic_layer_by_relation(results_dir, figures_dir, lens, output_dir)
    fig_synthetic_patching_existing_vs_synthetic(results_dir, figures_dir)


# =============================================================================
# LaTeX tables
# =============================================================================

def _table1_body(s: pd.DataFrame, lens: str) -> str:
    """Pooled Table 1: one row train + one row paraphrase for a single lens."""
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        rf"\caption{{{lens.capitalize()}-lens results before and after LoRA "
        r"(pooled across conditions). "
        r"\emph{First layer $\Delta$} = shift in mean first-appearance layer "
        r"(negative = earlier). Para.\ = held-out paraphrase prompts.}",
        rf"\label{{tab:main_results_{lens}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Acc@1}} & "
        r"\multicolumn{2}{c}{\textbf{Mean log-prob}} & \textbf{First layer} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Prompts} & Base & LoRA & Base & LoRA & $\Delta$ \\",
        r"\midrule",
    ]
    for ptype, label in (("train", "train"), ("paraphrase", "para.")):
        b = s[(s["variant"] == "base") & (s["prompt_type"] == ptype)]
        f = s[(s["variant"] == "final") & (s["prompt_type"] == ptype)]
        if b.empty or f.empty:
            continue
        # Weight by n_prompts when pooling
        def _wavg(frame, col):
            if "n_prompts" in frame.columns and frame["n_prompts"].sum():
                return np.average(frame[col], weights=frame["n_prompts"])
            return frame[col].mean()
        acc_b, acc_f = _wavg(b, "final_accuracy"), _wavg(f, "final_accuracy")
        lp_b, lp_f = _wavg(b, "mean_final_logprob"), _wavg(f, "mean_final_logprob")
        fl_b, fl_f = _wavg(b, "mean_first_layer"), _wavg(f, "mean_first_layer")
        shift = fl_f - fl_b
        sign = "+" if shift >= 0 else ""
        shift_str = "--" if pd.isna(shift) else f"{sign}{shift:.1f}"
        lines.append(
            f"{label} & {acc_b:.3f} & {acc_f:.3f} & "
            f"{lp_b:.2f} & {lp_f:.2f} & {shift_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _table1b_body(s_all: pd.DataFrame) -> str | None:
    cmp_df = s_all[(s_all["variant"] == "final") & (s_all["prompt_type"] == "train")]
    if "tuned" not in cmp_df["lens"].unique():
        return None
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Final-checkpoint logit-lens vs.\ tuned-lens agreement on "
        r"training prompts.}",
        r"\label{tab:lens_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Acc@1}} & "
        r"\multicolumn{2}{c}{\textbf{Mean first layer}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Condition} & Logit & Tuned & Logit & Tuned \\",
        r"\midrule",
    ]
    for cond in COND_ORDER:
        lg = cmp_df[(cmp_df["condition"] == cond) & (cmp_df["lens"] == "logit")]
        tn = cmp_df[(cmp_df["condition"] == cond) & (cmp_df["lens"] == "tuned")]
        if lg.empty or tn.empty:
            continue
        lg, tn = lg.iloc[0], tn.iloc[0]
        lines.append(
            f"{CONDITION_LABELS[cond]} & "
            f"{lg['final_accuracy']:.3f} & {tn['final_accuracy']:.3f} & "
            f"{lg['mean_first_layer']:.1f} & {tn['mean_first_layer']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _table2_body(results_dir: Path) -> str | None:
    patch_path = results_dir / "patching.csv"
    if not patch_path.exists():
        return None
    p = pd.read_csv(patch_path)
    p = p[(p["layer"] == -1) & (p["variant"] == "final")]
    # Pooled stats
    n_flipped = int(p["first_flip_layer"].notna().sum())
    n_total = len(p)
    mean = p["first_flip_layer"].mean()
    median = p["first_flip_layer"].median()
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Activation patching at the final LoRA checkpoint (pooled).}",
        r"\label{tab:patching}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Set} & \textbf{Flipped} & \textbf{Mean first-flip} & "
        r"\textbf{Median} \\",
        r"\midrule",
        f"Pooled & {n_flipped}/{n_total} & "
        f"{'--' if pd.isna(mean) else f'{mean:.2f}'} & "
        f"{'--' if pd.isna(median) else f'{median:.0f}'} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _tab_traj_class_shares(results_dir: Path, lens: str) -> str | None:
    path = results_dir / "trajectory_summary.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping tab_trajectory_class_shares ({lens}).")
        return None
    s = pd.read_csv(path)
    s = s[(s["lens"] == lens) & (s["prompt_type"] == "train") &
          (s["variant"].isin(["base", "final"]))]
    if s.empty:
        return None
    # Pool across conditions
    from .trajectory import CLASS_ORDER
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        rf"\caption{{Trajectory class shares ({lens} lens, train prompts, pooled).}}",
        rf"\label{{tab:traj_class_shares_{lens}}}",
        r"\begin{tabular}{l" + "c" * len(CLASS_ORDER) + "}",
        r"\toprule",
        r"\textbf{Variant} & " + " & ".join(CLASS_ORDER) + r" \\",
        r"\midrule",
    ]
    for variant in ("base", "final"):
        sub = s[s["variant"] == variant]
        totals = {c: sub[c].sum() if c in sub.columns else 0 for c in CLASS_ORDER}
        n = sum(totals.values()) or 1
        fracs = " & ".join(f"{totals[c] / n:.3f}" for c in CLASS_ORDER)
        lines.append(f"{variant} & {fracs} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _tab_traj_transitions(results_dir: Path, lens: str) -> str | None:
    path = results_dir / "trajectory_transitions.csv"
    if not path.exists():
        print(f"[viz] {path} not found — skipping tab_trajectory_transitions ({lens}).")
        return None
    t = pd.read_csv(path)
    if "lens" in t.columns:
        t = t[t["lens"] == lens]
    from .trajectory import CLASS_ORDER
    blocks = []
    for cond in COND_ORDER:
        mat = t[(t["condition"] == cond) & (t["base_class"] != "ANY")]
        if mat.empty:
            continue
        pivot = mat.pivot_table(index="base_class", columns="final_class",
                                values="n", aggfunc="sum").reindex(
            index=CLASS_ORDER, columns=CLASS_ORDER).fillna(0)
        # Highlight transient → persistent
        highlight = int(pivot.loc["transient", "persistent"]) if (
            "transient" in pivot.index and "persistent" in pivot.columns) else 0
        mcn = t[(t["condition"] == cond) & (t["base_class"] == "ANY")]
        mcn_str = ""
        if not mcn.empty:
            row = mcn.iloc[0]
            mcn_str = (f" McNemar ANY$\\to$SURVIVES: "
                       f"n01={int(row.get('n01', 0))}, n10={int(row.get('n10', 0))}, "
                       f"$p$={row.get('p_value', float('nan')):.3g}.")
        lines = [
            r"\begin{table}[t]",
            r"\centering\small",
            rf"\caption{{Base$\to$final trajectory transitions: "
            rf"{CONDITION_LABELS[cond]} ({lens}). "
            rf"transient$\to$persistent $n$={highlight}.{mcn_str}}}",
            rf"\label{{tab:traj_trans_{cond}_{lens}}}",
            r"\begin{tabular}{l" + "c" * len(CLASS_ORDER) + "}",
            r"\toprule",
            r"base$\backslash$final & " + " & ".join(CLASS_ORDER) + r" \\",
            r"\midrule",
        ]
        for bc in CLASS_ORDER:
            vals = " & ".join(str(int(pivot.loc[bc, fc])) for fc in CLASS_ORDER)
            lines.append(f"{bc} & {vals} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        blocks.append("\n".join(lines))
    return ("\n\n".join(blocks) + "\n") if blocks else None


def print_latex_tables(results_dir: Path, figures_dir: Path) -> None:
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        s_all = pd.read_csv(summary_path)

        for lens in LENSES:
            s = s_all[(s_all["lens"] == lens) &
                      (s_all["variant"].isin(["base", "final"]))]
            body = _table1_body(s, lens)
            print(f"\n% -- Table 1 ({lens}) ------------------------------------------")
            print(body)
            _write_tex(figures_dir, "tab_main_results", body, lens=lens)

        body1b = _table1b_body(s_all)
        if body1b:
            print("\n% -- Table 1b: Logit vs tuned ---------------------------------")
            print(body1b)
            _write_tex(figures_dir, "tab_lens_comparison", body1b, lens=None)
        else:
            print("\n[viz] no tuned-lens rows — skipping Table 1b.")
    else:
        print(f"[viz] {summary_path} not found — skipping Table 1/1b.")

    body2 = _table2_body(results_dir)
    if body2:
        print("\n% -- Table 2: Causal patching (pooled) ---------------------------")
        print(body2)
        _write_tex(figures_dir, "tab_patching", body2, lens=None)
    else:
        print("[viz] patching.csv not found — skipping Table 2.")

    for lens in LENSES:
        body = _tab_traj_class_shares(results_dir, lens)
        if body:
            print(f"\n% -- Trajectory class shares ({lens}) -------------------------")
            print(body)
            _write_tex(figures_dir, "tab_trajectory_class_shares", body, lens=lens)

        body = _tab_traj_transitions(results_dir, lens)
        if body:
            print(f"\n% -- Trajectory transitions ({lens}) --------------------------")
            print(body)
            _write_tex(figures_dir, "tab_trajectory_transitions", body, lens=lens)

    print("\n[viz] LaTeX snippets printed and written under figures/.")


# =============================================================================
# Entry points
# =============================================================================

def run_visualize(cfg, *_args) -> None:
    output_dir = Path(cfg.output_dir)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _run(results_dir, figures_dir, output_dir)


def _run(results_dir: Path, figures_dir: Path,
         output_dir: Path | None = None) -> None:
    output_dir = output_dir if output_dir is not None else results_dir.parent

    # 1. H1/H2 figures
    for lens in LENSES:
        fig_delta_layer_hist(results_dir, figures_dir, lens)
        fig_layer_hist(results_dir, figures_dir, lens)
        fig_delta_logprob(results_dir, figures_dir, lens)
        fig_layer_top1_combined_comparison(results_dir, figures_dir, lens)
        fig_moderator_scatter(results_dir, figures_dir, lens)
    fig_patching(results_dir, figures_dir)
    fig_patching_pooled(results_dir, figures_dir)
    fig_patching_layer_hist(results_dir, figures_dir)
    print_latex_tables(results_dir, figures_dir)

    # 2. Graphs 1–10
    for lens in LENSES:
        fig_layer_top1_detectable(results_dir, figures_dir, lens)
        fig_layer_top1_all_facts(results_dir, figures_dir, lens)
        fig_final_vs_paraphrase(results_dir, figures_dir, lens)
        fig_paraphrase_layers_after_lora(results_dir, figures_dir, lens)
        fig_paraphrase_layer_delay(results_dir, figures_dir, lens)
        fig_fact_state_sankey(results_dir, figures_dir, lens)
        fig_rank_final_accuracy(results_dir, figures_dir, lens)
        fig_rank_layer_accuracy(results_dir, figures_dir, lens)
        fig_delta_detectable(results_dir, figures_dir, lens)
        fig_delta_by_relation(results_dir, figures_dir, lens, output_dir)

    # 3. Synthetic S1–S9
    _run_synthetic_figures(results_dir, figures_dir, output_dir)


if __name__ == "__main__":
    from .utils import configure_stdout

    configure_stdout()
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/results")
    figures = results.parent / "figures" if results.name == "results" else results / "figures"
    # Flat results dir like results/run31_07_02 → figures under that dir
    if (results / "layerwise.parquet").exists():
        figures = results / "figures"
        output_dir = results  # conditions.parquet may sit beside results files
    else:
        output_dir = results.parent
    figures.mkdir(parents=True, exist_ok=True)
    _run(results, figures, output_dir)
