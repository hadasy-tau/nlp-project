"""Aggregate per-seed results produced by scripts/run_multi_seed.py into a
cross-seed mean +/- std (and range) for the headline causal-patching result,
replacing single-seed point estimates with genuine multi-seed robustness
evidence.

Usage:
    python scripts/aggregate_seeds.py --output-prefix outputs_seed \
        --seeds 42 123 7 2024 99

Reads, for each seed S, '<output-prefix><S>/results/patching.csv' (and, if
present, 'stats_summary.csv' for the per-seed bootstrap CI). Writes a combined
'<output-prefix>_aggregate.csv' next to the per-seed directories and prints a
report + LaTeX snippet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _utf8_stdout() -> None:
    """UTF-8 stdout so a cp1252 console cannot kill the report on a print."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

COND_ORDER = ["known", "latent", "unknown", "synthetic"]
CONDITION_LABELS = {
    "known": "Known", "latent": "Latent (top-5)",
    "unknown": "Unknown", "synthetic": "Synthetic",
}


def _per_seed_summary(output_dir: Path, seed: int) -> pd.DataFrame:
    path = output_dir / "results" / "patching.csv"
    if not path.exists():
        print(f"[aggregate] WARNING: {path} not found — skipping seed {seed}.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    final = df[(df["layer"] == -1) & (df["variant"] == "final")]
    rows = []
    for cond, g in final.groupby("condition"):
        rows.append({
            "seed": seed, "condition": cond,
            "median_first_flip": g["first_flip_layer"].median(),
            "mean_first_flip": g["first_flip_layer"].mean(),
            "flip_rate": g["flipped"].mean(),
            "n_total": len(g),
            "n_flipped": int(g["first_flip_layer"].notna().sum()),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="Aggregate multi-seed lora_lens results")
    ap.add_argument("--output-prefix", default="outputs_seed",
                    help="output_dir for seed S is '<prefix><S>' (must match run_multi_seed.py)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args(argv)

    per_seed = []
    for seed in args.seeds:
        per_seed.append(_per_seed_summary(Path(f"{args.output_prefix}{seed}"), seed))
    per_seed_df = pd.concat([d for d in per_seed if not d.empty], ignore_index=True)

    if per_seed_df.empty:
        raise SystemExit("[aggregate] No per-seed results found — run scripts/run_multi_seed.py first.")

    combined_path = Path(f"{args.output_prefix}_per_seed.csv")
    per_seed_df.to_csv(combined_path, index=False)
    print(f"[aggregate] Per-seed results written to {combined_path}")

    agg = (per_seed_df.groupby("condition")
          .agg(median_mean=("median_first_flip", "mean"),
               median_std=("median_first_flip", "std"),
               median_min=("median_first_flip", "min"),
               median_max=("median_first_flip", "max"),
               flip_rate_mean=("flip_rate", "mean"),
               flip_rate_std=("flip_rate", "std"),
               n_seeds=("seed", "count"))
          .reindex(COND_ORDER)
          .dropna(how="all")
          .reset_index())
    agg_path = Path(f"{args.output_prefix}_aggregate.csv")
    agg.to_csv(agg_path, index=False)

    print(f"\n[aggregate] Cross-seed summary (n={len(args.seeds)} seeds: {args.seeds}):")
    print(agg.to_string(index=False))
    print(f"\n[aggregate] Written to {agg_path}")

    # n_total, not n_seeds: the caption reports facts per condition.
    n_facts = int(per_seed_df["n_total"].max()) if "n_total" in per_seed_df else 0
    print("\n% -- Table: Multi-seed robustness of causal encoding depth --------")
    print(r"""\begin{table}[t]
\centering\small
\caption{Median causal first-flip layer, mean $\pm$ std across %d independent
seeds (full pipeline re-run per seed, including data sampling and LoRA
training). Flip rate = fraction of the %d sampled facts per condition that
flip at any layer, mean $\pm$ std across seeds.}
\label{tab:multiseed}
\begin{tabular}{lcc}
\toprule
\textbf{Condition} & \textbf{Median first-flip layer} & \textbf{Flip rate} \\
\midrule""" % (len(args.seeds), n_facts))
    for _, row in agg.iterrows():
        std = row["median_std"] if not np.isnan(row["median_std"]) else 0.0
        fr_std = row["flip_rate_std"] if not np.isnan(row["flip_rate_std"]) else 0.0
        print(f"{CONDITION_LABELS.get(row['condition'], row['condition'])} & "
              f"{row['median_mean']:.1f} $\\pm$ {std:.1f} "
              f"[{row['median_min']:.0f}, {row['median_max']:.0f}] & "
              f"{row['flip_rate_mean']:.2f} $\\pm$ {fr_std:.2f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
