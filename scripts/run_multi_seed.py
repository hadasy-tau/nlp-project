"""Run the full pipeline at several seeds into separate output directories, so
the headline result (causal encoding-depth ordering) can be reported as a
cross-seed mean +/- std/range instead of a single seed run 2-3 times during
iterative development.

Each seed gets its own output_dir (<output_prefix><seed>), so runs don't
clobber each other and can be inspected/re-run independently.

Usage:
    python scripts/run_multi_seed.py --config configs/default.yaml
    python scripts/run_multi_seed.py --config configs/default.yaml \
        --seeds 42 123 7 2024 99 --output-prefix outputs_seed --stages all

After all seeds finish, aggregate with:
    python scripts/aggregate_seeds.py --output-prefix outputs_seed --seeds 42 123 7 2024 99
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SEEDS = [42, 123, 7, 2024, 99]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run lora_lens pipeline across multiple seeds")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help=f"Seeds to run (default: {DEFAULT_SEEDS})")
    ap.add_argument("--output-prefix", default="outputs_seed",
                    help="output_dir for seed S is '<prefix><S>' (default: outputs_seed)")
    ap.add_argument("--stages", default="all",
                    help="Comma-separated stage subset, or 'all' (default: all)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="Extra dotted config override, repeatable, "
                    "applied on top of --set seed=... --set output_dir=... for every run")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue to the next seed if one run fails, instead of aborting")
    args = ap.parse_args(argv)

    failures = []
    for seed in args.seeds:
        output_dir = f"{args.output_prefix}{seed}"
        cmd = [
            sys.executable, "-m", "lora_lens.run",
            "--config", args.config,
            "--stages", args.stages,
            "--set", f"seed={seed}",
            "--set", f"output_dir={output_dir}",
        ]
        for ov in args.overrides:
            cmd += ["--set", ov]

        print(f"\n{'=' * 70}\n[multi-seed] seed={seed} -> output_dir={output_dir}\n{'=' * 70}")
        print("[multi-seed] " + " ".join(cmd))
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"[multi-seed] seed={seed} FAILED (exit {result.returncode}, {elapsed:.0f}s)")
            failures.append(seed)
            if not args.keep_going:
                sys.exit(result.returncode)
        else:
            print(f"[multi-seed] seed={seed} completed in {elapsed:.0f}s")

    if failures:
        print(f"\n[multi-seed] DONE with failures on seeds: {failures}")
        sys.exit(1)
    print(f"\n[multi-seed] All {len(args.seeds)} seeds completed successfully.")
    print("Next: python scripts/aggregate_seeds.py --output-prefix "
          f"{args.output_prefix} --seeds {' '.join(str(s) for s in args.seeds)}")


if __name__ == "__main__":
    main()
