"""Feasibility pipeline: confirm the whole stack works end-to-end before the
real experiments.

Steps (each timed):
  1. load model + tokenizer
  2. load dataset, filter to single-token targets
  3. score the base model on every fact -> success rate + known count
  4. logit lens on a sample of prompts
  5. tuned lens on the same sample (same hidden states)

Outputs (in <output_dir>/<run name>/):
  results.json          config echo, timings, scoring + lens summaries
  per_item_scores.csv   one row per fact — the known/unknown split for later
  lens_trajectories.json  per-layer target prob/rank for the lens sample

Usage:
  python run_pipeline.py --config configs/kaggle.yaml
  python run_pipeline.py --config configs/smoke.yaml --model EleutherAI/pythia-70m-deduped --limit 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from pipeline.config import load_config
from pipeline.data import load_fact_items
from pipeline.lenses import (
    collect_hidden_states,
    pick_lens_sample,
    run_logit_lens,
    run_tuned_lens,
)
from pipeline.modeling import load_model_and_tokenizer
from pipeline.scoring import print_scoring_report, score_items
from pipeline.timing import StepTimer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/kaggle.yaml")
    p.add_argument("--model", help="override model.name from the config")
    p.add_argument("--limit", type=int, help="override data.limit (n rows)")
    p.add_argument("--lens-sample", type=int, help="override lens.sample_size")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    if args.limit is not None:
        cfg.data.limit = args.limit
    if args.lens_sample is not None:
        cfg.lens.sample_size = args.lens_sample

    torch.manual_seed(cfg.seed)
    print("config:", json.dumps(cfg.to_dict(), indent=2))

    run_name = time.strftime("%Y%m%d-%H%M%S") + "-" + cfg.model.name.split("/")[-1]
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    timer = StepTimer()
    results: dict = {"config": cfg.to_dict()}

    with timer.step("1. load model"):
        model, tokenizer, device = load_model_and_tokenizer(cfg.model)

    with timer.step("2. load + filter data"):
        items, data_stats = load_fact_items(cfg.data, tokenizer)
        results["data"] = data_stats

    with timer.step("3. score base model (full dataset)"):
        scores, score_summary = score_items(
            model, tokenizer, items, cfg.model.batch_size, device
        )
        print_scoring_report(scores, score_summary)
        results["scoring"] = score_summary
        pd.DataFrame(scores).to_csv(out_dir / "per_item_scores.csv", index=False)

    sample = pick_lens_sample(scores, items, cfg.lens.sample_size, cfg.seed)
    print(f"\nlens sample: {len(sample)} prompts")

    lens_out: dict = {}
    with timer.step("4. logit lens (sample)"):
        hidden, final_logits = collect_hidden_states(
            model, tokenizer, sample, cfg.lens.batch_size, device
        )
        lens_out["logit_lens"] = run_logit_lens(
            model, hidden, final_logits, sample, tokenizer, cfg.lens.top_k
        )
        results["logit_lens"] = lens_out["logit_lens"]["summary"]

    with timer.step("5. tuned lens (sample)"):
        lens_out["tuned_lens"] = run_tuned_lens(
            model, hidden, final_logits, sample, tokenizer, cfg.lens.top_k
        )
        results["tuned_lens"] = lens_out["tuned_lens"]["summary"]

    with open(out_dir / "lens_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(lens_out, f, indent=2)

    print(timer.summary())
    results["timings_seconds"] = timer.records

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {out_dir}")


if __name__ == "__main__":
    main()
