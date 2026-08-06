"""Pipeline CLI. Stages read/write artifacts under cfg.output_dir, so runs resume
across Kaggle sessions (persist output_dir to /kaggle/working).

Usage:
    python -m lora_lens.run --config configs/default.yaml --stages all
    python -m lora_lens.run --config configs/dev.yaml --stages score_base,train_lora \
        --set model.name=EleutherAI/pythia-1b-deduped --set lora.r=8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config, save_config
from .utils import (configure_stdout, disable_tf32, free_model, load_model, load_tokenizer,
                    resolve_device, set_seed)

STAGES = ["prepare_data", "make_synthetic", "score_base", "build_conditions", "train_lora",
          "analyze", "patch", "trajectory", "rank_ablation", "visualize"]


def _out(cfg) -> Path:
    p = Path(cfg.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _preflight(stages: list[str]) -> None:
    """Fail in seconds on known environment breakage, not 15 minutes into a run.

    peft's LoRA dispatcher calls is_torchao_available() while wrapping each layer,
    and that helper *raises* when torchao is installed but older than peft wants
    (Kaggle images ship an old one).
    """
    if "train_lora" not in stages:
        return
    try:
        from peft.import_utils import is_torchao_available
    except ImportError:
        return
    try:
        is_torchao_available()
    except ImportError as e:
        raise SystemExit(
            f"[preflight] peft cannot wrap layers with the installed torchao:\n  {e}\n"
            "This pipeline does not use torchao quantization, so removing it is the fix:\n"
            "    pip uninstall -y torchao"
        )


def stage_prepare_data(cfg, tokenizer, device):
    from .data import prepare_facts

    facts = prepare_facts(cfg, tokenizer)
    facts.to_parquet(_out(cfg) / "facts.parquet", index=False)


def stage_make_synthetic(cfg, tokenizer, device):
    from .data import make_synthetic

    facts = pd.read_parquet(_out(cfg) / "facts.parquet")
    synthetic = make_synthetic(facts, tokenizer, cfg)
    synthetic.to_parquet(_out(cfg) / "synthetic.parquet", index=False)


def stage_score_base(cfg, tokenizer, device):
    from .scoring import score_base_model

    facts = pd.read_parquet(_out(cfg) / "facts.parquet")
    # Defines known/latent/unknown; analyze asserts against it.
    model = load_model(cfg, device=device)
    scored = score_base_model(cfg, model, tokenizer, facts, device)
    free_model(model)
    scored.to_parquet(_out(cfg) / "facts_scored.parquet", index=False)


def stage_build_conditions(cfg, tokenizer, device):
    from .conditions import build_conditions

    scored = pd.read_parquet(_out(cfg) / "facts_scored.parquet")
    synthetic = pd.read_parquet(_out(cfg) / "synthetic.parquet")
    conditions = build_conditions(cfg, scored, synthetic)
    conditions.to_parquet(_out(cfg) / "conditions.parquet", index=False)


def stage_train_lora(cfg, tokenizer, device):
    from .training import train_lora

    conditions = pd.read_parquet(_out(cfg) / "conditions.parquet")
    train_lora(cfg, tokenizer, conditions, device)


def stage_analyze(cfg, tokenizer, device):
    from .analysis import run_analysis

    conditions = pd.read_parquet(_out(cfg) / "conditions.parquet")
    run_analysis(cfg, tokenizer, conditions, device)


def stage_patch(cfg, tokenizer, device):
    if not cfg.patching.enabled:
        print("[patch] disabled in config, skipping.")
        return
    from .patching import run_patching

    conditions = pd.read_parquet(_out(cfg) / "conditions.parquet")
    run_patching(cfg, tokenizer, conditions, device)


def stage_trajectory(cfg, tokenizer, device):
    from .trajectory import run_trajectory

    run_trajectory(cfg)


def stage_rank_ablation(cfg, tokenizer, device):
    from .rank_ablation import run_rank_ablation

    conditions = pd.read_parquet(_out(cfg) / "conditions.parquet")
    run_rank_ablation(cfg, tokenizer, conditions, device)


def stage_visualize(cfg, tokenizer, device):
    from .visualize import run_visualize

    run_visualize(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description="LoRA layer-wise factual prediction pipeline")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--stages", default="all",
                    help=f"Comma-separated subset of {STAGES}, or 'all'")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="Dotted config override, repeatable")
    args = ap.parse_args(argv)

    configure_stdout()
    cfg = load_config(args.config, tuple(args.overrides))
    set_seed(cfg.seed)
    disable_tf32()
    device = resolve_device(cfg)
    print(f"[run] model={cfg.model.name} device={device} output_dir={cfg.output_dir} "
          f"dtype={cfg.model.inference_dtype}")
    save_config(cfg, _out(cfg) / "config_resolved.yaml")

    stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        ap.error(f"Unknown stage(s) {unknown}; valid: {STAGES}")
    _preflight(stages)

    tokenizer = load_tokenizer(cfg)
    for stage in STAGES:  # canonical order regardless of input order
        if stage in stages:
            print(f"\n=== stage: {stage} ===")
            globals()[f"stage_{stage}"](cfg, tokenizer, device)


if __name__ == "__main__":
    main()
