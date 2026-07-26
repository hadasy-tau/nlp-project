"""Configuration loading for the feasibility pipeline.

Everything user-tunable lives in a YAML file (see configs/). Any missing key
falls back to the dataclass default, so configs only need to state what they
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "EleutherAI/pythia-410m-deduped"
    # auto -> float16 on CUDA, float32 on CPU. Kaggle's T4/P100 do not
    # support bf16, so bf16 is deliberately not an option.
    dtype: str = "auto"  # auto | float16 | float32
    device: str = "auto"  # auto | cuda | cpu
    batch_size: int = 32


@dataclass
class DataConfig:
    name: str = "NeelNanda/counterfact-tracing"
    split: str = "train"
    prompt_column: str = "prompt"
    target_column: str = "target_true"
    # set to null in YAML if the dataset has no relation column
    relation_column: Optional[str] = "relation_id"
    # null = the whole dataset
    limit: Optional[int] = None


@dataclass
class LensConfig:
    # number of prompts for the logit/tuned lens smoke test
    sample_size: int = 50
    batch_size: int = 8
    top_k: int = 5


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lens: LensConfig = field(default_factory=LensConfig)
    output_dir: str = "outputs"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        lens=LensConfig(**raw.get("lens", {})),
        output_dir=raw.get("output_dir", "outputs"),
        seed=raw.get("seed", 0),
    )
