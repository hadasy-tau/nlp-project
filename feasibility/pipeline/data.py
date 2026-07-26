"""Dataset loading and filtering.

Two pitfalls handled here, both from the project handoff:
- Leading-space tokenization: after "...the city of" the model predicts the
  space-prefixed token (" Paris", not "Paris"). Targets are always tokenized
  as " " + target.
- Single-token objects only: answer rank at a single readout position is not
  well-defined for multi-token answers. Multi-token targets are dropped and
  the drop rate is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset

from .config import DataConfig


@dataclass
class FactItem:
    idx: int  # row index in the source dataset
    prompt: str
    target: str  # target string, no leading space
    target_token_id: int
    relation: Optional[str] = None


def load_fact_items(cfg: DataConfig, tokenizer) -> tuple[list[FactItem], dict]:
    print(f"loading dataset {cfg.name} (split={cfg.split})")
    ds = load_dataset(cfg.name, split=cfg.split)
    total_rows = len(ds)
    if cfg.limit is not None:
        ds = ds.select(range(min(cfg.limit, total_rows)))
    print(f"rows: {len(ds)} (of {total_rows} total)")

    prompts = ds[cfg.prompt_column]
    targets = [str(t).strip() for t in ds[cfg.target_column]]
    relations = (
        ds[cfg.relation_column]
        if cfg.relation_column and cfg.relation_column in ds.column_names
        else [None] * len(ds)
    )

    # Batch-tokenize all targets with a leading space.
    encoded = tokenizer([" " + t for t in targets], add_special_tokens=False)["input_ids"]

    items: list[FactItem] = []
    n_multi = 0
    for i, (prompt, target, rel, tok) in enumerate(zip(prompts, targets, relations, encoded)):
        if len(tok) != 1:
            n_multi += 1
            continue
        items.append(
            FactItem(
                idx=i,
                prompt=prompt.rstrip(),
                target=target,
                target_token_id=tok[0],
                relation=rel,
            )
        )

    stats = {
        "rows_loaded": len(ds),
        "single_token_kept": len(items),
        "multi_token_dropped": n_multi,
    }
    print(
        f"single-token filter: kept {len(items)}, "
        f"dropped {n_multi} multi-token targets "
        f"({100 * n_multi / max(len(ds), 1):.1f}%)"
    )
    return items, stats
