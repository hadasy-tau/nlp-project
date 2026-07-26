"""Build the three experimental conditions with relation-stratified sampling.

known      — base model already answers correctly (does LoRA move it earlier?)
unknown    — real fact the base model gets wrong (learning an unstored fact)
synthetic  — pseudo-entity facts (pure injection, no prior signal)

Paraphrase prompts are carried along but are NEVER used in training — they are the
held-out generalization probes (pitfall 3: otherwise a late-layer shift may be
prompt memorization rather than fact encoding).
"""

from __future__ import annotations

import random

import pandas as pd


def _stratified_sample(df: pd.DataFrame, n: int | None, max_frac: float,
                       rng: random.Random) -> pd.DataFrame:
    """Sample up to n rows, capping each relation at max_frac of the target size."""
    if n is None or len(df) <= 0:
        return df
    n = min(n, len(df))
    cap = max(1, int(n * max_frac))
    groups = {rel: g.sample(frac=1.0, random_state=rng.randrange(2**32))
              for rel, g in df.groupby("relation")}
    # Round-robin across relations until n rows or every group is exhausted/capped.
    taken: dict[str, int] = {rel: 0 for rel in groups}
    picked = []
    progress = True
    while len(picked) < n and progress:
        progress = False
        for rel in sorted(groups, key=lambda r: taken[r]):
            g = groups[rel]
            if taken[rel] < min(cap, len(g)) and len(picked) < n:
                picked.append(g.iloc[taken[rel]])
                taken[rel] += 1
                progress = True
    return pd.DataFrame(picked).reset_index(drop=True)


def build_conditions(cfg, scored: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    n = cfg.data.get("n_per_condition")
    max_frac = cfg.data.max_relation_fraction

    known = scored[scored["top1_correct"]]
    unknown = scored[~scored["top1_correct"]]

    parts = []
    for name, pool in (("known", known), ("unknown", unknown), ("synthetic", synthetic)):
        sample = _stratified_sample(pool, n, max_frac, rng).copy()
        sample["condition"] = name
        parts.append(sample)
        print(f"[conditions] {name}: {len(sample)} facts "
              f"({sample['relation'].nunique()} relations)")

    cols = ["fact_id", "condition", "relation", "subject", "prompt", "paraphrases",
            "answer", "answer_token_id"]
    return pd.concat(parts, ignore_index=True)[cols]
