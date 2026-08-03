"""Build the four experimental conditions with relation-stratified sampling.

known      — base model answers correctly (rank 1); does LoRA move it earlier?
latent     — real fact, base model has it in top 5 but not top 1 (rank 2-5);
             the representation exists but is suppressed.
unknown    — real fact, base model has no top-5 signal (rank > 5);
             the model genuinely lacks the association.
synthetic  — pseudo-entity facts (pure injection, no prior signal whatsoever).

Splitting old 'unknown' into latent vs unknown lets us test whether depth of
LoRA's update correlates with how absent the knowledge is: we expect
known < latent < unknown < synthetic in causal first-flip layer.

Paraphrase prompts are NEVER used in training.
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

    # Log-prob secondary gate for the latent bucket: rank 2-5 facts with near-zero
    # probability on the correct answer are not meaningfully "latent" — the signal is
    # there only by rank ordering noise, not genuine sub-threshold confidence.
    # Facts that pass the rank gate but fail the logprob gate are reclassified to unknown.
    # Configurable via data.latent_logprob_threshold (default -5.0, i.e. p ≈ 0.0067).
    latent_threshold = cfg.data.get("latent_logprob_threshold", -5.0)

    known     = scored[scored["top1_correct"]]
    in_top5   = scored["top5_correct"] & ~scored["top1_correct"]
    has_signal = in_top5 & (scored["answer_logprob"] > latent_threshold)
    latent    = scored[has_signal]
    unknown   = scored[~scored["top1_correct"] & ~has_signal]

    reclassified = int((in_top5 & ~has_signal).sum())
    if reclassified:
        print(f"[conditions] {reclassified} rank-2-5 facts reclassified to unknown "
              f"(answer_logprob <= {latent_threshold})")

    parts = []
    for name, pool in (("known", known), ("latent", latent),
                       ("unknown", unknown), ("synthetic", synthetic)):
        sample = _stratified_sample(pool, n, max_frac, rng).copy()
        sample["condition"] = name
        if "neighborhood_prompts" not in sample.columns:
            sample["neighborhood_prompts"] = [[] for _ in range(len(sample))]
        parts.append(sample)
        print(f"[conditions] {name}: {len(sample)} facts "
              f"({sample['relation'].nunique()} relations)")

    # Carry the ORIGINAL score_base pass's answer_rank/answer_logprob through
    # (renamed base_*) so the analyze stage can reconcile its own, independently
    # re-run base-model pass against the values that actually *defined* known/
    # latent/unknown, rather than silently trusting a second fp16 forward pass to
    # reproduce the first bit-for-bit (it doesn't always — see analysis.py).
    # NaN for synthetic facts, which were never scored by the base model.
    combined = pd.concat(parts, ignore_index=True)
    for c in ("answer_rank", "answer_logprob"):
        if c not in combined.columns:
            combined[c] = float("nan")
    combined = combined.rename(columns={"answer_rank": "base_answer_rank",
                                        "answer_logprob": "base_answer_logprob"})

    cols = ["fact_id", "condition", "relation", "subject", "prompt", "paraphrases",
            "neighborhood_prompts", "answer", "answer_token_id",
            "base_answer_rank", "base_answer_logprob"]
    return combined[cols]
