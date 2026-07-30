"""Base-model scoring: the decision-rule gate that must run before anything else.

Scores every candidate fact with the base model (final-layer only, no lens) and
splits known (top-1 correct) from unknown. If fewer than cfg.data.min_known_facts
are known, the run aborts with instructions to move one model size up.
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import batched, gather_last


@torch.no_grad()
def score_prompts(model, tokenizer, prompts: list[str], answer_token_ids: list[int],
                  batch_size: int, device) -> pd.DataFrame:
    """Final-layer answer metrics for each (prompt, answer-token) pair."""
    records = []
    items = list(zip(prompts, answer_token_ids))
    for chunk in tqdm(list(batched(items, batch_size)), desc="scoring"):
        texts = [p for p, _ in chunk]
        ans = torch.tensor([a for _, a in chunk], device=device)
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        logits = model(**enc).logits
        last = gather_last(logits, enc["attention_mask"])
        logprobs = F.log_softmax(last.float(), dim=-1)  # fp32 log-softmax always
        ans_lp = logprobs.gather(1, ans[:, None]).squeeze(1)
        rank = (logprobs > ans_lp[:, None]).sum(dim=1) + 1
        top_id = logprobs.argmax(dim=-1)
        for j in range(len(chunk)):
            r = int(rank[j].item())
            records.append({
                "answer_logprob": ans_lp[j].item(),
                "answer_rank": r,
                "top1_correct": r == 1,
                "top5_correct": r <= 5,
                "top_pred_token": tokenizer.decode(top_id[j]),
            })
    return pd.DataFrame(records)


def score_base_model(cfg, model, tokenizer, facts: pd.DataFrame, device) -> pd.DataFrame:
    scored = score_prompts(
        model, tokenizer,
        facts["prompt"].tolist(), facts["answer_token_id"].tolist(),
        cfg.scoring.batch_size, device,
    )
    out = pd.concat([facts.reset_index(drop=True), scored], axis=1)

    n_known  = int(out["top1_correct"].sum())
    n_latent = int((out["top5_correct"] & ~out["top1_correct"]).sum())
    n_unknown = int((~out["top5_correct"]).sum())
    print(f"\n[score] Base-model fact breakdown (out of {len(out)}):")
    print(f"  known  (rank=1):   {n_known}")
    print(f"  latent (rank 2-5): {n_latent}")
    print(f"  unknown (rank>5):  {n_unknown}")

    # Relation stratification report (pitfall 7): if the known set is dominated by
    # one relation, results describe that relation, not factual recall.
    known = out[out["top1_correct"]]
    if len(known):
        shares = known["relation"].value_counts(normalize=True)
        print("[score] Known-set relation shares (top 5):")
        print(shares.head(5).to_string())
        if shares.iloc[0] > cfg.data.max_relation_fraction:
            print(f"[score] WARNING: relation {shares.index[0]!r} is "
                  f"{shares.iloc[0]:.0%} of the known set "
                  f"(cap {cfg.data.max_relation_fraction:.0%}) — sampling will cap it.")

    # What do the wrong answers look like? (pitfall 8) A plausible wrong entity is a
    # competing belief to overwrite; punctuation means no representation at all.
    wrong = out[~out["top1_correct"]]
    if len(wrong):
        junk = wrong["top_pred_token"].str.strip().str.len().le(1).mean()
        print(f"[score] Wrong-answer preview: {junk:.0%} of wrong top-1 predictions are "
              "single-character/punctuation-like; most common:")
        print(wrong["top_pred_token"].value_counts().head(5).to_string())

    if n_known < cfg.data.min_known_facts:
        raise SystemExit(
            f"\n[score] DECISION RULE FAILED: only {n_known} known facts "
            f"(need >= {cfg.data.min_known_facts}).\n"
            "Move one model size up (410m -> 1b -> 1.4b), e.g.:\n"
            "  --set model.name=EleutherAI/pythia-1b-deduped\n"
            "or lower data.min_known_facts if you know what you're doing."
        )
    return out
