"""Step 1: score the base model on the full fact set.

For each prompt, read the logits at the last real token (explicit gather via
the attention mask — right padding) and check whether the target token is the
top-1 prediction. Log-softmax is computed in fp32 even when the model runs in
fp16.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import FactItem


@torch.no_grad()
def score_items(
    model,
    tokenizer,
    items: list[FactItem],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    results: list[dict] = []

    for start in tqdm(range(0, len(items), batch_size), desc="scoring"):
        batch = items[start : start + batch_size]
        enc = tokenizer(
            [it.prompt for it in batch], return_tensors="pt", padding=True
        ).to(device)

        logits = model(**enc).logits  # [B, T, V]

        last_idx = enc["attention_mask"].sum(dim=1) - 1  # last real token
        rows = torch.arange(len(batch), device=device)
        final_logits = logits[rows, last_idx]  # [B, V]

        logprobs = F.log_softmax(final_logits.float(), dim=-1)
        top1_ids = logprobs.argmax(dim=-1)
        target_ids = torch.tensor(
            [it.target_token_id for it in batch], device=device
        )
        target_lp = logprobs[rows, target_ids]
        # rank 1 = top prediction
        ranks = (logprobs > target_lp.unsqueeze(1)).sum(dim=1) + 1

        for i, it in enumerate(batch):
            results.append(
                {
                    "idx": it.idx,
                    "prompt": it.prompt,
                    "target": it.target,
                    "relation": it.relation,
                    "target_token_id": it.target_token_id,
                    "top1_token_id": top1_ids[i].item(),
                    "top1_token": tokenizer.decode([top1_ids[i].item()]),
                    "correct": bool(top1_ids[i] == target_ids[i]),
                    "target_prob": target_lp[i].exp().item(),
                    "target_rank": ranks[i].item(),
                }
            )

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    summary = {
        "n_scored": n,
        "n_correct_top1": n_correct,
        "success_rate": n_correct / n if n else 0.0,
    }

    # Per-relation breakdown (pitfall: don't let one easy relation dominate).
    if any(r["relation"] for r in results):
        by_rel_total = Counter(r["relation"] for r in results)
        by_rel_correct = Counter(r["relation"] for r in results if r["correct"])
        summary["by_relation"] = {
            rel: {"n": by_rel_total[rel], "correct": by_rel_correct.get(rel, 0)}
            for rel in sorted(by_rel_total)
        }

    return results, summary


def print_scoring_report(results: list[dict], summary: dict, known_threshold: int = 300):
    n, n_correct = summary["n_scored"], summary["n_correct_top1"]
    print(f"\nscored {n} facts")
    print(
        f"success rate (top-1): {summary['success_rate']:.3f} "
        f"({n_correct} known facts)"
    )
    verdict = "PASS" if n_correct >= known_threshold else "FAIL — move one model size up"
    print(f"decision rule (need >= {known_threshold} known facts): {verdict}")

    if "by_relation" in summary:
        rels = sorted(
            summary["by_relation"].items(), key=lambda kv: -kv[1]["correct"]
        )
        print("\ntop relations by known count:")
        for rel, s in rels[:10]:
            print(f"  {rel}: {s['correct']}/{s['n']} correct")

    print("\nsample predictions:")
    for r in results[:5]:
        mark = "OK " if r["correct"] else "X  "
        print(
            f"  {mark} {r['prompt']!r} -> predicted {r['top1_token']!r}, "
            f"target {r['target']!r} (rank {r['target_rank']}, p={r['target_prob']:.3f})"
        )
