"""Step 2: logit lens + tuned lens smoke test.

Both lenses read the same hidden states (one forward pass with
output_hidden_states=True, gathered at the last real token). Layer l for
l in [0, L-1] is decoded through the lens (layer 0 = embeddings, layer l =
output of block l); the "final" row uses the model's own logits, so the
already-normalized last hidden state is never re-normalized.

- Logit lens: softmax(W_U · LN_f(h_l)) — implemented manually so it works
  for both GPT-NeoX (Pythia) and GPT-2 checkpoints.
- Tuned lens: pretrained translators fetched from the
  AlignmentResearch/tuned-lens HF space via the tuned-lens package.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import FactItem


def get_final_norm_and_unembed(model):
    unembed = model.get_output_embeddings()
    base = model.base_model
    for attr in ("final_layer_norm", "ln_f", "norm"):
        if hasattr(base, attr):
            return getattr(base, attr), unembed
    raise ValueError(
        f"cannot find final layernorm on {type(base).__name__}; "
        "add its attribute name to get_final_norm_and_unembed()"
    )


def pick_lens_sample(results: list[dict], items: list[FactItem], sample_size: int, seed: int) -> list[FactItem]:
    """Half items the model got right, half it got wrong — both trajectories
    are informative. Falls back gracefully if either pool is small."""
    by_idx = {it.idx: it for it in items}
    correct = [r["idx"] for r in results if r["correct"]]
    wrong = [r["idx"] for r in results if not r["correct"]]
    rng = random.Random(seed)
    rng.shuffle(correct)
    rng.shuffle(wrong)

    half = sample_size // 2
    chosen = correct[:half] + wrong[: sample_size - min(half, len(correct))]
    chosen = chosen[:sample_size]
    return [by_idx[i] for i in chosen]


@torch.no_grad()
def collect_hidden_states(model, tokenizer, items: list[FactItem], batch_size: int, device):
    """One forward pass; returns hidden states at the answer position for
    every layer [L+1, N, d] and the model's final-position logits [N, V]."""
    per_layer: list[list[torch.Tensor]] = []
    final_logits: list[torch.Tensor] = []

    for start in tqdm(range(0, len(items), batch_size), desc="hidden states"):
        batch = items[start : start + batch_size]
        enc = tokenizer(
            [it.prompt for it in batch], return_tensors="pt", padding=True
        ).to(device)
        out = model(**enc, output_hidden_states=True)

        last_idx = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=device)

        gathered = [h[rows, last_idx] for h in out.hidden_states]  # (L+1) x [B, d]
        if not per_layer:
            per_layer = [[] for _ in gathered]
        for l, h in enumerate(gathered):
            per_layer[l].append(h)
        final_logits.append(out.logits[rows, last_idx])

    hidden = torch.stack([torch.cat(chunks) for chunks in per_layer])  # [L+1, N, d]
    return hidden, torch.cat(final_logits)


@torch.no_grad()
def decode_trajectories(
    hidden: torch.Tensor,
    final_logits: torch.Tensor,
    decode_fn,
    items: list[FactItem],
    tokenizer,
    lens_name: str,
    top_k: int,
) -> dict:
    """decode_fn(h [N, d], layer_idx) -> logits [N, V] for layers 0..L-1;
    the final row comes from the model's own output."""
    n_lens_layers = hidden.shape[0] - 1
    target_ids = torch.tensor(
        [it.target_token_id for it in items], device=hidden.device
    )
    rows = torch.arange(len(items), device=hidden.device)

    layer_logprob, layer_rank, layer_top1 = [], [], []
    for l in range(n_lens_layers + 1):
        logits = decode_fn(hidden[l], l) if l < n_lens_layers else final_logits
        logprobs = F.log_softmax(logits.float(), dim=-1)
        tgt_lp = logprobs[rows, target_ids]
        layer_logprob.append(tgt_lp)
        layer_rank.append((logprobs > tgt_lp.unsqueeze(1)).sum(dim=1) + 1)
        layer_top1.append(logprobs.argmax(dim=-1))

    logprob = torch.stack(layer_logprob)  # [L+1, N]
    rank = torch.stack(layer_rank)
    top1 = torch.stack(layer_top1)

    per_item = []
    for i, it in enumerate(items):
        is_top1 = (top1[:, i] == target_ids[i]).tolist()
        first_layer = is_top1.index(True) if True in is_top1 else None
        in_topk = (rank[:, i] <= top_k).tolist()
        first_topk = in_topk.index(True) if True in in_topk else None
        per_item.append(
            {
                "idx": it.idx,
                "target": it.target,
                "first_top1_layer": first_layer,
                f"first_top{top_k}_layer": first_topk,
                "final_target_prob": logprob[-1, i].exp().item(),
                "final_correct": bool(top1[-1, i] == target_ids[i]),
                "target_prob_by_layer": logprob[:, i].exp().tolist(),
                "target_rank_by_layer": rank[:, i].tolist(),
            }
        )

    appeared = [p["first_top1_layer"] for p in per_item if p["first_top1_layer"] is not None]
    summary = {
        "lens": lens_name,
        "n_items": len(items),
        "n_target_top1_somewhere": len(appeared),
        "mean_first_top1_layer": sum(appeared) / len(appeared) if appeared else None,
        "mean_target_prob_by_layer": logprob.exp().mean(dim=1).tolist(),
    }

    _print_lens_report(summary, per_item, top1, items, tokenizer, n_lens_layers)
    return {"summary": summary, "per_item": per_item}


def _print_lens_report(summary, per_item, top1, items, tokenizer, n_layers):
    print(
        f"\n[{summary['lens']}] target becomes top-1 somewhere in the stack for "
        f"{summary['n_target_top1_somewhere']}/{summary['n_items']} items"
    )
    if summary["mean_first_top1_layer"] is not None:
        print(
            f"[{summary['lens']}] mean first top-1 layer: "
            f"{summary['mean_first_top1_layer']:.1f} (of {n_layers} layers + final)"
        )
    probe_layers = sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers})
    print(f"[{summary['lens']}] example trajectories (top-1 token at layers {probe_layers}):")
    for i, it in enumerate(items[:3]):
        toks = [repr(tokenizer.decode([top1[l, i].item()])) for l in probe_layers]
        print(
            f"  {it.prompt!r} (target {it.target!r}, first top-1 layer "
            f"{per_item[i]['first_top1_layer']}): {' -> '.join(toks)}"
        )


def run_logit_lens(model, hidden, final_logits, items, tokenizer, top_k) -> dict:
    norm, unembed = get_final_norm_and_unembed(model)

    def decode(h, _layer):
        return unembed(norm(h))

    return decode_trajectories(
        hidden, final_logits, decode, items, tokenizer, "logit lens", top_k
    )


def run_tuned_lens(model, hidden, final_logits, items, tokenizer, top_k) -> dict:
    from tuned_lens.nn.lenses import TunedLens

    print("fetching pretrained tuned lens (AlignmentResearch/tuned-lens space)...")
    # map_location: pretrained lens checkpoints store CUDA tensors, which
    # otherwise fail to load on CPU-only machines.
    lens = TunedLens.from_model_and_pretrained(model, map_location=hidden.device)
    lens = lens.to(hidden.device)
    lens_dtype = next(lens.parameters()).dtype

    def decode(h, layer):
        return lens(h.to(lens_dtype), layer)

    return decode_trajectories(
        hidden, final_logits, decode, items, tokenizer, "tuned lens", top_k
    )
