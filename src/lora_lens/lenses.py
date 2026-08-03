"""Logit lens and tuned lens readouts over hidden states.

Layer index convention: layer k = residual stream after k transformer blocks
(k=0 is the embedding output, k=L is the final hidden state). The tuned lens has
translators for k in [0, L-1]; at k=L both lenses report the model's own output.

Methodological note (the LoRA-specific catch): the tuned lens is loaded ONCE from
the BASE model and reused for the fine-tuned model. Keeping the measuring
instrument fixed while the model changes is what licenses attributing differences
to LoRA. The tuned lens also systematically reports "first layer of appearance"
earlier than a strict reading would (it is trained to predict the final output) —
which is exactly why both lenses are reported.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def unwrap_base(model):
    """Unwrap a PeftModel to the underlying transformers model (no-op otherwise)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def get_decoder_parts(model):
    """(layers, final_norm, unembed_head) for GPTNeoX/Pythia and GPT-2 families."""
    base = unwrap_base(model)
    if hasattr(base, "gpt_neox"):
        return base.gpt_neox.layers, base.gpt_neox.final_layer_norm, base.embed_out
    if hasattr(base, "transformer"):
        return base.transformer.h, base.transformer.ln_f, base.get_output_embeddings()
    raise ValueError(f"Unsupported architecture: {type(base).__name__}")


def load_tuned_lens(cfg, base_model, device):
    """Load a pretrained tuned lens for the base model; returns None if unavailable."""
    if not cfg.lens.use_tuned:
        return None
    try:
        from tuned_lens.nn.lenses import TunedLens

        lens_id = cfg.lens.get("tuned_lens_id") or cfg.model.name
        lens = TunedLens.from_model_and_pretrained(
            unwrap_base(base_model), lens_resource_id=lens_id
        )
        # Pinned to fp32 so the probe is independent of the per-variant model dtype.
        lens = lens.to(device=device, dtype=torch.float32)
        lens.eval()
        print(f"[lens] tuned lens loaded for {lens_id}")
        return lens
    except Exception as e:  # missing lens on the hub, version mismatch, ...
        print(f"[lens] WARNING: tuned lens unavailable ({e!r}); continuing with logit lens only.")
        return None


@torch.no_grad()
def layerwise_answer_metrics(model, enc, answer_ids: torch.Tensor, tuned_lens,
                             use_logit: bool, top_k: list[int]) -> list[dict]:
    """Per-layer answer metrics at the last real token for one right-padded batch.

    Returns long-format dicts: {row, lens, layer, answer_logprob, answer_rank,
    in_top_<k>...}. `row` indexes into the batch.
    """
    from .utils import gather_last

    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states  # tuple of L+1 tensors [B, T, D]
    n_layers = len(hidden) - 1
    mask = enc["attention_mask"]
    final_logits = gather_last(out.logits, mask)

    _, final_norm, head = get_decoder_parts(model)

    records = []

    def add(lens_name: str, layer: int, logits: torch.Tensor):
        logprobs = F.log_softmax(logits.float(), dim=-1)  # fp32 always (pitfall 6)
        ans_lp = logprobs.gather(1, answer_ids[:, None]).squeeze(1)
        rank = (logprobs > ans_lp[:, None]).sum(dim=1) + 1
        for row in range(logits.shape[0]):
            rec = {
                "row": row, "lens": lens_name, "layer": layer,
                "answer_logprob": ans_lp[row].item(),
                "answer_rank": int(rank[row].item()),
            }
            for k in top_k:
                rec[f"in_top_{k}"] = rec["answer_rank"] <= k
            records.append(rec)

    for layer in range(n_layers + 1):
        h = gather_last(hidden[layer], mask)
        if use_logit:
            if layer == n_layers:
                add("logit", layer, final_logits)
            else:
                add("logit", layer, head(final_norm(h)))
        if tuned_lens is not None:
            if layer == n_layers:
                add("tuned", layer, final_logits)
            else:
                add("tuned", layer, tuned_lens(h.float(), idx=layer))
    return records
