"""Lens-free causal check: cross-model activation patching (CMAP, Prakash et al. ICLR
2024), on the full residual stream rather than their circuit heads, so first_flip_layer
is a sufficiency threshold, not a component localization."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .conditions import _stratified_sample
from .lenses import get_decoder_parts
from .training import list_checkpoints
from .utils import batched, free_model, gather_last, load_model


@torch.no_grad()
def _patched_top1(base_model, enc, replacement: torch.Tensor, layer_module) -> torch.Tensor:
    """Run base_model with layer_module's output hidden states replaced; return top-1 ids."""

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (replacement,) + output[1:]
        return replacement

    handle = layer_module.register_forward_hook(hook)
    try:
        logits = base_model(**enc).logits
    finally:
        handle.remove()
    last = gather_last(logits, enc["attention_mask"])
    return last.float().argmax(dim=-1)


def _length_matched_donor(attention_mask: torch.Tensor) -> list[int | None]:
    """Map each row to a different row of identical real-token length, or None; a
    donor's residual stream is only positionally comparable at matched lengths."""
    lengths = attention_mask.sum(dim=1).tolist()
    by_len: dict[int, list[int]] = {}
    for i, length in enumerate(lengths):
        by_len.setdefault(int(length), []).append(i)
    donor: list[int | None] = [None] * len(lengths)
    for idxs in by_len.values():
        if len(idxs) < 2:
            continue
        for src, dst in zip(idxs, idxs[1:] + idxs[:1]):   # rotation, so never self
            donor[src] = dst
    return donor


def _flip_matrix(base_model, enc, ans, hidden, base_layers, n_layers, base_top1):
    """Per-layer boolean 'base output equals the target answer' after patching."""
    flips = torch.zeros((ans.shape[0], n_layers + 1), dtype=torch.bool)
    flips[:, 0] = (base_top1 == ans).cpu()
    for layer in range(1, n_layers + 1):
        top1 = _patched_top1(base_model, enc, hidden[layer], base_layers[layer - 1])
        flips[:, layer] = (top1 == ans).cpu()
    return flips


def _rows_from_flips(flips, sub, variant, step, extra=None):
    rows = []
    for j in range(len(sub)):
        r = sub.iloc[j]
        hit = torch.nonzero(flips[j]).flatten().tolist()
        base = {"variant": variant, "step": step, "fact_id": r["fact_id"],
                "condition": r["condition"], **(extra or {})}
        for layer in range(flips.shape[1]):
            rows.append({**base, "layer": layer, "flipped": bool(flips[j, layer])})
        rows.append({**base, "layer": -1, "flipped": bool(hit),
                     "first_flip_layer": hit[0] if hit else None})
    return rows


def _patch_one_checkpoint(base_model, lora_model, base_layers, n_layers,
                           sample, tokenizer, cfg, device, variant, step):
    """Run patching for a single LoRA checkpoint; return list of row dicts."""
    rows = []
    idx = list(range(len(sample)))
    for chunk in tqdm(list(batched(idx, cfg.patching.batch_size)),
                      desc=f"patching[{variant}]"):
        sub = sample.iloc[chunk]
        enc = tokenizer(sub["prompt"].tolist(), return_tensors="pt", padding=True).to(device)
        ans = torch.tensor(sub["answer_token_id"].tolist(), device=device)

        lora_hidden = lora_model(**enc, output_hidden_states=True).hidden_states
        base_top1 = gather_last(base_model(**enc).logits, enc["attention_mask"]) \
            .float().argmax(dim=-1)

        # lora_hidden[l] is the output of block l-1; l=0 is the unpatched base model.
        flips = _flip_matrix(base_model, enc, ans, lora_hidden, base_layers, n_layers,
                             base_top1)
        rows.extend(_rows_from_flips(flips, sub, variant, step))
    return rows


def _run_controls(base_model, lora_model, base_layers, n_layers, sample, tokenizer,
                  cfg, device, variant, step):
    """self: base patched into itself, must reproduce the unpatched output; mismatched:
    a different fact's LoRA activations, must rarely flip or LoRA isn't fact-specific."""
    rows = []
    idx = list(range(len(sample)))
    for chunk in tqdm(list(batched(idx, cfg.patching.batch_size)),
                      desc=f"controls[{variant}]"):
        sub = sample.iloc[chunk]
        enc = tokenizer(sub["prompt"].tolist(), return_tensors="pt", padding=True).to(device)
        ans = torch.tensor(sub["answer_token_id"].tolist(), device=device)

        base_out = base_model(**enc, output_hidden_states=True)
        base_hidden = base_out.hidden_states
        base_top1 = gather_last(base_out.logits, enc["attention_mask"]).float().argmax(dim=-1)

        flips = _flip_matrix(base_model, enc, ans, base_hidden, base_layers, n_layers,
                             base_top1)
        rows.extend(_rows_from_flips(flips, sub, variant, step, {"control": "self"}))

        donor = _length_matched_donor(enc["attention_mask"])
        keep = [i for i, d in enumerate(donor) if d is not None]
        if not keep:
            continue
        order = torch.tensor([donor[i] for i in keep], device=device)
        rows_keep = torch.tensor(keep, device=device)
        lora_hidden = lora_model(**enc, output_hidden_states=True).hidden_states
        mixed = [h.index_copy(0, rows_keep, h.index_select(0, order)) for h in lora_hidden]
        flips = _flip_matrix(base_model, enc, ans, mixed, base_layers, n_layers, base_top1)
        rows.extend(_rows_from_flips(flips.index_select(0, torch.tensor(keep)),
                                     sub.iloc[keep], variant, step,
                                     {"control": "mismatched"}))
    return rows


def run_patching(cfg, tokenizer, conditions: pd.DataFrame, device) -> None:
    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list_checkpoints(cfg)
    if not checkpoints:
        raise SystemExit("[patch] No trained adapters found — run train_lora first.")

    cap = cfg.patching.max_facts_per_condition
    # "known" is excluded: the base model already predicts correctly for these
    # facts, so any LoRA patch trivially preserves the correct answer — the
    # result reflects the selection criterion, not LoRA's causal footprint.
    patchable = conditions[conditions["condition"] != "known"]
    # Seeded relation-stratified sample; .head(cap) took whatever order the file held.
    rng = random.Random(cfg.seed)
    sample = pd.concat(
        [_stratified_sample(g, cap, cfg.data.max_relation_fraction, rng).assign(condition=c)
         for c, g in patchable.groupby("condition")],
        ignore_index=True)

    log_path = Path(cfg.output_dir) / "lora" / "training_log.csv"
    final_step = int(pd.read_csv(log_path)["step"].max()) if log_path.exists() else -1

    base_model = load_model(cfg, device=device)
    base_layers, _, _ = get_decoder_parts(base_model)
    n_layers = len(base_layers)

    all_rows, control_rows = [], []
    for label, adapter_path in checkpoints:
        step = final_step if label == "final" else int(label.split("_")[1])
        lora_model = load_model(cfg, device=device, adapter_path=adapter_path)
        all_rows.extend(_patch_one_checkpoint(
            base_model, lora_model, base_layers, n_layers,
            sample, tokenizer, cfg, device, label, step))
        # Controls are a validity check, so the final checkpoint is enough.
        if label == "final" and cfg.patching.get("controls", True):
            control_rows.extend(_run_controls(
                base_model, lora_model, base_layers, n_layers,
                sample, tokenizer, cfg, device, label, step))
        free_model(lora_model)

    free_model(base_model)

    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir / "patching.csv", index=False)

    summary = (df[df["layer"] == -1]
               .groupby(["variant", "step", "condition"])
               .agg(n_facts=("flipped", "size"), n_flipped=("flipped", "sum"),
                    mean=("first_flip_layer", "mean"),
                    median=("first_flip_layer", "median"))
               .reset_index())
    print("\n[patch] Earliest layer at which patching flips base -> answer (per checkpoint).")
    print("        mean/median are conditional on flipping, so read them against n_flipped:")
    print(summary.to_string(index=False))

    _report_matched_dynamics(df)

    if control_rows:
        controls = pd.DataFrame(control_rows)
        controls.to_csv(results_dir / "patching_controls.csv", index=False)
        _report_controls(controls, df)


def _report_matched_dynamics(df: pd.DataFrame) -> None:
    """First-flip depth over training, restricted to facts that flip at EVERY
    checkpoint, since the full-sample population grows with training and could make
    a stable depth claim pure survivorship."""
    flips = df[df["layer"] == -1]
    n_ckpt = flips["variant"].nunique()
    always = (flips.groupby("fact_id")["flipped"].sum() == n_ckpt)
    keep = set(always[always].index)
    if not keep:
        print("\n[patch] No fact flips at every checkpoint; matched-subset dynamics skipped.")
        return
    matched = flips[flips["fact_id"].isin(keep)]
    table = (matched.groupby(["step", "condition"])["first_flip_layer"]
             .agg(["median", "mean", "count"]).reset_index())
    print(f"\n[patch] Matched subset: {len(keep)} facts that flip at every checkpoint.")
    print("        The full-sample view is confounded (the flipping population grows with")
    print("        training); this one is biased toward early-learned facts. Any claim about")
    print("        depth over training has to survive both.")
    print(table.to_string(index=False))


def _report_controls(controls: pd.DataFrame, main: pd.DataFrame) -> None:
    """Print the two specificity controls against the real flip rate."""
    ctl = controls[controls["layer"] == -1]
    real = main[(main["layer"] == -1) & (main["variant"] == "final")]
    rows = []
    for cond, g in real.groupby("condition"):
        rec = {"condition": cond, "lora_flip_rate": g["flipped"].mean(), "n": len(g)}
        for name in ("self", "mismatched"):
            sub = ctl[(ctl["control"] == name) & (ctl["condition"] == cond)]
            rec[f"{name}_flip_rate"] = sub["flipped"].mean() if len(sub) else float("nan")
            rec[f"{name}_n"] = len(sub)
        rows.append(rec)
    print("\n[patch] Specificity controls (final checkpoint):")
    print(pd.DataFrame(rows).to_string(index=False))
    print("        self should equal the base model's own accuracy (the hook is a no-op);")
    print("        mismatched should be near zero, else flips are not fact-specific.")
