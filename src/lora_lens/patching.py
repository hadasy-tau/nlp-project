"""Lens-free causal check: cross-model activation patching (CMAP, Prakash et al. ICLR
2024) on the full residual stream, so first_flip_layer is a sufficiency threshold, not
a component localization; logit_diff is the graded alternative Zhang & Nanda (ICLR
2024) recommend over a saturating binary flip."""

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
def _patched_logits(base_model, enc, replacement: torch.Tensor, layer_module) -> torch.Tensor:
    """Run base_model with layer_module's output hidden states replaced; return the
    last-position logits [B, V]."""

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (replacement,) + output[1:]
        return replacement

    handle = layer_module.register_forward_hook(hook)
    try:
        logits = base_model(**enc).logits
    finally:
        handle.remove()
    return gather_last(logits, enc["attention_mask"]).float()


def _logit_diff(logits: torch.Tensor, ans: torch.Tensor,
                false_id: torch.Tensor) -> torch.Tensor:
    """Answer logit minus a competing logit: the fixed counterfactual token where
    false_id >= 0, else the best non-answer logit at this layer."""
    ans_logit = logits.gather(1, ans[:, None]).squeeze(1)
    has_false = false_id >= 0
    false_logit = logits.gather(1, false_id.clamp(min=0)[:, None]).squeeze(1)
    masked = logits.scatter(1, ans[:, None], float("-inf"))
    best_other = masked.max(dim=1).values
    return ans_logit - torch.where(has_false, false_logit, best_other)


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


def _flip_matrix(base_model, enc, ans, false_id, hidden, base_layers, n_layers, base_logits):
    """Per-layer flip (argmax == answer) and logit_diff after patching each layer."""
    flips = torch.zeros((ans.shape[0], n_layers + 1), dtype=torch.bool)
    diffs = torch.zeros((ans.shape[0], n_layers + 1))
    flips[:, 0] = (base_logits.argmax(dim=-1) == ans).cpu()
    diffs[:, 0] = _logit_diff(base_logits, ans, false_id).cpu()
    for layer in range(1, n_layers + 1):
        logits = _patched_logits(base_model, enc, hidden[layer], base_layers[layer - 1])
        flips[:, layer] = (logits.argmax(dim=-1) == ans).cpu()
        diffs[:, layer] = _logit_diff(logits, ans, false_id).cpu()
    return flips, diffs


def _rows_from_flips(flips, diffs, sub, variant, step, extra=None):
    rows = []
    for j in range(len(sub)):
        r = sub.iloc[j]
        hit = torch.nonzero(flips[j]).flatten().tolist()
        pos = torch.nonzero(diffs[j] > 0).flatten().tolist()
        base = {"variant": variant, "step": step, "fact_id": r["fact_id"],
                "condition": r["condition"], **(extra or {})}
        for layer in range(flips.shape[1]):
            rows.append({**base, "layer": layer, "flipped": bool(flips[j, layer]),
                         "logit_diff": float(diffs[j, layer])})
        rows.append({**base, "layer": -1, "flipped": bool(hit),
                     "first_flip_layer": hit[0] if hit else None,
                     "first_positive_diff_layer": pos[0] if pos else None})
    return rows


def _false_ids(sub: pd.DataFrame, device) -> torch.Tensor:
    """target_false_token_id as a long tensor, -1 where missing (NaN)."""
    vals = sub["target_false_token_id"].fillna(-1).astype(int).to_numpy()
    return torch.tensor(vals, device=device, dtype=torch.long)


@torch.no_grad()
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
        false_id = _false_ids(sub, device)

        lora_hidden = lora_model(**enc, output_hidden_states=True).hidden_states
        base_logits = gather_last(base_model(**enc).logits, enc["attention_mask"]).float()

        # lora_hidden[l] is the output of block l-1; l=0 is the unpatched base model.
        flips, diffs = _flip_matrix(base_model, enc, ans, false_id, lora_hidden,
                                    base_layers, n_layers, base_logits)
        rows.extend(_rows_from_flips(flips, diffs, sub, variant, step))
    return rows


@torch.no_grad()
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
        false_id = _false_ids(sub, device)

        base_out = base_model(**enc, output_hidden_states=True)
        base_hidden = base_out.hidden_states
        base_logits = gather_last(base_out.logits, enc["attention_mask"]).float()

        flips, diffs = _flip_matrix(base_model, enc, ans, false_id, base_hidden,
                                    base_layers, n_layers, base_logits)
        rows.extend(_rows_from_flips(flips, diffs, sub, variant, step, {"control": "self"}))

        donor = _length_matched_donor(enc["attention_mask"])
        keep = [i for i, d in enumerate(donor) if d is not None]
        if not keep:
            continue
        # The target's own answer/false-token ids, not the donor's.
        order = torch.tensor([donor[i] for i in keep], device=device)
        rows_keep = torch.tensor(keep, device=device)
        lora_hidden = lora_model(**enc, output_hidden_states=True).hidden_states
        mixed = [h.index_copy(0, rows_keep, h.index_select(0, order)) for h in lora_hidden]
        flips, diffs = _flip_matrix(base_model, enc, ans, false_id, mixed, base_layers,
                                    n_layers, base_logits)
        keep_t = torch.tensor(keep)
        rows.extend(_rows_from_flips(flips.index_select(0, keep_t), diffs.index_select(0, keep_t),
                                     sub.iloc[keep], variant, step, {"control": "mismatched"}))
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
    n_false = int((sample["target_false_token_id"].notna()).sum())
    print(f"\n[patch] {n_false}/{len(sample)} facts have a single-token counterfactual "
          "(target_false); the rest use the per-layer best-other-token contrast.")
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

    _report_near_misses(df)

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
    layers = df[(df["layer"] >= 1) & df["fact_id"].isin(keep)]
    diff_table = (layers.groupby(["step", "condition"])["logit_diff"].mean()
                  .rename("mean_logit_diff").reset_index())
    table = table.merge(diff_table, on=["step", "condition"], how="left")
    print(f"\n[patch] Matched subset: {len(keep)} facts that flip at every checkpoint.")
    print("        The full-sample view is confounded (the flipping population grows with")
    print("        training); this one is biased toward early-learned facts. Any claim about")
    print("        depth over training has to survive both.")
    print(table.to_string(index=False))


def _report_near_misses(df: pd.DataFrame, margin: float = 1.0) -> None:
    """Count layers that didn't flip but came within `margin` logits of flipping --
    exactly what a binary flip/no-flip metric cannot show."""
    rows = df[(df["layer"] >= 0) & ~df["flipped"]]
    near = rows[(rows["logit_diff"] > -margin) & (rows["logit_diff"] <= 0)]
    print(f"\n[patch] Near-misses: {len(near)}/{len(rows)} non-flipped (variant, fact, layer) "
          f"rows are within {margin:g} logit of flipping (logit_diff in (-{margin:g}, 0]) -- "
          "invisible to a binary flip metric.")


def _report_controls(controls: pd.DataFrame, main: pd.DataFrame) -> None:
    """Flip rate and mean logit_diff (from the per-layer rows, not layer==-1) for both
    controls against the real run."""
    ctl_final, ctl_layers = controls[controls["layer"] == -1], controls[controls["layer"] >= 1]
    real_final = main[(main["layer"] == -1) & (main["variant"] == "final")]
    real_layers = main[(main["layer"] >= 1) & (main["variant"] == "final")]
    rows = []
    for cond, g in real_final.groupby("condition"):
        rec = {"condition": cond, "lora_flip_rate": g["flipped"].mean(),
               "lora_mean_diff": real_layers.loc[real_layers["condition"] == cond,
                                                 "logit_diff"].mean(),
               "n": len(g)}
        for name in ("self", "mismatched"):
            f = ctl_final[(ctl_final["control"] == name) & (ctl_final["condition"] == cond)]
            l = ctl_layers[(ctl_layers["control"] == name) & (ctl_layers["condition"] == cond)]
            rec[f"{name}_flip_rate"] = f["flipped"].mean() if len(f) else float("nan")
            rec[f"{name}_mean_diff"] = l["logit_diff"].mean() if len(l) else float("nan")
            rec[f"{name}_n"] = len(f)
        rows.append(rec)
    print("\n[patch] Specificity controls (final checkpoint):")
    print(pd.DataFrame(rows).to_string(index=False))
    print("        self should equal the base model's own accuracy (the hook is a no-op);")
    print("        mismatched flip rate and mean logit_diff should both be near zero/very")
    print("        negative, else flips (or a near-miss margin) are not fact-specific.")
