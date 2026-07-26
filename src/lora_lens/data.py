"""Dataset preparation: load facts, filter to single-token answers, build synthetic facts.

The source dataset and its column names are fully configurable (cfg.data.dataset,
cfg.data.fields), so CounterFact can be swapped for LAMA/T-REx, ZsRE, etc. as long
as the mapped columns exist.
"""

from __future__ import annotations

import random

import pandas as pd
from datasets import load_dataset

from .utils import encode_answer

# Syllables for pseudo-entity names ("Zorbath Kell" style).
_SYLLABLES = [
    "zor", "bath", "kell", "vran", "mira", "thel", "dros", "quin", "fal", "nex",
    "gor", "lith", "sarn", "vek", "olm", "tris", "hax", "brun", "yel", "dask",
]


def _pseudo_name(rng: random.Random) -> str:
    def word():
        return "".join(rng.sample(_SYLLABLES, rng.choice([2, 3]))).capitalize()

    return f"{word()} {word()}"


_MISSING = object()


def _get_path(rec, path: str):
    """Look up a dotted path, so nested datasets work without code changes.

    'requested_rewrite.target_true.str' reaches into the full CounterFact schema;
    a plain 'target_true' still works for flattened datasets.
    """
    cur = rec
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _fix_mojibake(s: str) -> str:
    """Repair double-encoded UTF-8 ('FranÃ§ois' -> 'François').

    Some CounterFact mirrors were uploaded with UTF-8 bytes decoded as latin-1.
    Left as-is when the round-trip is not possible.
    """
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _fill_subject(prompt: str, subject: str) -> str:
    """CounterFact prompts are templates ('The mother tongue of {} is')."""
    if "{subject}" in prompt:
        return prompt.replace("{subject}", subject)
    if "{}" in prompt:
        return prompt.replace("{}", subject)
    return prompt


def prepare_facts(cfg, tokenizer) -> pd.DataFrame:
    """Load the fact dataset, normalize columns, and keep single-token answers only.

    Multi-token answers are dropped because "rank of the correct answer" is not
    well-defined at a single readout position (pitfall 2). Expect to lose a large
    fraction of CounterFact.
    """
    f = cfg.data.fields
    ds = load_dataset(cfg.data.dataset, split=cfg.data.split)
    if cfg.data.get("max_records"):
        ds = ds.select(range(min(cfg.data.max_records, len(ds))))

    probe = ds[0]
    for key in ("prompt", "subject", "target_true", "relation"):
        if _get_path(probe, f[key]) is _MISSING:
            raise KeyError(
                f"Field {f[key]!r} (mapped from data.fields.{key}) not in dataset "
                f"{cfg.data.dataset}. Top-level columns: {sorted(ds.column_names)}. "
                "Nested fields are addressable with dots, e.g. "
                "'requested_rewrite.target_true.str'."
            )

    para_path = f.get("paraphrases")
    if para_path and _get_path(probe, para_path) is _MISSING:
        print(f"[data] WARNING: paraphrase field {para_path!r} not found in "
              f"{cfg.data.dataset} — the held-out paraphrase probe (pitfall 3) will be "
              "SKIPPED, so a late-layer shift cannot be distinguished from prompt "
              "memorization. Use a dataset that ships paraphrases (e.g. azhx/counterfact).")
        para_path = None

    false_path = f.get("target_false")
    has_false = bool(false_path) and _get_path(probe, false_path) is not _MISSING

    rows = []
    n_multi = 0
    for i, rec in enumerate(ds):
        subject = _fix_mojibake(str(_get_path(rec, f["subject"])).strip())
        # The answer carries the leading space, so the prompt must not (pitfall 1).
        prompt = _fill_subject(str(_get_path(rec, f["prompt"])), subject).rstrip()
        answer = str(_get_path(rec, f["target_true"])).strip()
        answer_ids = encode_answer(tokenizer, answer)
        if len(answer_ids) != 1:
            n_multi += 1
            continue
        paras = [_fix_mojibake(str(p).rstrip()) for p in (_get_path(rec, para_path) or [])] \
            if para_path else []
        rows.append({
            "fact_id": f"cf_{i}",
            "relation": str(_get_path(rec, f["relation"])),
            "subject": subject,
            "prompt": prompt,
            "paraphrases": paras,
            "answer": answer,
            "answer_token_id": answer_ids[0],
            "target_false": str(_get_path(rec, false_path)).strip() if has_false else "",
        })

    df = pd.DataFrame(rows)
    n_para = int(df["paraphrases"].str.len().gt(0).sum()) if len(df) else 0
    print(f"[data] {len(df)} facts kept, {n_multi} dropped (multi-token answer) "
          f"out of {len(ds)} records; {n_para} have held-out paraphrases.")
    return df


def make_synthetic(real_df: pd.DataFrame, tokenizer, cfg) -> pd.DataFrame:
    """Synthetic facts: same relation templates as the real data, pseudo-entity subjects.

    Using the *same* templates matters — otherwise layer-dynamics differences could
    just be template differences. The object is sampled from the real objects seen
    for that relation (so a "capital of" template still gets a city), but attached
    to a pseudo-entity subject, making the triple new by construction.
    """
    rng = random.Random(cfg.data.synthetic.seed)
    n = cfg.data.synthetic.n_facts

    # One (template, object pool) per relation. Template = prompt with the subject
    # replaced by a placeholder; skip prompts where the subject string is absent.
    by_rel: dict[str, dict] = {}
    for _, r in real_df.iterrows():
        if r["subject"] and r["subject"] in r["prompt"]:
            entry = by_rel.setdefault(r["relation"], {"templates": [], "objects": []})
            entry["templates"].append(r["prompt"].replace(r["subject"], "{subject}"))
            entry["objects"].append((r["answer"], r["answer_token_id"]))

    relations = sorted(by_rel)
    if not relations:
        raise RuntimeError("No usable relation templates found for synthetic facts.")

    rows, seen_names = [], set()
    for i in range(n):
        rel = relations[i % len(relations)]
        entry = by_rel[rel]
        template = rng.choice(entry["templates"])
        answer, answer_tok = rng.choice(entry["objects"])
        while True:
            name = _pseudo_name(rng)
            if name not in seen_names:
                seen_names.add(name)
                break
        rows.append({
            "fact_id": f"syn_{i}",
            "relation": rel,
            "subject": name,
            "prompt": template.format(subject=name),
            "paraphrases": [],
            "answer": answer,
            "answer_token_id": answer_tok,
            "target_false": "",
        })

    df = pd.DataFrame(rows)
    print(f"[data] {len(df)} synthetic facts over {len(relations)} relations.")
    return df
