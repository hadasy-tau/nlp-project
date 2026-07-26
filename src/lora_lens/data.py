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

    cols = set(ds.column_names)
    for key in ("prompt", "subject", "target_true", "target_false", "relation"):
        if f[key] not in cols:
            raise KeyError(
                f"Column {f[key]!r} (mapped from data.fields.{key}) not in dataset "
                f"{cfg.data.dataset}. Available: {sorted(cols)}"
            )
    para_col = f.get("paraphrases")
    if para_col and para_col not in cols:
        print(f"[data] WARNING: paraphrase column {para_col!r} not found — "
              "paraphrase holdout evaluation will be skipped.")
        para_col = None

    rows = []
    n_multi = 0
    for i, rec in enumerate(ds):
        prompt = rec[f["prompt"]].rstrip()  # answer carries the leading space, not the prompt
        answer = str(rec[f["target_true"]]).strip()
        answer_ids = encode_answer(tokenizer, answer)
        if len(answer_ids) != 1:
            n_multi += 1
            continue
        paras = list(rec[para_col]) if para_col else []
        rows.append({
            "fact_id": f"cf_{i}",
            "relation": str(rec[f["relation"]]),
            "subject": str(rec[f["subject"]]),
            "prompt": prompt,
            "paraphrases": paras,
            "answer": answer,
            "answer_token_id": answer_ids[0],
            "target_false": str(rec[f["target_false"]]).strip(),
        })

    df = pd.DataFrame(rows)
    print(f"[data] {len(df)} facts kept, {n_multi} dropped (multi-token answer) "
          f"out of {len(ds)} records.")
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
