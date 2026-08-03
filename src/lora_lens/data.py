"""Dataset preparation: load facts, filter to single-token answers, build synthetic facts.

The source dataset and its column names are fully configurable (cfg.data.dataset,
cfg.data.fields), so CounterFact can be swapped for LAMA/T-REx, ZsRE, etc. as long
as the mapped columns exist.
"""

from __future__ import annotations

import ast
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from .utils import encode_answer

# Syllables for pseudo-entity names ("Zorbath Kell" style).
_SYLLABLES = [
    "zor", "bath", "kell", "vran", "mira", "thel", "dros", "quin", "fal", "nex",
    "gor", "lith", "sarn", "vek", "olm", "tris", "hax", "brun", "yel", "dask",
]


class _Cycler:
    """Yields items in shuffled order, only reshuffling once every item has been
    used once — so a relation's templates/paraphrases don't repeat until the
    full set has appeared, instead of i.i.d. sampling with replacement.
    """

    def __init__(self, items: list, rng: random.Random):
        self._items = list(items)
        self._rng = rng
        self._pool: list = []

    def next(self):
        if not self._items:
            return None
        if not self._pool:
            self._pool = list(self._items)
            self._rng.shuffle(self._pool)
        return self._pool.pop()


def _dedup(seq) -> list:
    return list(dict.fromkeys(seq))


def _pseudo_name(rng: random.Random, tokenizer, token_range: tuple[int, int],
                  syllable_counts: dict[str, int], total_slots: int, tolerance: float,
                  seen_names: set[str], max_attempts: int = 50) -> tuple[str, int, int]:
    """Generate a pseudo-entity name whose BPE token length matches the real subjects'
    typical range (tokenization confound) and whose syllables stay roughly uniform in
    usage across all generated names so far (lexical diversity).

    Returns (name, n_tokens, new_total_syllable_slots). Falls back to the least-bad
    candidate seen if no candidate satisfies both constraints within max_attempts,
    rather than looping forever.
    """
    lo, hi = token_range
    max_rate = tolerance / len(_SYLLABLES)
    best = None
    for _ in range(max_attempts):
        syl1 = rng.sample(_SYLLABLES, rng.choice([2, 3]))
        syl2 = rng.sample(_SYLLABLES, rng.choice([2, 3]))
        name = f"{''.join(syl1).capitalize()} {''.join(syl2).capitalize()}"
        if name in seen_names:
            continue
        used = syl1 + syl2
        n_tokens = len(tokenizer(name, add_special_tokens=False)["input_ids"])
        token_ok = lo <= n_tokens <= hi

        new_total = total_slots + len(used)
        used_counts = Counter(used)
        worst_rate = max((syllable_counts.get(s, 0) + c) / new_total
                          for s, c in used_counts.items())
        syllable_ok = worst_rate <= max_rate

        if token_ok and syllable_ok:
            for s, c in used_counts.items():
                syllable_counts[s] = syllable_counts.get(s, 0) + c
            return name, n_tokens, new_total

        badness = (0 if token_ok else min(abs(n_tokens - lo), abs(n_tokens - hi))) \
            + max(0.0, worst_rate - max_rate)
        if best is None or badness < best[0]:
            best = (badness, name, n_tokens, used_counts, new_total)

    _, name, n_tokens, used_counts, new_total = best
    for s, c in used_counts.items():
        syllable_counts[s] = syllable_counts.get(s, 0) + c
    return name, n_tokens, new_total


def _pick_object(rng: random.Random, objects: list[tuple[str, int]], template: str,
                  rel_counts: Counter, global_counts: Counter, pair_counts: Counter,
                  rel_cap: int, global_cap: int, pair_cap: int,
                  max_attempts: int = 50) -> tuple[str, int]:
    """Sample an (answer, answer_token_id) pair while capping per-relation reuse,
    cross-relation global reuse, and (template, object) pair reuse — so a handful
    of objects/pairs can't dominate a relation or the synthetic set overall.

    Falls back to the least-over-cap candidate seen if nothing satisfies all three
    caps within max_attempts (e.g. a relation with very few distinct real objects).
    """
    best = None
    for _ in range(max_attempts):
        answer, answer_tok = rng.choice(objects)
        r_over = max(0, rel_counts.get(answer, 0) - rel_cap + 1)
        g_over = max(0, global_counts.get(answer, 0) - global_cap + 1)
        p_over = max(0, pair_counts.get((template, answer), 0) - pair_cap + 1)
        if r_over == 0 and g_over == 0 and p_over == 0:
            return answer, answer_tok
        badness = r_over + g_over + p_over
        if best is None or badness < best[0]:
            best = (badness, answer, answer_tok)
    return best[1], best[2]


def _build_relation_pools(real_df: pd.DataFrame) -> dict[str, dict]:
    """One (template pool, object pool) per relation.

    Template = prompt with the subject replaced by a placeholder; skip prompts
    where the subject string is absent. Paraphrases themselves are no longer
    pooled here — both real and synthetic facts get their paraphrases from an
    exact (relation, template) lookup into the curated CSV (see
    _load_paraphrase_map()), which preserves the per-fact template<->paraphrase
    link instead of flattening it across a relation.
    """
    by_rel: dict[str, dict] = {}
    for _, r in real_df.iterrows():
        if r["subject"] and r["subject"] in r["prompt"]:
            entry = by_rel.setdefault(r["relation"], {"templates": [], "objects": []})
            entry["templates"].append(r["prompt"].replace(r["subject"], "{subject}"))
            entry["objects"].append((r["answer"], r["answer_token_id"]))
    return by_rel


def _load_paraphrase_map(csv_path) -> dict[tuple[str, str], list[str]]:
    """Load the curated (relation_id, original_prompt) -> [paraphrase templates]
    CSV (additional_data/paraphreases_per_prompt.csv) into a lookup dict.

    Templates use the same '{}' placeholder convention as the source dataset's
    own prompt templates, so a fact's exact template (its filled prompt with
    the subject substituted back out, see prepare_facts()/make_synthetic())
    can be looked up directly — giving an exact per-template paraphrase set
    instead of a relation-wide pool.
    """
    df = pd.read_csv(csv_path)
    paraphrase_map: dict[tuple[str, str], list[str]] = {}
    for _, row in df.iterrows():
        key = (str(row["relation_id"]), str(row["original_prompt"]))
        paraphrase_map[key] = ast.literal_eval(row["list_of_paraphrases"])
    return paraphrase_map


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

    paraphrase_map = _load_paraphrase_map(cfg.data.paraphrase_templates_csv)

    neighbor_path = f.get("neighborhood")
    if neighbor_path and _get_path(probe, neighbor_path) is _MISSING:
        print(f"[data] WARNING: neighborhood field {neighbor_path!r} not found — "
              "locality scoring will be skipped.")
        neighbor_path = None

    false_path = f.get("target_false")
    has_false = bool(false_path) and _get_path(probe, false_path) is not _MISSING

    rows = []
    n_multi = 0
    n_no_paraphrase_match = 0
    for i, rec in enumerate(ds):
        subject = _fix_mojibake(str(_get_path(rec, f["subject"])).strip())
        # The answer carries the leading space, so the prompt must not (pitfall 1).
        prompt = _fill_subject(str(_get_path(rec, f["prompt"])), subject).rstrip()
        answer = str(_get_path(rec, f["target_true"])).strip()
        answer_ids = encode_answer(tokenizer, answer)
        if len(answer_ids) != 1:
            n_multi += 1
            continue
        relation = str(_get_path(rec, f["relation"]))
        template = prompt.replace(subject, "{}") if subject and subject in prompt else prompt
        paraphrase_templates = paraphrase_map.get((relation, template), [])
        if not paraphrase_templates:
            n_no_paraphrase_match += 1
        paras = [_fill_subject(t, subject) for t in paraphrase_templates]
        neighbors = [str(p).rstrip() for p in (_get_path(rec, neighbor_path) or [])] \
            if neighbor_path else []
        rows.append({
            "fact_id": f"cf_{i}",
            "relation": relation,
            "subject": subject,
            "prompt": prompt,
            "paraphrases": paras,
            "neighborhood_prompts": neighbors,
            "answer": answer,
            "answer_token_id": answer_ids[0],
            "target_false": str(_get_path(rec, false_path)).strip() if has_false else "",
        })

    df = pd.DataFrame(rows)
    n_para = int(df["paraphrases"].str.len().gt(0).sum()) if len(df) else 0
    print(f"[data] {len(df)} facts kept, {n_multi} dropped (multi-token answer) "
          f"out of {len(ds)} records; {n_para} have curated paraphrases "
          f"({n_no_paraphrase_match} facts had no matching (relation, template) "
          "in the paraphrase CSV).")
    return df


def make_synthetic(real_df: pd.DataFrame, tokenizer, cfg) -> pd.DataFrame:
    """Synthetic facts: same relation templates as the real data, pseudo-entity subjects.

    Using the *same* templates matters — otherwise layer-dynamics differences could
    just be template differences. The object is sampled from the real objects seen
    for that relation (so a "capital of" template still gets a city), but attached
    to a pseudo-entity subject, making the triple new by construction.

    Diversity controls (beyond exact-duplicate-name avoidance) guard against a
    handful of names/objects/templates dominating the synthetic set, which would
    otherwise make "synthetic" partly a template/vocabulary effect instead of a
    pure novel-entity effect:
      - pseudo-name BPE token length is matched to the real subjects' distribution
      - object reuse is capped per relation and globally (and per (template, object)
        pair), reusing the max_relation_fraction pattern from conditions.py
      - templates are cycled through before repeating; each fact's paraphrases
        follow directly from its own template via the curated per-template map
        (see _load_paraphrase_map()), mirroring prepare_facts()'s real-fact path
      - syllable usage across generated names is kept close to uniform
    A diversity validation report (<output_dir>/synthetic_diversity_report.csv) and
    a human-readable export (<output_dir>/synthetic_facts.csv) make all of this an
    auditable, checkable artifact instead of an assumption.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    syn_cfg = cfg.data.synthetic
    rng = random.Random(syn_cfg.seed)
    n = syn_cfg.n_facts

    paraphrase_map = _load_paraphrase_map(cfg.data.paraphrase_templates_csv)

    by_rel = _build_relation_pools(real_df)
    relations = sorted(by_rel)
    if not relations:
        raise RuntimeError("No usable relation templates found for synthetic facts.")

    for entry in by_rel.values():
        entry["templates"] = _dedup(entry["templates"])
        entry["template_cycler"] = _Cycler(entry["templates"], rng)

    # Real subject token-length distribution -> "typical" range for pseudo-entities.
    # Otherwise pseudo names could systematically tokenize shorter/longer than real
    # subjects, which would itself explain layer-depth differences independent of
    # "novelty" (the tokenization confound).
    low_pct = syn_cfg.get("token_length_low_pct", 5)
    high_pct = syn_cfg.get("token_length_high_pct", 95)
    real_tok_lens = [len(ids) for ids in
                      tokenizer(real_df["subject"].tolist(), add_special_tokens=False)["input_ids"]]
    token_lo = max(1, int(np.floor(np.percentile(real_tok_lens, low_pct))))
    token_hi = max(token_lo, int(np.ceil(np.percentile(real_tok_lens, high_pct))))

    # Per-relation totals are deterministic (round-robin assignment below), so caps
    # can be computed up-front.
    rel_totals = Counter(relations[i % len(relations)] for i in range(n))
    max_object_relation_fraction = syn_cfg.get("max_object_relation_fraction", 0.3)
    max_object_global_fraction = syn_cfg.get("max_object_global_fraction", 0.15)
    max_pair_fraction = syn_cfg.get("max_pair_fraction", 0.15)
    syllable_tolerance = syn_cfg.get("syllable_tolerance", 2.5)

    rel_object_counts: dict[str, Counter] = {rel: Counter() for rel in relations}
    global_object_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    template_counts: dict[str, Counter] = {rel: Counter() for rel in relations}
    syllable_counts: dict[str, int] = {}
    total_syllable_slots = 0
    seen_names: set[str] = set()

    rows = []
    synthetic_token_lens = []
    for i in range(n):
        rel = relations[i % len(relations)]
        entry = by_rel[rel]
        rel_total = rel_totals[rel]

        template = entry["template_cycler"].next()
        template_counts[rel][template] += 1

        rel_cap = max(1, math.ceil(max_object_relation_fraction * rel_total))
        global_cap = max(1, math.ceil(max_object_global_fraction * n))
        pair_cap = max(1, math.ceil(max_pair_fraction * rel_total))
        answer, answer_tok = _pick_object(
            rng, entry["objects"], template,
            rel_object_counts[rel], global_object_counts, pair_counts,
            rel_cap, global_cap, pair_cap)
        rel_object_counts[rel][answer] += 1
        global_object_counts[answer] += 1
        pair_counts[(template, answer)] += 1

        name, n_tokens, total_syllable_slots = _pseudo_name(
            rng, tokenizer, (token_lo, token_hi), syllable_counts,
            total_syllable_slots, syllable_tolerance, seen_names)
        seen_names.add(name)
        synthetic_token_lens.append(n_tokens)

        # Paraphrases, instantiated for this pseudo-entity subject via the exact
        # same (relation, template) curated lookup as prepare_facts() — never used
        # in training, only for the held-out generalization probe. Mirroring the
        # real-fact path exactly (rather than sampling from a relation-wide pool)
        # keeps the per-fact template<->paraphrase link intact for synthetic facts too.
        paraphrase_templates = paraphrase_map.get((rel, template.replace("{subject}", "{}")), [])
        paraphrases = [_fill_subject(t, name) for t in paraphrase_templates]

        rows.append({
            "fact_id": f"syn_{i}",
            "relation": rel,
            "subject": name,
            "prompt": template.format(subject=name),
            "paraphrases": paraphrases,
            "neighborhood_prompts": [],
            "answer": answer,
            "answer_token_id": answer_tok,
            "target_false": "",
        })

    df = pd.DataFrame(rows)
    n_with_para = sum(1 for r in rows if r["paraphrases"])
    print(f"[data] {len(df)} synthetic facts over {len(relations)} relations "
          f"({n_with_para} with paraphrases).")

    _write_diversity_report(out_dir, real_tok_lens, synthetic_token_lens,
                             rel_object_counts, global_object_counts,
                             template_counts, pair_counts,
                             syllable_counts, total_syllable_slots, n)
    _write_synthetic_facts_csv(out_dir, df)
    return df


def _write_diversity_report(out_dir: Path, real_tok_lens: list[int], syn_tok_lens: list[int],
                             rel_object_counts: dict[str, Counter], global_object_counts: Counter,
                             template_counts: dict[str, Counter], pair_counts: Counter,
                             syllable_counts: dict[str, int], total_syllable_slots: int,
                             n: int) -> pd.DataFrame:
    """Write <out_dir>/synthetic_diversity_report.csv and print a summary, so
    "no small subset dominates" is a checkable artifact instead of an assumption.
    """

    def pct(arr, p):
        return float(np.percentile(arr, p)) if len(arr) else float("nan")

    rows = []
    for label, arr in [("real", real_tok_lens), ("synthetic", syn_tok_lens)]:
        rows.append({"section": "token_length", "key1": label, "key2": "mean",
                     "count": len(arr), "value": float(np.mean(arr)) if arr else float("nan")})
        for p in (5, 25, 50, 75, 95):
            rows.append({"section": "token_length", "key1": label, "key2": f"p{p}",
                         "count": len(arr), "value": pct(arr, p)})

    for rel, counts in rel_object_counts.items():
        total = sum(counts.values())
        for obj, c in counts.most_common():
            rows.append({"section": "object_frequency_by_relation", "key1": rel, "key2": obj,
                         "count": c, "value": c / total if total else float("nan")})

    total_global = sum(global_object_counts.values())
    for obj, c in global_object_counts.most_common():
        rows.append({"section": "object_frequency_global", "key1": obj, "key2": "",
                     "count": c, "value": c / total_global if total_global else float("nan")})

    for rel, counts in template_counts.items():
        total = sum(counts.values())
        for template, c in counts.most_common():
            rows.append({"section": "template_frequency", "key1": rel, "key2": template[:80],
                         "count": c, "value": c / total if total else float("nan")})

    for (template, obj), c in pair_counts.most_common():
        rows.append({"section": "template_object_pair_frequency", "key1": template[:80],
                     "key2": obj, "count": c, "value": c / n if n else float("nan")})

    pool_size = len(_SYLLABLES)
    used_syllables = len(syllable_counts)
    max_syllable_rate = max((c / total_syllable_slots for c in syllable_counts.values()),
                             default=0.0)
    rows.append({"section": "syllable_coverage", "key1": "pool_fraction_used", "key2": "",
                 "count": used_syllables, "value": used_syllables / pool_size})
    rows.append({"section": "syllable_coverage", "key1": "max_single_syllable_rate", "key2": "",
                 "count": total_syllable_slots, "value": max_syllable_rate})
    rows.append({"section": "syllable_coverage", "key1": "uniform_rate", "key2": "",
                 "count": pool_size, "value": 1 / pool_size})

    report = pd.DataFrame(rows)
    path = out_dir / "synthetic_diversity_report.csv"
    report.to_csv(path, index=False)

    max_obj_global_rate = max((c / total_global for c in global_object_counts.values()),
                               default=0.0)
    print("\n[data] Synthetic diversity report:")
    print(f"  subject token length (BPE): real   mean={np.mean(real_tok_lens):.2f} "
          f"median={pct(real_tok_lens, 50):.1f} p5={pct(real_tok_lens, 5):.1f} "
          f"p95={pct(real_tok_lens, 95):.1f}")
    if syn_tok_lens:
        print(f"                              synth  mean={np.mean(syn_tok_lens):.2f} "
              f"median={pct(syn_tok_lens, 50):.1f} p5={pct(syn_tok_lens, 5):.1f} "
              f"p95={pct(syn_tok_lens, 95):.1f}")
    print(f"  syllable pool coverage: {used_syllables}/{pool_size} used, "
          f"max single-syllable usage rate={max_syllable_rate:.3f} "
          f"(uniform={1 / pool_size:.3f})")
    print(f"  most-reused object overall: rate={max_obj_global_rate:.3f} "
          f"out of {n} facts")
    print(f"  full report ({len(report)} rows) written to {path}")
    return report


def _write_synthetic_facts_csv(out_dir: Path, df: pd.DataFrame) -> None:
    """Plain-CSV export of the generated facts/prompts for manual spot-checking
    without any tooling (the parquet artifacts need pandas to open).
    """
    export = df[["fact_id", "relation", "subject", "prompt", "paraphrases", "answer"]].copy()
    export["paraphrases"] = export["paraphrases"].apply(lambda ps: " | ".join(ps))
    path = out_dir / "synthetic_facts.csv"
    export.to_csv(path, index=False)
    print(f"[data] human-readable synthetic facts written to {path}")
