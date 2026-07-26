# LoRA's Effect on Layer-wise Factual Predictions

Does LoRA fine-tuning on factual knowledge make the correct answer emerge
**earlier** in the forward pass, or does it mainly shift the **final layers**?
And does this differ between facts the model already knows and newly introduced
facts?

Method: compare a small decoder-only LM before/after LoRA fine-tuning on factual
statements, decoding hidden states at every layer with **logit lens** and
**tuned lens**, plus a lens-free **activation patching** causal check.

TAU NLP course project.

> **Feasibility gate**: before real runs, [feasibility/](feasibility/) scores the
> base model on the full dataset and smoke-tests both lenses — it answers the
> decision rule (enough known facts?) and produces the known/unknown split.

## Pipeline

```
prepare_data      load facts (CounterFact by default), keep single-token answers,
                  build synthetic pseudo-entity facts from the same templates
score_base        score the base model; split known/unknown; DECISION-RULE gate
                  (needs >= data.min_known_facts known facts or it aborts and
                  tells you to move one model size up)
build_conditions  relation-stratified sampling of known / unknown / synthetic
train_lora        LoRA fine-tune; adapter checkpoint every N steps
analyze           layer-wise lens metrics for base + every checkpoint
                  (train prompts + held-out paraphrases)
patch             activation patching: earliest layer whose patched stream flips
                  the base model to the learned answer
```

Every stage reads/writes artifacts under `output_dir`, so stages can run in
separate (Kaggle) sessions.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# fast end-to-end sanity run (pythia-160m, tiny samples)
python -m lora_lens.run --config configs/dev.yaml --stages all

# real run (pythia-410m-deduped)
python -m lora_lens.run --config configs/default.yaml --stages all
```

Run a subset of stages, or override any config key from the CLI:

```bash
python -m lora_lens.run --config configs/default.yaml \
    --stages score_base \
    --set model.name=EleutherAI/pythia-1b-deduped \
    --set data.max_records=5000
```

## Configuring model and data

Everything lives in the YAML configs ([configs/default.yaml](configs/default.yaml)):

- **Model**: `model.name` — any HF causal LM. Pythia models are preferred because
  pretrained tuned lenses exist for the whole deduped suite (70m…6.9b) plus
  `gpt2`/`gpt2-large`. For GPT-2 also set `lora.target_modules: [c_attn]`.
- **Data**: `data.dataset` + `data.fields` — the column mapping lets you swap
  CounterFact for LAMA/T-REx, ZsRE, etc. without code changes. If the dataset has
  no paraphrase column, set `data.fields.paraphrases: null` (paraphrase holdout
  eval is skipped).
- **Lenses**: `lens.use_logit` / `lens.use_tuned`; `lens.tuned_lens_id` defaults
  to the model name. If no pretrained tuned lens exists for the model, the
  pipeline warns and continues with the logit lens only.

## Running on Kaggle

Use [kaggle/kaggle_pipeline.ipynb](kaggle/kaggle_pipeline.ipynb). Enable a GPU
(T4/P100), set the repo URL in the first cell, and run. Key points:

- Output goes to `/kaggle/working/outputs` (persisted; download `results/`).
- fp16 everywhere — Kaggle's T4 (compute 7.5) and P100 (6.0) **do not support
  bf16**. Log-softmax is computed in fp32 regardless. This is already the
  config default; don't change it to bf16 on Kaggle.
- The whole 410m run fits comfortably in a single session (LoRA on ~1.5k short
  prompts is minutes on a T4; analysis is the longer part). Use *Save & Run All*
  for free background execution.
- Stages are resumable: if a session dies, re-run with `--stages` starting from
  the last completed stage (artifacts are on disk).

## Outputs

```
outputs/
  config_resolved.yaml       exact config used
  facts.parquet              filtered single-token facts
  synthetic.parquet          pseudo-entity facts
  facts_scored.parquet       base-model scores incl. rank, logprob, top-1 pred
  conditions.parquet         sampled known/unknown/synthetic sets
  lora/step_*/, final/       adapter checkpoints + training_log.csv
  results/
    layerwise.parquet        long format: variant x fact x lens x layer metrics
    summary.csv              accuracy, mean logprob, mean first-layer-of-appearance
    patching.csv             per-layer flip results + first_flip_layer
```

Headline metrics: **first layer of appearance** (earliest layer where the answer
is top-1) per lens, answer probability/rank per layer, and the patching
first-flip layer — all as a function of training step and per condition.

## Methodological guardrails built in

1. **Leading-space tokenization** — answers are tokenized as `" " + answer`
   everywhere (`utils.encode_answer`).
2. **Single-token answers only** — rank is ill-defined otherwise.
3. **Paraphrase holdout** — paraphrases are never in LoRA training; they are the
   generalization probes in `analyze`.
4. **Metrics vs. training step** — adapters checkpointed every
   `lora.checkpoint_every` steps; don't compare conditions at one fixed step.
5. **Right padding + explicit last-real-token gather** — never left padding.
6. **fp16 weights, fp32 log-softmax** — Kaggle GPUs have no bf16.
7. **Relation stratification** — per-relation cap in condition sampling
   (`data.max_relation_fraction`).
8. **Wrong-answer audit** — `score_base` reports what wrong predictions look
   like (competing entity vs. punctuation).

**The lens-validity catch**: the tuned lens is trained on the *base* model and
is technically invalid for the LoRA model. We deliberately use the base lens
for both — keeping the ruler fixed is what licenses attributing differences to
LoRA. The robustness check (retraining the lens on the fine-tuned model) and
the lens-free activation patching stage exist to triangulate this.

## References

- LoRA: Low-Rank Adaptation of Large Language Models (2021)
- The Hidden Space of Transformer Language Adapters (2024)
- Eliciting Latent Predictions from Transformers with the Tuned Lens (Belrose et al., 2023) — https://arxiv.org/pdf/2303.08112
- Pretrained lenses: `AlignmentResearch/tuned-lens` HF space; docs: https://tuned-lens.readthedocs.io
