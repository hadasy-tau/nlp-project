# LoRA's Effect on Layer-wise Factual Predictions

TAU NLP course project. Research question: does LoRA fine-tuning on factual
knowledge make the correct answer emerge **earlier** in the forward pass, or
does it mainly shift the **final layers** toward it?

## Feasibility pipeline

Before any real experiments, `run_pipeline.py` confirms the whole stack works
and surfaces surprises early. It:

1. **Scores the base model on the full dataset** (CounterFact by default) —
   reports the top-1 success rate and the **known-fact count**. Decision rule:
   we need ≥300 known facts; below that, move one model size up
   (410m → 1b → 1.4b).
2. **Runs the logit lens and the tuned lens** on a sample of prompts (half the
   model got right, half wrong) — verifies hidden-state extraction works and
   that a pretrained tuned lens exists and loads for the chosen model.

Every step prints its own elapsed time, and a timing summary is printed and
saved at the end.

### Outputs

Written to `outputs/<timestamp>-<model>/`:

- `results.json` — config echo, timings, success rate, per-relation breakdown,
  lens summaries
- `per_item_scores.csv` — one row per fact (correct / prob / rank). This is the
  known vs. unknown split for the later LoRA conditions.
- `lens_trajectories.json` — per-layer target prob and rank for the lens sample

### Run on Kaggle

1. Create a Kaggle notebook, enable a GPU (P100 or T4 ×2) and internet access.
2. Copy the three cells from [kaggle/launcher.ipynb](kaggle/launcher.ipynb)
   (clone → `pip install` → run), or upload that notebook directly.
3. Use *Save & Run All (Commit)* for free background execution; grab
   `outputs/` from the notebook's output tab when it finishes.

Alternatively, from a terminal with the Kaggle CLI: fill in your username in
[kaggle/kernel-metadata.json](kaggle/kernel-metadata.json), then
`kaggle kernels push -p kaggle/` and poll with `kaggle kernels status`.

### Run locally (smoke test)

No GPU needed — pythia-70m on 64 facts, a few minutes on CPU:

```
pip install -r requirements.txt
python run_pipeline.py --config configs/smoke.yaml
```

### Configuration

Model and data are configurable via YAML ([configs/](configs/)) — any HF
causal LM and any HF dataset with a prompt column and a target column:

```yaml
model:
  name: EleutherAI/pythia-410m-deduped   # any HF causal LM
  dtype: auto        # fp16 on GPU, fp32 on CPU; Kaggle GPUs have no bf16
  batch_size: 64
data:
  name: NeelNanda/counterfact-tracing    # any HF dataset
  prompt_column: prompt
  target_column: target_true
  limit: null        # null = all rows
lens:
  sample_size: 50    # prompts for the lens smoke test
```

CLI overrides for quick experiments:
`--model EleutherAI/pythia-1b-deduped`, `--limit 500`, `--lens-sample 20`.

### Implementation notes (pitfalls baked in)

- Targets are tokenized with a leading space (`" Paris"`, not `"Paris"`).
- Multi-token targets are dropped (answer rank is ill-defined for them); the
  drop rate is reported.
- Right padding with an explicit logit gather at the last real token.
- fp16 on GPU, but log-softmax always in fp32.
- Both lenses decode the same hidden states; the final row always uses the
  model's own logits (the last hidden state is already normalized — applying
  the lens there would double-normalize).
- Pretrained tuned lenses exist for Pythia deduped models (70m…6.9b), `gpt2`,
  and `gpt2-large` in the `AlignmentResearch/tuned-lens` HF space. Other
  models will fail step 5 with a clear error — that, too, is a feasibility
  answer.
