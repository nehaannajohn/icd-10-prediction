# ICD-10-CM Coding with Confidence

Predicts ICD-10-CM (FY2025) diagnosis codes from a doctor–patient transcript and gives every code a
confidence score, so low-confidence codes can be picked out for human review. An LLM walks the ICD-10-CM
hierarchy one level at a time; three models × three runs each vote, and a code's confidence is the
rank-weighted share of runs that chose it.

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │ data/                                        │
                         │ corpus/fy2025 (CMS tabular + index XML)      │
                         │ eval/ (33 transcripts + reference codes)     │
                         └──────────────────────────────────────────────┘
                                              │
                                              ▼
PREDICT · one model, one run (src/code_prediction)
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│ │ Prompt       │──▶│ LLM call per     │──▶│ Check reply      │──▶│ Expand selected codes│  │
│ │ system + ICL │   │ level (Bedrock), │   │ JSON, codes from │   │ to children (≤ 7     │  │
│ │ + transcript │   │ all open branches│   │ candidate list;  │   │ levels); billable    │  │
│ │ + candidates │   │ in one call      │   │ retry on error   │   │ codes are final      │  │
│ └──────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────────┘  │
│                                                                              │             │
│                                         ┌────────────────────────────────────┘             │
│                                         ▼                                                  │
│                              ┌───────────────────────────┐                                 │
│                              │ Post-process: validate,   │                                 │
│                              │ tree path, Excludes1 flags│                                 │
│                              └───────────────────────────┘                                 │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │ × 3 models × 3 runs
                                              ▼
CONFIDENCE (src/uncertainty_estimation)
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Vote per code: rank-weighted share of the 9 runs that chose it (Opus 0.55 · Sonnet 0.27 ·  │
│ Terra 0.18, split across each model's runs), at every hierarchy level                      │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
EVALUATE (src/evaluation)                                  APP (app/)
hierarchical + exact P/R/F1; ECE (with noise floor),       Streamlit: paste a transcript or pick an
Brier, AUROC, AURC / E-AURC, risk–coverage                 example; codes sorted by confidence
```

## Repository layout

```
data/
  corpus/fy2025/          CMS ICD-10-CM FY2025 (April 2025) tabular + index XML
  eval/                   transcripts + reference codes (not committed; see data/README.md)
src/
  common/                 config, Bedrock client (plain-JSON + retries + tracing), ICD-10 tree, trace, artifacts
  code_prediction/        tree search, prompts, post-processing, runner
  uncertainty_estimation/ voting (U0–U2), method comparison, plots
  evaluation/             cleaned eval set + split, hierarchical/exact metrics, calibration, model ranking
app/                      Streamlit app: ensemble, rendering, cached examples
test/                     unit tests mirroring src/ and app/ (Bedrock tests marked `integration`)
artifacts/                results (only these summary folders are committed)
  model_selection_20261101_131756/            model ranking on tune: ranking.json, ranking.csv
  uncertainty_eval_tune_20261101_152756/      confidence methods on tune: methods.csv, metrics.json,
                                              reliability + risk–coverage plots
  uncertainty_eval_heldout_20261101_155219/   confidence methods on held-out (same files)
  <stage>_<model>_<ts>/                       per-run outputs + traces (created locally, git-ignored)
config.yaml               paths, models, split, prediction, uncertainty, evaluation, app settings
pyproject.toml            package and dependencies
```

## Setup

### 1. Get the code
The repository is private. Log in to GitHub with an account that has been given access, then clone:
```bash
gh auth login
gh repo clone nehaannajohn/icd-10-prediction
cd icd-10-prediction
```

### 2. Create the environment
Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):
```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

### 3. AWS Bedrock access
The pipeline calls Amazon Bedrock (us-west-2) with your own AWS account. The account needs model access to
the models in `config.yaml` (`claude-opus-5-5`, `claude-sonnet-5-5`, `gpt-6.1-sol`, `gpt-6.1-terra`).
Credentials are never stored in the repo; set them per shell:
```bash
export AWS_PROFILE=<your-profile>
export AWS_REGION=us-west-2
```

### 4. Data
Put the take-home CSV at `data/eval/Test_Project_ICD10_Dataset.csv`, then build the cleaned eval set and the
fixed split (3 ICL / 10 tune / 20 held-out):
```bash
.venv/bin/python -m src.evaluation.dataset
```

### 5. Check
```bash
.venv/bin/pytest -m "not integration"   # unit tests, no Bedrock calls
```

## Usage

### Predict codes for an eval split
```bash
.venv/bin/python -m src.code_prediction.run --model claude-opus-5-5 --split tune --seeds 0 --stage prediction
.venv/bin/python -m src.evaluation.run_eval --run artifacts/prediction_claude-opus-5-5_<ts>
.venv/bin/python -m src.evaluation.compare_runs artifacts/prediction_*_<ts>          # rank models
```
Each run writes `artifacts/<stage>_<model>_<ts>/` with a config snapshot, a summary, and per-transcript outputs
and step-by-step traces in `raw/`. Run folders, traces and checks stay local; only summary folders are committed.

### Confidence: compare methods
```bash
.venv/bin/python -m src.code_prediction.run --model <model> --split tune --seeds 1 2 --stage uncertainty
.venv/bin/python -m src.uncertainty_estimation.compare_methods --selection artifacts/model_selection_<ts> \
    --runs artifacts/prediction_<m>_<ts> ... artifacts/uncertainty_<m>_<ts> ... --split tune
```

### App
```bash
.venv/bin/python -m app.build_examples --runs artifacts/uncertainty_<m>_<ts> ...   # cached examples (optional)
.venv/bin/streamlit run app/streamlit_app.py                                        # http://127.0.0.1:8501
```
Paste a transcript and choose **Run ensemble** (9 runs, about 1–2 minutes), or pick a cached example.
The app binds to 127.0.0.1 only.

## Where AI helped

I used Claude Code as my AI agent across design and implementation.

- **Design.** I set out the approach and iterated with the agent section by section until we agreed on each
  part. The agent read the brief, the data, the CMS code files and the literature, and surfaced issues
  (FY2025-only reference codes, single-category block ids, models that reject `temperature`, the small-sample
  ECE problem); I made the design decisions.
- **Implementation.** Step by step: the agent proposed each code snippet, I reviewed it and changed it where
  needed, and only then was it written. Every live model run was approved before it ran.

## Results

**Model selection** (tune, 10 transcripts, 1 run each; `artifacts/model_selection_20261101_131756/`):

| Model | Exact P / R / F1 | Hierarchical F1 |
|---|---|---|
| claude-opus-5-5 | 0.51 / 0.71 / 0.59 | 0.68 |
| claude-sonnet-5-5 | 0.49 / 0.61 / 0.54 | 0.63 |
| gpt-6.1-terra | 0.43 / 0.56 / 0.48 | 0.56 |
| gpt-6.1-sol | 0.36 / 0.59 / 0.45 | 0.53 |

**Confidence on held-out** (20 transcripts, 3 models × 3 runs; `artifacts/uncertainty_eval_heldout_20261101_155219/`):

| Method | AUROC | E-AURC [95% CI] | Brier | Predicted-set exact F1 |
|---|---|---|---|---|
| **U2-rank (chosen on tune)** | 0.754 | 0.183 [0.138, 0.226] | 0.286 | 0.39 |
| U0 stated confidence | 0.727 | 0.231 [0.159, 0.298] | 0.350 | 0.37 |
| U1 Opus × 3 runs | 0.577 | 0.317 [0.282, 0.343] | 0.524 | 0.38 |

The ensemble ranks correct codes above incorrect ones better than the model's stated confidence or a single
model's repeat runs. Scores are agreement rates, not probabilities (ECE 0.33 against a chance level of 0.06).
Exact F1 fell from 0.59 on tune to about 0.37–0.40 on held-out, mostly in precision. With 10 tune and 20
held-out transcripts, these numbers are indicative only.
