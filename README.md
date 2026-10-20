# CSCI 3052U: Benchmarking LLMs on CantoNLU

This project looks at how well large language models understand Cantonese using the CantoNLU benchmark (Min et al., 2025). We focus on three core tasks: language detection, sentiment analysis, and natural language inference. The goal is to see how LLMs stack up against simple baselines: a majority‑class guesser and a TF‑IDF + logistic regression model.

**Milestone 3** covers the baseline. This includes a working data pipeline, a naive benchmark, a course-level baseline, a reproducible environment, a baseline table and a discrepancy analysis.

## Repository structure

| Folder / file | Contents |
|---|---|
| `src/data.py` | Data loaders for the three tasks and a seeded stratified sampler |
| `notebooks/` | Exploration and data-processing notebooks |
| `data/` | Data instructions (`data/README.md`) |
| `results/` | Experiment results (`baselines.csv`, one row per run) |
| `reports/` | Figures and report material |
| `requirements.txt` | Python dependencies |
| `AI_USE.md`, `CONTRIBUTIONS.md` | AI-use disclosure and contribution record |

**Note:** The CantoNLU files are cloned into `data/canto-nlu/` and are git-ignored

## Setup

Tested on Python 3.13.

```bash
git clone https://github.com/babytara/csci3052u-cantonlu.git
cd csci3052u-cantonlu
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
```

## Get the data

LD and sentiment come from the lab's `canto-nlu` repository, cloned into `data/canto-nlu/` and pinned to commit `2283874`. From the repository root:

```bash
git clone https://github.com/Aatlantise/canto-nlu.git data/canto-nlu
git -C data/canto-nlu checkout 2283874
```

NLI is downloaded from Hugging Face (`hon9kon9ize/yue-all-nli`, pinned to revision `f43c2e709cc3bd448a9c5d5cb5465646cd5f36bb`) when `load_nli()` is called. The first call downloads the data, this takes a few minutes. The lab repo however contains only a 100 row NLI sample, so we use the full Hugging Face data. You can see `data/README.md` for the source, licence and attribution details.

## Check that it works

From the repository root:

```bash
python -c "from src.data import load_ld, load_sentiment, load_nli; print([len(x) for x in load_ld()], [len(x) for x in load_sentiment()], [len(x) for x in load_nli()])"
```

Expected output (train, val, test sizes for each task):

```
[42753, 2425, 2341] [9999, 999, 999] [1114734, 13144, 13202]
```

To try the pipeline without downloading anything, use `load_sample("ld")`, `load_sample("sentiment")` or `load_sample("nli")` instead.

## How to use the loaders

Run Python from the repository root and import them from `src.data`.

| Function | Returns | Label format |
|---|---|---|
| `load_ld()` | `(train, val, test)`, each a list of `(sentence, label)` | ints: 0 = Cantonese, 1 = Mandarin, 2 = code-mixed |
| `load_sentiment()` | `(train, val, test)`, each a list of `(sentence, label)` | strings: `smile`, `ok`, `cry` |
| `load_nli()` | `(train, val, test)`, each a list of `(premise, hypothesis, label)` | strings: `entailment`, `not_entailment` |

**NLI details.** Each Hugging Face triplet (`anchor`, `positive`, `negative`) is unrolled into two pairs, as in the lab's `sample_nli.py`: `(anchor, positive)` is `entailment` and `(anchor, negative)` is `not_entailment`. The pairs alternate by label, so any prefix of the list is balanced. The Hugging Face `dev` split is returned as `val`. Train contains about 1.1 million pairs, so subsample it before training.

**Sampling.** `stratified_sample(data, n, seed=42)` returns `n` rows with the same label proportions as `data`, using a fixed seed. The label must be the last element of each row.

```python
from src.data import load_nli, stratified_sample
train, val, test = load_nli()
train_small = stratified_sample(train, 50_000, seed=42)
```

**Small sample.** `data/sample/` holds a 100-row sample that is committed for each task (`ld_sample.jsonl`, `sentiment_sample.jsonl`, `nli_sample.jsonl`), drawn from the train split with `stratified_sample(..., seed=42)`. It allows you to run the pipeline without needing the full dataset. Regenerate it with `python -m src.make_sample`.

```python
from src.data import load_sample
ld_sample = load_sample("ld")   # also "sentiment" and "nli"
```
**LD + Sentiment Baseline.** 'python -m ld_sentiment_baseline.py' is used to get baseline for LD and sentiment results

## Reproducing results

[TODO]

## Source, licence and attribution

CantoNLU is by Min et al. (2025), arXiv:2510.20670. The `canto-nlu` repository is released under MIT, and the project team lead confirmed that contributions should be treated as CC-BY. Attribute to Min et al. (2025).

## Team

See `CONTRIBUTIONS.md` for each member and `AI_USE.md` for AI-use disclosure.