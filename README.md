# Predicting offline political behavior from online network structure

Code for a study that predicts individuals' **offline survey intentions** from their
**online digital traces**, and asks *which* kind of trace carries the signal. Four survey
outcomes are modeled; for each, a set of feature **blocks** is scored out-of-fold and compared:

| Block | Meaning |
|---|---|
| **Demo** | attribute baseline — demographics (age, gender, party) |
| **Net**  | relational — a leak-free **audience-lean** of the accounts a person follows |
| **Auth** | content — the person's own (authored) tweets, via an attention-MIL CT-BERT model |
| **Feed** | content — the tweets in the person's simulated feed, via the same model |

The headline comparison is **attribute vs. relational**: demographics predict the *modal*
partisan, while network position predicts the *within-party* deviant. Text content is the foil
("structure beats text"). One outcome (turnout) is a **negative control**.

## The four outcomes

| Dir | Outcome | Positive class | Response taken |
|---|---|---|---|
| `cand20/`  | 2020 candidate choice | Biden vs. Trump | latest pre-election answer |
| `cov_vac/` | COVID-19 vaccine refusal (anchor) | refuser (most-hesitant == 1) | most-hesitant value, waves 5–13 |
| `mail_nov/`| mail-voting intent | pole {3,4} vs. {1,2} | latest pre-election answer |
| `vote20/`  | turnout intent (**negative control**) | "definitely vote" | latest pre-election answer |

## Layout

```
config_paths.py        central path config (set COVID_TRACES_DATA); the ONLY place data paths live
<outcome>/             one self-contained pipeline per outcome:
  config.py              outcome settings (outcome coding, keyword filter, PCA dims, content source)
  dataprep.py            build full_df: sample, outcome label, demographics, audience-lean
  finetune_mil_*.py      GPU: attention-MIL CT-BERT -> per-person head-score + pooled embedding
  (cov_vac has finetune_authored_mil.py + finetune_exposure_mil.py)
  build_election_simfeed*.py   (election outcomes) build the pre-election simulated feed from decahose
  grid.py                score every feature-block combination out-of-fold, sliced by subgroup
  figs.py                per-outcome figures
robustness/            cross-cutting robustness checks (FB1/FB2/FB3) — see robustness/README.md
paper_assets/          make_paper_assets.py -> the paper's figures & tables (committed under figs/, tables/)
DATA.md                what input data the code expects
requirements.txt       pinned core dependencies
```

## Running

Each pipeline reads its own `config.py`, so run it with that outcome directory on `PYTHONPATH`:

```bash
export COVID_TRACES_DATA=/path/to/data          # see DATA.md

# one outcome, end to end (cov_vac shown; the GPU step needs CUDA)
PYTHONPATH="$PWD/cov_vac" python cov_vac/dataprep.py
BASE= PYTHONPATH="$PWD/cov_vac" python cov_vac/finetune_authored_mil.py   # GPU
PYTHONPATH="$PWD/cov_vac" python cov_vac/finetune_exposure_mil.py         # GPU
PYTHONPATH="$PWD/cov_vac" python cov_vac/grid.py
PYTHONPATH="$PWD/cov_vac" python cov_vac/figs.py

# the paper's combined figures & tables (after all outcomes' grids exist)
PYTHONPATH="$PWD" python paper_assets/make_paper_assets.py
```

The election outcomes (`cand20`, `mail_nov`, `vote20`) use a single `finetune_mil_<outcome>.py`
with `CORPUS=authored` then `CORPUS=feed`, and can instead **reuse** the vaccine content caches
(skipping the GPU step) — see the `CONTENT` block in each `config.py`.

## Notes for reviewers

- **Anonymized for release.** No absolute paths, author identity, or raw data are included; all
  inputs resolve from `config_paths.py` / `COVID_TRACES_DATA`. See `DATA.md`.
- **`cov_vac/dataprep.py` was ported from a notebook.** The original vaccine analysis was
  notebook-driven; the data-building was extracted into `cov_vac/dataprep.py` so this release is
  notebook-free. It reproduces the exact `full_df` the grid consumes (sample, most-hesitant label,
  demographics, audience-lean); earlier-exploration features the final model does not use were dropped.
- **`robustness/fb1_neighbor_demographics.py` is a reconstruction (validated).** Its original build
  script was not preserved; it re-implements the documented method and was verified to reproduce the
  committed `cand20_fb1_neighbor_demo_grid.csv` **exactly** (max |AUC diff| = 0.0000). It reuses each
  outcome's `dataprep` helpers so the neighbour block is encoded from the same panelist universe as
  the audience-lean.
- **Privacy.** No script writes handles, user ids, or tweet text to any output; the diverging-users
  check (`robustness/fb3_diverging_users.py`) emits only an aggregate, de-identified table.
- **Cross-validation.** Every grid uses `LogisticRegressionCV` (nested `C`, `cv=3`),
  `StratifiedKFold(5)` repeated 25×, out-of-fold predictions, `class_weight="balanced"`.
