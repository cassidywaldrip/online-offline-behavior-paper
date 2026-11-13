# Data

This repository contains **code only**. The input data are not redistributed: they include
survey microdata, voter-file party, a reconstructed follow graph, and tweet corpora, all of
which are access-restricted. This file documents what the code expects so a holder of the data
can wire it up.

## Where paths are configured

Every file location is declared in **`config_paths.py`** at the repository root. Nothing is
hard-coded in the individual scripts. Set one environment variable:

```bash
export COVID_TRACES_DATA=/path/to/your/data
```

or edit `DATA_ROOT` in `config_paths.py`. All inputs are resolved relative to it; the per-outcome
pipelines write their run-time outputs back into the repository under `<outcome>/`.

## Expected inputs (under `$COVID_TRACES_DATA`)

| Path (default sub-path) | What it is |
|---|---|
| `raw/stacked_waves_handles.csv` | Stacked COVID States survey waves: `id`, `wave`, `EndDate`, the outcome items (`cov_vac`, `cand20`, `mail_nov`, `vote20`), demographics, and Twitter `handle`. |
| `raw/vote20_waves/CSP_W*.csv` | Per-wave survey exports that carry the election items (`cand20`/`mail_nov`/`vote20`) with `id`/`wave`/`EndDate` but no handle (handle is recovered by joining `id` to the stacked file). |
| `intermediate/uid_to_handle.csv` | Map `handle` → Twitter `user_id`. |
| `intermediate/modeling_respondents_intersection.csv` | Canonical set of modeled respondents (`uid`) — those with usable traces. |
| `intermediate/uid_to_followee_uids.pkl` | Follow graph: `dict {user_id_str -> [followee_user_id_str, ...]}`. |
| `intermediate/vaccine_relevant_simfeed_tweets/*.parquet` | Vaccine-topic simulated-feed tweets (`tweet_id`, `text`). |
| `intermediate/uid_to_vaccine_simfeed_tweet_ids.pkl` | `dict {uid_str -> [tweet_id, ...]}` for the vaccine simfeed. |
| `intermediate/authored_canonical_tweets.parquet` | Respondents' own tweets, canonicalized (`id_str`, `userid`, `eff_text`). |
| `raw/decahose_samples/`, `raw/decahose_dense/` | Decahose archives used to build the **election** simulated feed (sparse `*.json.gz` samples; dense daily `gnip.lz4`). Only needed to rebuild the election feed. |

## Content-model caches (produced by this code, shared)

The MIL+CT-BERT fine-tunes write out-of-fold head-scores and pooled embeddings. The vaccine ones
are shared (cov_vac uses them, and the election outcomes can reuse them instead of re-fine-tuning):

```
intermediate/finetune_authored_mil_oof.pkl   finetune_authored_mil_emb.pkl
intermediate/finetune_exposure_mil_oof.pkl   finetune_exposure_mil_emb.pkl
```

The election outcomes write their own per-outcome caches into `<outcome>/` by default (see each
`config.py`); the "reuse the vaccine caches" alternative is documented there as well.

## Pretrained encoder

The MIL fine-tune downloads **CT-BERT** (`digitalepidemiologylab/covid-twitter-bert-v2`) from
HuggingFace on first use. No local copy is required.
