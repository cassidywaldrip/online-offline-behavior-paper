"""Central path configuration for the whole pipeline.

All code in this repository reads its file locations from here, so that no
absolute or machine-specific paths are hard-coded in the individual scripts.

Two roots:
  * CODE_ROOT  -- this repository (auto-detected). Per-outcome pipelines, their
                 run-time outputs, and the paper assets live under here.
  * DATA_ROOT  -- where the (not-redistributed) input data lives. Set the
                 COVID_TRACES_DATA environment variable, or edit DATA_ROOT below.

The raw inputs are survey microdata, voter-file party, a follow graph, and
tweet corpora, none of which are included in this repository (see DATA.md).
The expected sub-paths under DATA_ROOT are declared here; adjust them to match
your local copy. Nothing here reads or transmits any data on import.
"""
import os

# --------------------------------------------------------------------------- roots
CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("COVID_TRACES_DATA", "/PATH/TO/COVID_STATES_DATA")

# Per-outcome code + run-time outputs live under CODE_ROOT/<outcome>/.
# (The original internal layout called this directory EXPO_DIR; kept as an alias
#  so the per-outcome configs read naturally.)
EXPO_DIR = CODE_ROOT

# --------------------------------------------------------------------------- raw inputs (DATA_ROOT)
# Survey: stacked per-wave COVID States microdata with respondent id, wave, EndDate,
# the outcome items, demographics, and (for the handle join) Twitter handles.
SURVEY_HANDLES_PATH = os.path.join(DATA_ROOT, "raw", "stacked_waves_handles.csv")
# Map from survey respondent id to Twitter user id.
UID_HANDLE_PATH     = os.path.join(DATA_ROOT, "intermediate", "uid_to_handle.csv")
# The modeling-respondent canonical intersection (respondents with usable traces).
INTERSECTION_PATH   = os.path.join(DATA_ROOT, "intermediate", "modeling_respondents_intersection.csv")

# Follow graph: dict {user_id_str -> [followee_user_id_str, ...]}.
FOLLOW_PATH         = os.path.join(DATA_ROOT, "intermediate", "uid_to_followee_uids.pkl")

# Simulated-feed (vaccine topic): the tweets each respondent could have seen.
SIMFEED_TWEETS_DIR    = os.path.join(DATA_ROOT, "intermediate", "vaccine_relevant_simfeed_tweets")
UID_TO_TWEET_IDS_PATH = os.path.join(DATA_ROOT, "intermediate", "uid_to_vaccine_simfeed_tweet_ids.pkl")

# Decahose tweet archives used to build the election simulated feed.
#  * sparse daily json.gz samples, and
#  * dense daily gnip.lz4 full archives.
DECAHOSE_DIR        = os.path.join(DATA_ROOT, "raw", "decahose_samples")
DENSE_DECAHOSE_DIR  = os.path.join(DATA_ROOT, "raw", "decahose_dense")

# Respondents' own (authored) tweets, canonicalized to one parquet
# (columns: id_str, userid, eff_text). A derived cache, not raw data.
AUTHORED_PARQUET    = os.path.join(DATA_ROOT, "intermediate", "authored_canonical_tweets.parquet")

# Vaccine-study content caches (MIL+CT-BERT out-of-fold scores + pooled embeddings),
# shared by cov_vac and available as the "reuse" option for the election outcomes.
VACCINE_AUTHORED_OOF = os.path.join(DATA_ROOT, "intermediate", "finetune_authored_mil_oof.pkl")
VACCINE_AUTHORED_EMB = os.path.join(DATA_ROOT, "intermediate", "finetune_authored_mil_emb.pkl")
VACCINE_FEED_OOF     = os.path.join(DATA_ROOT, "intermediate", "finetune_exposure_mil_oof.pkl")
VACCINE_FEED_EMB     = os.path.join(DATA_ROOT, "intermediate", "finetune_exposure_mil_emb.pkl")

# Pretrained encoder used for the MIL fine-tune (downloaded from HuggingFace).
BASE_MODEL   = "digitalepidemiologylab/covid-twitter-bert-v2"
RANDOM_STATE = 42
