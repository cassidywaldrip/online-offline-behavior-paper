# ============================================================================
# cov_vac experiment — central config (vaccine refusal; the anchor outcome).
# Same layout as the other outcomes (cand20 / mail_nov / vote20): dataprep.py builds
# full_df, finetune_*.py produce the MIL content caches, grid.py scores the combination
# grid, figs.py draws the per-outcome figures. dataprep.py here was ported from the
# original analysis notebook (14_summary_and_figures.ipynb); see README.md.
# ============================================================================
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> config_paths
import config_paths as P

# ---------------------------------------------------------------- paths
EXPO_DIR    = P.EXPO_DIR
COV_VAC_DIR = os.path.join(EXPO_DIR, "cov_vac")
FIG_DIR     = os.path.join(COV_VAC_DIR, "figures")

SURVEY_PATH       = P.SURVEY_HANDLES_PATH
UID_HANDLE_PATH   = P.UID_HANDLE_PATH
INTERSECTION_PATH = P.INTERSECTION_PATH
AUTHORED_PARQUET      = P.AUTHORED_PARQUET
SIMFEED_TWEETS_DIR    = P.SIMFEED_TWEETS_DIR
UID_TO_TWEET_IDS_PATH = P.UID_TO_TWEET_IDS_PATH
FOLLOW_PATH           = P.FOLLOW_PATH

# ---------------------------------------------------------------- outcome
# Per respondent, the outcome is their most-hesitant (minimum) cov_vac value across RELEVANT_WAVES.
# refuser = cov_vac == 1 ("definitely will NOT get vaccinated") on the 1-5 scale.
OUTCOME_COL    = "cov_vac"
RELEVANT_WAVES = [5, 6, 7, 8, 9, 10, 11, 12, 13]   # Jun-Nov 2020, pre-availability
POS_VALUES     = {1}                # refuser -> positive
NEG_VALUES     = {2, 3, 4, 5}       # everyone else
MISSING        = {-99}
POS_LABEL      = "vaccine refuser"
# (The original pipeline also ran an 'acceptor' task, cov_vac==5; the committed figures are refuser.)

# ---------------------------------------------------------------- leakage / temporal cutoff
# Behavior = getting vaccinated; first possible 2020-12-11 (Pfizer EUA; first shots Dec 14).
# The committed corpus already predates this (authored ends 2020-12-10, feed ends 2020-12-09),
# so every tweet is pre-behavior and no re-cut was needed. See ../PAPER_DESIGN_NOTES.md §2.
BEHAVIOR_FLOOR = "2020-12-11"

# ---------------------------------------------------------------- content keyword filter (COVID topic)
KWRE = re.compile(r'covid|corona|vaccin|\bvax|pfizer|moderna|astrazeneca|biontech|pandemic|lockdown|'
                  r'quarantin|\bmask|fauci|\bcdc\b|sars-cov|wuhan|\bmrna\b|warp speed|social distan|anti-?vax')
NPCA_A, NPCA_S = 15, 20   # authored / feed PCA dims

# ---------------------------------------------------------------- content caches (MIL+CT-BERT)
# Out-of-fold head-prob + 1024-d pooled embeddings from the vaccine fine-tune
# (finetune_authored_mil.py / finetune_exposure_mil.py). Produced into the shared
# intermediate store (see config_paths); shared with the election "reuse" option.
CONTENT = {
  "authored": {"oof": P.VACCINE_AUTHORED_OOF, "emb": P.VACCINE_AUTHORED_EMB},
  "feed":     {"oof": P.VACCINE_FEED_OOF,     "emb": P.VACCINE_FEED_EMB},
}

# Data-prep outputs (dataprep.py writes these; grid.py / figs.py read them).
FULL_DF_PATH = os.path.join(COV_VAC_DIR, "cov_vac_full_df.pkl")
META_PATH    = os.path.join(COV_VAC_DIR, "cov_vac_meta.pkl")

# ---------------------------------------------------------------- demographics (identical to nb14 / other variables)
INCOME_ORDER = {'Less than $15,000':0,'$15,000 to $24,999':1,'$25,000 to $34,999':2,'$35,000 to $49,999':3,
                '$50,000 to $74,999':4,'$75,000 to $99,999':5,'$100,000 to $149,999':6,'$150,000 or more':7}
AGE_ORDER  = {'18 to 24':0,'25 to 44':1,'45 to 64':2,'65 and over':3}
EDUC_ORDER = {'High School or Less':0,'Some College':1,'College Degree':2,'Graduate Degree':3}
ORDINAL_COLS = ['income_cat_10','age_cat_4','education_cat']
NOMINAL_COLS = ['gender','race','employ','party']
DEMO_COLS    = ORDINAL_COLS + NOMINAL_COLS
# Demographics BASELINE (Demo block) uses only the reported subgroup dimensions — age, gender, party.
DEMO_MODEL_ORDINAL = ['age_cat_4']
DEMO_MODEL_NOMINAL = ['gender','party']

BASE_MODEL   = P.BASE_MODEL
RANDOM_STATE = P.RANDOM_STATE
