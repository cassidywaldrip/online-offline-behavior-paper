# ============================================================================
# vote20 (turnout intent) experiment config. Election-keyword content.
# Turnout is the paper's NEGATIVE CONTROL outcome.
# Paths come from the repo-level config_paths.py (no machine-specific paths here).
# ============================================================================
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> config_paths
import config_paths as P

# ---------------------------------------------------------------- paths
EXPO_DIR   = P.EXPO_DIR
VOTE20_DIR = os.path.join(EXPO_DIR, "vote20")
FIG_DIR    = os.path.join(VOTE20_DIR, "figures")

# ----- survey data that CONTAINS vote20 (per-wave CSP_W*.csv files) -----
# Raw per-wave files have id/wave/EndDate/vote20 + demographics but NO handle; handle is
# recovered by joining `id` to SURVEY_HANDLES_PATH (the stacked file that carries handles).
CSP_GLOB       = os.path.join(P.DATA_ROOT, "raw", "vote20_waves", "CSP_W*.csv")
HANDLES_SOURCE = P.SURVEY_HANDLES_PATH
WAVES          = None   # None = use all waves present in the CSP files; or a list to restrict

UID_HANDLE_PATH   = P.UID_HANDLE_PATH
INTERSECTION_PATH = P.INTERSECTION_PATH
# same tweet corpora as the vaccine study
AUTHORED_PARQUET      = P.AUTHORED_PARQUET
SIMFEED_TWEETS_DIR    = P.SIMFEED_TWEETS_DIR
UID_TO_TWEET_IDS_PATH = P.UID_TO_TWEET_IDS_PATH
FOLLOW_PATH           = P.FOLLOW_PATH

# ---------------------------------------------------------------- outcome
OUTCOME_COL = "vote20"
POS_VALUES  = {1}        # "Yes, I will definitely vote"  -> positive class (1)
NEG_VALUES  = {2, 3}     # not sure / decided not to      -> negative class (0)
EXCLUDE     = {4}        # "not eligible to vote"          -> dropped
MISSING     = {-99}      # dropped
POS_LABEL   = "definite voter"   # used in figure axis/labels

# ---------------------------------------------------------------- demographics
INCOME_ORDER = {'Less than $15,000':0,'$15,000 to $24,999':1,'$25,000 to $34,999':2,'$35,000 to $49,999':3,
                '$50,000 to $74,999':4,'$75,000 to $99,999':5,'$100,000 to $149,999':6,'$150,000 or more':7}
AGE_ORDER  = {'18 to 24':0,'25 to 44':1,'45 to 64':2,'65 and over':3}
EDUC_ORDER = {'High School or Less':0,'Some College':1,'College Degree':2,'Graduate Degree':3}
ORDINAL_COLS = ['income_cat_10','age_cat_4','education_cat']   # used for the SAMPLE filter (require non-null)
NOMINAL_COLS = ['gender','race','employ','party']
DEMO_COLS    = ORDINAL_COLS + NOMINAL_COLS
# Demographics BASELINE (Demo block) uses only the reported subgroup dimensions — age, gender, party.
DEMO_MODEL_ORDINAL = ['age_cat_4']
DEMO_MODEL_NOMINAL = ['gender','party']
INCOME_NUM_TO_STR = {1.0:'Less than $10,000',2.0:'$10,000 to $14,999',3.0:'$15,000 to $24,999',4.0:'$25,000 to $34,999',
                     5.0:'$35,000 to $49,999',6.0:'$50,000 to $74,999',7.0:'$75,000 to $99,999',8.0:'$100,000 to $149,999',
                     9.0:'$150,000 to $199,999',10.0:'$200,000 or more'}
RACE_MAP  = {'White':'White','African American':'African American','Hispanic':'Hispanic','Asian American':'Asian American',
             'Other':'Other','Pacific Islander':'Other','Native American':'Other'}
EDUC_MAP  = {'College Degree':'College Degree','Graduate Degree':'Graduate Degree','High School Graduate':'High School or Less',
             'Some College':'Some College','Some High School or Less':'High School or Less'}
PARTY_MAP = {'Democrat':'Democrat','Independent':'Independent/Other','Other':'Independent/Other','Republican':'Republican'}
INCOME_MAP = {'Less than $10,000':'Less than $15,000','$10,000 to $14,999':'Less than $15,000','$15,000 to $24,999':'$15,000 to $24,999',
              '$25,000 to $34,999':'$25,000 to $34,999','$35,000 to $49,999':'$35,000 to $49,999','$50,000 to $74,999':'$50,000 to $74,999',
              '$75,000 to $99,999':'$75,000 to $99,999','$100,000 to $149,999':'$100,000 to $149,999',
              '$150,000 to $199,999':'$150,000 or more','$200,000 or more':'$150,000 or more'}
EMPLOY_MAP = {'Full-time':'Full-time','Gig/Contract':'Part-time/Gig/Contract','Home-maker':'Self-employed/Home-maker',
              'Part-time':'Part-time/Gig/Contract','Retired':'Retired','Self-employed':'Self-employed/Home-maker',
              'Student':'Student','Unemployed':'Unemployed'}

# ---------------------------------------------------------------- content (MIL+CT-BERT)
# ELECTION keyword filter (vote20 is a voting-intent outcome, so content = election tweets).
KWRE = re.compile(r'\b(vote|voting|voter|voters|ballot|election|electoral|poll|polls|turnout|'
                  r'mail-in|mail in|absentee|early voting|register to vote|registered to vote|'
                  r'trump|biden|pence|harris|maga|democrat|democrats|democratic|republican|republicans|'
                  r'gop|campaign|swing state|debate|electionday)\b|#election2020', re.I)
NPCA_A, NPCA_S = 10, 10   # authored / feed content-embedding PCA dims (tuned per outcome by CV add-on ROC-AUC)

# election simfeed source options:
#  (a) sparse  : decahose json.gz samples (~55 MB/day) -> build_election_simfeed.py
#  (b) DENSE   : full decahose gnip.lz4  (~4.25 GB/day) -> build_election_simfeed_dense.py  [default]
DECAHOSE_DIR       = P.DECAHOSE_DIR
DENSE_DECAHOSE_DIR = P.DENSE_DECAHOSE_DIR
DENSE_WINDOW       = ("2020-09-01", "2020-11-03")   # election run-up (inclusive); the pre-vote exposure window
# gnip TSV field positions: id=0, author_uid=1, text=9
GNIP_COL_TID, GNIP_COL_AUID, GNIP_COL_TEXT = 0, 1, 9
EL_SIMFEED_TEXT    = os.path.join(VOTE20_DIR, "election_simfeed_text.pkl")                # dict tweet_id -> text
EL_UID_TO_TWEETS   = os.path.join(VOTE20_DIR, "uid_to_election_simfeed_tweet_ids.pkl")    # dict uid_str -> [tweet_id]

# Content signal source. Default: RE-FINE-TUNE on vote20 (faithful replication); the
# pipeline writes these caches into VOTE20_DIR. To REUSE the vaccine-tuned embeddings
# instead (skip the GPU fine-tune), point these at the vaccine caches (see config_paths).
CONTENT = {
  "authored": {"oof": os.path.join(VOTE20_DIR, "vote20_authored_mil_oof.pkl"),
               "emb": os.path.join(VOTE20_DIR, "vote20_authored_mil_emb.pkl")},
  "feed":     {"oof": os.path.join(VOTE20_DIR, "vote20_exposure_mil_oof.pkl"),
               "emb": os.path.join(VOTE20_DIR, "vote20_exposure_mil_emb.pkl")},
}
# Reuse alternative (uncomment to skip GPU fine-tuning):
# CONTENT = {
#   "authored": {"oof": P.VACCINE_AUTHORED_OOF, "emb": P.VACCINE_AUTHORED_EMB},
#   "feed":     {"oof": P.VACCINE_FEED_OOF,     "emb": P.VACCINE_FEED_EMB},
# }

BASE_MODEL   = P.BASE_MODEL   # CT-BERT, for the re-fine-tune path
RANDOM_STATE = P.RANDOM_STATE
