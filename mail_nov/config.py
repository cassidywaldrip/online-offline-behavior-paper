# ============================================================================
# mail_nov experiment — central config. Mirrors vote20 (CSP-file source, W7-10)
# but the outcome is the 4-point mail_nov item.
#   positive (1): mail_nov in {3, 4}
#   negative (0): mail_nov in {1, 2}
#   dropped:      -99 / missing
# "Same tweets" as vote20/voted20: election-filtered authored + dense election
# simfeed (rebuilt for THIS respondent set), re-fine-tuned on this target.
# ============================================================================
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> config_paths
import config_paths as P

# ---------------------------------------------------------------- paths
EXPO_DIR     = P.EXPO_DIR
MAIL_NOV_DIR = os.path.join(EXPO_DIR, "mail_nov")
FIG_DIR      = os.path.join(MAIL_NOV_DIR, "figures")

# ----- survey data: mail_nov lives in the shared per-wave CSP files (W7-10). dataprep globs
# them (auto-skips waves without mail_nov) and recovers `handle` by joining `id` to the stacked
# handles file. -----
CSP_GLOB       = os.path.join(P.DATA_ROOT, "raw", "vote20_waves", "CSP_W*.csv")
HANDLES_SOURCE = P.SURVEY_HANDLES_PATH
WAVES          = None   # None = use every wave whose CSP has mail_nov (7-10)

UID_HANDLE_PATH   = P.UID_HANDLE_PATH
INTERSECTION_PATH = P.INTERSECTION_PATH
AUTHORED_PARQUET  = P.AUTHORED_PARQUET
FOLLOW_PATH       = P.FOLLOW_PATH

# ---------------------------------------------------------------- outcome
OUTCOME_COL = "mail_nov"
POS_VALUES  = {3, 4}     # positive pole
NEG_VALUES  = {1, 2}
EXCLUDE     = set()
MISSING     = {-99}
POS_LABEL   = "mail_nov = 3/4"   # placeholder label (semantic pending); used in figure axis

# ---------------------------------------------------------------- demographics (identical to nb14)
INCOME_ORDER = {'Less than $15,000':0,'$15,000 to $24,999':1,'$25,000 to $34,999':2,'$35,000 to $49,999':3,
                '$50,000 to $74,999':4,'$75,000 to $99,999':5,'$100,000 to $149,999':6,'$150,000 or more':7}
AGE_ORDER  = {'18 to 24':0,'25 to 44':1,'45 to 64':2,'65 and over':3}
EDUC_ORDER = {'High School or Less':0,'Some College':1,'College Degree':2,'Graduate Degree':3}
ORDINAL_COLS = ['income_cat_10','age_cat_4','education_cat']   # used for the SAMPLE filter (require non-null)
NOMINAL_COLS = ['gender','race','employ','party']
DEMO_COLS    = ORDINAL_COLS + NOMINAL_COLS
# Demographics BASELINE (Demo block) uses only the reported subgroup dimensions — age, gender, party
# — not the full set above, so the Demo bar matches what we slice/report on. (Sample is unchanged.)
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
KWRE = re.compile(r'\b(vote|voting|voter|voters|ballot|election|electoral|poll|polls|turnout|'
                  r'mail-in|mail in|absentee|early voting|register to vote|registered to vote|'
                  r'trump|biden|pence|harris|maga|democrat|democrats|democratic|republican|republicans|'
                  r'gop|campaign|swing state|debate|electionday)\b|#election2020', re.I)
NPCA_A, NPCA_S = 10, 10   # authored / feed content-embedding PCA dims (tuned per outcome by CV add-on ROC-AUC)

# dense election simfeed (rebuilt by build_election_simfeed_dense.py for THIS respondent set).
# mail_nov was asked Jul-Sep 2020, so the feed window PRECEDES the question; tweets are then
# trimmed PER RESPONDENT to strictly before their own cutoff_date (set in dataprep) at bag-build.
DENSE_DECAHOSE_DIR = P.DENSE_DECAHOSE_DIR
DENSE_WINDOW       = ("2020-06-01", "2020-09-30")   # covers up to the latest pre-election mail_nov answers
GNIP_COL_TID, GNIP_COL_AUID, GNIP_COL_TEXT = 0, 1, 9
# Per-respondent cutoff = max(latest mail_nov answer date, FLOOR_CUTOFF). Answering an intent
# question implies not-yet-voted, so tweets up to the answer date are safe; everyone is bumped to
# at least the first-possible-mail-vote date (Sept 4) since no vote could occur before then.
FIXED_CUTOFF       = None             # uniform cutoff (unused)
FLOOR_CUTOFF       = "2020-09-04"     # first possible 2020 mail vote; cutoff = max(answer_date, this)
CUTOFF_COL         = "cutoff_date"    # per-respondent LATEST answer date
EL_SIMFEED_TEXT    = os.path.join(MAIL_NOV_DIR, "election_simfeed_text.pkl")
EL_UID_TO_TWEETS   = os.path.join(MAIL_NOV_DIR, "uid_to_election_simfeed_tweet_ids.pkl")

CONTENT = {
  "authored": {"oof": os.path.join(MAIL_NOV_DIR, "mail_nov_authored_mil_oof.pkl"),
               "emb": os.path.join(MAIL_NOV_DIR, "mail_nov_authored_mil_emb.pkl")},
  "feed":     {"oof": os.path.join(MAIL_NOV_DIR, "mail_nov_exposure_mil_oof.pkl"),
               "emb": os.path.join(MAIL_NOV_DIR, "mail_nov_exposure_mil_emb.pkl")},
}

BASE_MODEL   = P.BASE_MODEL
RANDOM_STATE = P.RANDOM_STATE
