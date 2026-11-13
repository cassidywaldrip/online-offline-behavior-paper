"""Robustness check FB1 — does the network audience-lean add value OVER neighbours' demographics?

A reviewer could worry that the relational "audience-lean" block (Net) is just an indirect
read of who your followees' audiences ARE demographically. FB1 builds an explicit relational
block from neighbours' DEMOGRAPHICS (NbrDemo) and asks whether Net still adds predictive value
on top of Demo + NbrDemo.

  NbrDemo : for each EXTERNAL panelist (a surveyed respondent NOT in the modeled set), take their
            party/gender/age; target-encode onto the accounts they follow (minimum MINF=5 panelists
            per account); each modeled respondent's NbrDemo feature is the mean over the accounts
            THEY follow -- exactly the leak-free construction used for the audience-lean, but with
            demographics as the encoded quantity instead of the outcome.
  Grid    : Demo, NbrDemo, Net and their unions, scored out-of-fold (25x 5-fold) and sliced by
            party, with the headline contrast being the marginal AUC of adding Net to Demo+NbrDemo.
  Output  : <outcome>/<outcome>_neighbor_demo.pkl  and
            <outcome>/figures/<outcome>_fb1_neighbor_demo_grid.csv   (consumed by Table 6).

  Reconstructed script: the original FB1 build script was not preserved, so this re-implements
  the documented method. It was VALIDATED by re-running it on candidate choice and confirming it
  reproduces the committed cand20_fb1_neighbor_demo_grid.csv exactly (max |AUC diff| = 0.0000
  across all cells; neighbour-demographic features identical). The key is that NbrDemo is encoded
  from the SAME external-panelist universe as the audience-lean -- i.e. the outcome's *preprocessed*
  survey -- which is why this reuses the outcome's own dataprep helpers below.

Run per outcome:  PYTHONPATH="<repo>/<outcome>" python robustness/fb1_neighbor_demographics.py
"""
import os, glob, pickle
from collections import defaultdict
import numpy as np, pandas as pd
import config as C
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

O    = C.OUTCOME_COL
ODIR = os.path.join(C.EXPO_DIR, O)
MINF = 5
ND_COLS = ["nd_demshare", "nd_repshare", "nd_femshare", "nd_age"]


def panelist_demographics():
    """Map every surveyed respondent (with a Twitter uid) -> (party, gender, age ordinal).

    This MUST match the external-panelist universe the outcome's dataprep uses to build the
    audience-lean: the *preprocessed* survey (complete demographics, valid outcome) with the
    same response selection. We therefore reuse the outcome's own dataprep helpers rather than
    re-deriving the sample, so NbrDemo is encoded from exactly the same panelists as Net."""
    import dataprep as DP                                    # the outcome's own dataprep.py
    uh = pd.read_csv(C.UID_HANDLE_PATH)
    handle_to_uid = dict(zip(uh["handle"], uh["user_id"]))

    if hasattr(C, "CSP_GLOB"):                               # election outcomes (cand20/mail_nov/vote20)
        survey = DP.preprocess_survey(DP.load_csp())
        id2h   = DP.id_to_handle()
        pm = DP.latest_response_map(survey, "party")
        gm = DP.latest_response_map(survey, "gender")
        am = DP.latest_response_map(survey, "age_cat_4")
        rids = list(DP.latest_response_map(survey, C.OUTCOME_COL))
        handle_of = lambda r: id2h.get(r)
    else:                                                    # cov_vac
        survey = DP.preprocess_survey(pd.read_csv(C.SURVEY_PATH, low_memory=False))
        pm = DP.most_hesitant_map(survey, "party")
        gm = DP.most_hesitant_map(survey, "gender")
        am = DP.most_hesitant_map(survey, "age_cat_4")
        hm = DP.most_hesitant_map(survey, "handle")
        rids = list(pm)
        handle_of = lambda r: hm.get(r)

    out = {}
    for r in rids:
        uid = handle_to_uid.get(handle_of(r))
        if uid is None or pd.isna(uid): continue
        out[str(int(uid))] = (pm.get(r), gm.get(r), C.AGE_ORDER.get(am.get(r)))
    return out


def build_neighbor_demo(full):
    foll  = pickle.load(open(C.FOLLOW_PATH, "rb"))
    pdem  = panelist_demographics()
    modeled = set(full["uid_str"].astype(str))
    external = (set(pdem) & set(foll)) - modeled                 # panelists who are NOT modeled users
    dem = defaultdict(float); rep = defaultdict(float); fem = defaultdict(float)
    agesum = defaultdict(float); cnt = defaultdict(int)
    for u in external:
        p, g, a = pdem[u]
        for f in foll[u]:
            f = str(f)
            dem[f] += (p == "Democrat"); rep[f] += (p == "Republican"); fem[f] += (g == "Female")
            if pd.notna(a): agesum[f] += a
            cnt[f] += 1
    ok = {f for f in cnt if cnt[f] >= MINF}
    enc = {f: (dem[f]/cnt[f], rep[f]/cnt[f], fem[f]/cnt[f], agesum[f]/cnt[f]) for f in ok}
    rows = []
    for uid in full["uid_str"].astype(str):
        vals = [enc[str(f)] for f in foll.get(str(uid), []) if str(f) in enc]
        if vals:
            m = np.mean(vals, axis=0)
            rows.append({"uid_str": uid, "nd_demshare": m[0], "nd_repshare": m[1], "nd_femshare": m[2], "nd_age": m[3]})
        else:
            rows.append({"uid_str": uid, "nd_demshare": np.nan, "nd_repshare": np.nan, "nd_femshare": np.nan, "nd_age": np.nan})
    return pd.DataFrame(rows)


# ---- assemble modeled frame: demographics + y + network (+ neighbour demographics) ----
full = pd.read_pickle(os.path.join(ODIR, f"{O}_full_df.pkl")); full["uid_str"] = full["uid_str"].astype(str)
meta = pickle.load(open(os.path.join(ODIR, f"{O}_meta.pkl"), "rb")); NET = meta["NET"]

# NbrDemo is encoded from EXTERNAL panelists only — i.e. excluding the FULL canonical modeled set
# (leak-free, exactly as the audience-lean is built), so build it over all of full_df first.
nd_path = os.path.join(ODIR, f"{O}_neighbor_demo.pkl")
nd = build_neighbor_demo(full)
nd.to_pickle(nd_path); print(f"wrote {nd_path} ({nd['nd_demshare'].notna().sum()}/{len(nd)} covered)", flush=True)
df = full.merge(nd, on="uid_str", how="left")

# Score on the SAME population as the main combination grid: respondents with authored content
# (a_score not null). FB1's own blocks don't use content, but matching the grid's covered
# population keeps n and the subgroup set identical for a like-for-like Table 6.
asc = pd.read_pickle(C.CONTENT["authored"]["oof"])[["uid_str", "ft_score"]].rename(columns={"ft_score": "a_score"})
asc["uid_str"] = asc["uid_str"].astype(str)
df = df.merge(asc, on="uid_str", how="left")
df = df[df["a_score"].notna()].reset_index(drop=True)

# cov_vac stores y as the 1-5 scale (refuser == 1); the election outcomes store y as 0/1 already.
y = (df["y"].astype(int).values == 1).astype(int)
ORD, NOM = C.DEMO_MODEL_ORDINAL, C.DEMO_MODEL_NOMINAL
CGRID = np.logspace(-2, 2, 9)

def blocks(sig):
    t = []
    if "D" in sig: t += [("ord", StandardScaler(), ORD), ("nom", OneHotEncoder(handle_unknown="ignore"), NOM)]
    if "B" in sig: t += [("B", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), ND_COLS)]
    if "N" in sig: t += [("N", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NET)]
    return t
def clf(sig): return Pipeline([("f", ColumnTransformer(blocks(sig), remainder="drop")),
                               ("c", LogisticRegressionCV(Cs=CGRID, cv=3, scoring="roc_auc",
                                                          class_weight="balanced", max_iter=2000, n_jobs=1))])

# focused combinations (the headline test is Net added on top of Demo+NbrDemo)
COMBOS = [("Demo", "D"), ("NbrDemo", "B"), ("Net", "N"),
          ("Demo+NbrDemo", "DB"), ("Demo+Net", "DN"), ("NbrDemo+Net", "BN"), ("Demo+NbrDemo+Net", "DBN")]
SUBDEF = [("Overall", None, None), ("Democrat", "party", "Democrat"),
          ("Republican", "party", "Republican"), ("Independent/Other", "party", "Independent/Other")]
SUB = []
for lab, col, val in SUBDEF:
    mask = np.ones(len(df), bool) if col is None else (df[col] == val).values
    yy = y[mask]
    if yy.sum() >= 25 and (len(yy)-yy.sum()) >= 25: SUB.append((lab, mask))

R = 25
splits = [list(StratifiedKFold(5, shuffle=True, random_state=s).split(df, y)) for s in range(R)]
def ci(a): a = np.asarray(a); m = a.mean(); s = a.std(ddof=1)/np.sqrt(len(a)); return m, m-1.96*s, m+1.96*s

rows = []
for name, sig in COMBOS:
    per = {lab: [] for lab, _ in SUB}; oof_sum = np.zeros(len(df))
    for r in range(R):
        oof = cross_val_predict(clf(sig), df, y, cv=splits[r], method="predict_proba", n_jobs=-1)[:, 1]
        oof_sum += oof
        for lab, mask in SUB: per[lab].append(roc_auc_score(y[mask], oof[mask]))
    oof_mean = oof_sum / R
    for lab, mask in SUB:
        yy = y[mask]; pp = oof_mean[mask]; n = int(mask.sum()); prev = yy.mean(); floor = max(prev, 1-prev)
        rocm, rlo, rhi = ci(per[lab]); k = max(1, int(round(0.10*n))); sel = np.argsort(-pp)[:k]
        rows.append({"subgroup": lab, "n": n, "prevalence": round(prev, 4), "combination": name,
                     "roc_auc": round(rocm, 4), "roc_lo": round(rlo, 4), "roc_hi": round(rhi, 4),
                     "exact_acc_0.5": round(accuracy_score(yy, (pp >= 0.5).astype(int)), 4),
                     "majority_floor": round(floor, 4), "precision_top10pct": round(yy[sel].mean(), 4)})
    print(f"done combo {name}", flush=True)

os.makedirs(os.path.join(ODIR, "figures"), exist_ok=True)
out = os.path.join(ODIR, "figures", f"{O}_fb1_neighbor_demo_grid.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("saved", out, flush=True)
