"""Robustness check FB3 — who are the respondents whose traces diverge from their survey answer?

Built for candidate choice (the motivating example). Reconstructs a trace-only out-of-fold
P(Biden) per respondent from the saved MIL head-scores (authored, feed) + the audience-lean,
then contrasts confidently-misclassified HIGH-signal respondents with the rest. The point: the
diverging cases are NOT low-signal noise -- they have abundant, internally-consistent online
signal that simply points the opposite way from the survey answer (cross-pressured / genuine
attitude-behaviour mismatch), which is the irreducible residual.

PRIVACY: this script writes only an AGGREGATE, de-identified table (counts and medians). It never
writes handles, user ids, or tweet text to any file. (The paper's illustrative examples were
written by hand as fully de-identified paraphrases; no per-user export is produced here.)

Output: paper_assets/tables/appendixC_divergence_agg.{csv,tex}
Run:    PYTHONPATH="<repo>/cand20" python robustness/fb3_diverging_users.py
"""
import os, numpy as np, pandas as pd
import config as C
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

O    = C.OUTCOME_COL                       # cand20
ODIR = os.path.join(C.EXPO_DIR, O)
REPO = C.EXPO_DIR
TAB  = os.path.join(REPO, "paper_assets", "tables")

full = pd.read_pickle(os.path.join(ODIR, f"{O}_full_df.pkl"))
au   = pd.read_pickle(C.CONTENT["authored"]["oof"])[["uid_str", "ft_score"]].rename(columns={"ft_score": "auth_p"})
fe   = pd.read_pickle(C.CONTENT["feed"]["oof"])[["uid_str", "ft_score"]].rename(columns={"ft_score": "feed_p"})
for d in (full, au, fe): d["uid_str"] = d["uid_str"].astype(str)

df = full.merge(au, on="uid_str", how="inner").merge(fe, on="uid_str", how="inner").copy()
df = df.dropna(subset=["net_lean", "net_nacc_log", "auth_p", "feed_p", "y"])
print("modeled-with-all-traces n =", len(df), flush=True)

# ---- trace-only OOF P(positive): L2 logistic on [auth_p, feed_p, net_lean, net_nacc_log] ----
X = df[["auth_p", "feed_p", "net_lean", "net_nacc_log"]].values
y = df["y"].astype(int).values
pipe = Pipeline([("s", StandardScaler()),
                 ("c", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))])
probs = np.zeros(len(df))
for r in range(5):
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=r).split(X, y))
    probs += cross_val_predict(pipe, X, y, cv=splits, method="predict_proba")[:, 1]
df["trace_p"] = probs / 5
print("trace-only OOF ROC-AUC =", round(roc_auc_score(y, df["trace_p"]), 3), flush=True)

# ---- signal volume: scored followees + topic-relevant authored tweets ----
df["scored_followees"] = np.expm1(df["net_nacc_log"]).round().astype(int)
tw = pd.read_parquet(C.AUTHORED_PARQUET, columns=["userid", "eff_text"])
tw["userid"] = tw["userid"].astype(str)
tw = tw[tw["userid"].isin(set(df["uid_str"]))].copy()
rel = tw[tw["eff_text"].str.contains(C.KWRE, na=False)].groupby("userid").size()
df["auth_rel_tweets"] = df["uid_str"].map(rel).fillna(0).astype(int)

# ---- divergence + high-signal + cross-pressure ----
df["correct"]    = (df["trace_p"] >= 0.5) == (df["y"] == 1)
df["conf_wrong"] = (~df["correct"]) & ((df["trace_p"] - 0.5).abs() >= 0.30)
med_f = df["scored_followees"].median(); med_t = df["auth_rel_tweets"].median()
df["high_signal"] = (df["scored_followees"] >= med_f) & (df["auth_rel_tweets"] >= med_t)
# cross-pressured = voter-file party opposite to the stated vote
df["cross_pressured"] = ((df["party"] == "Republican") & (df["y"] == 1)) | ((df["party"] == "Democrat") & (df["y"] == 0))
# do all three trace channels point the same way? (systematic online stance, not noise)
chan = (df["auth_p"] > 0.5).astype(int) + (df["feed_p"] > 0.5).astype(int) + (df["net_lean"] > 0.5).astype(int)
df["channels_agree"] = chan.isin([0, 3])

# ---- aggregated, de-identified contrast table ----
def summ(sub, label):
    return dict(group=label, n=len(sub),
                med_tweets=int(sub["auth_rel_tweets"].median()) if len(sub) else 0,
                med_followees=int(sub["scored_followees"].median()) if len(sub) else 0,
                pct_channels_agree=round(100*sub["channels_agree"].mean(), 1) if len(sub) else 0,
                pct_cross=round(100*sub["cross_pressured"].mean(), 1) if len(sub) else 0)
hi = df[df["high_signal"]]
agg = pd.DataFrame([summ(df, "All modeled"),
                    summ(hi, "High-signal"),
                    summ(hi[hi["correct"]],   "High-signal, trace agrees w/ survey"),
                    summ(hi[hi["conf_wrong"]], "High-signal, trace DIVERGES from survey")])
print("\n==== AGGREGATED CONTRAST ====\n" + agg.to_string(index=False), flush=True)

os.makedirs(TAB, exist_ok=True)
agg.to_csv(os.path.join(TAB, "appendixC_divergence_agg.csv"), index=False)
hdr = {"group": "Group", "n": "$n$", "med_tweets": "Median elec. tweets", "med_followees": "Median scored followees",
       "pct_channels_agree": "\\% channels agree", "pct_cross": "\\% cross-pressured"}
L = [r"% Appendix C(a): characterizing trace--survey divergence in candidate choice.",
     r"% 'Trace diverges' = trace-only OOF prediction on the opposite side of 0.5 from the survey answer,",
     r"% with |p-0.5|>=0.30. 'Channels agree' = authored text, feed, and audience lean all point the same way.",
     r"% 'Cross-pressured' = voter-file party opposite to the stated vote.",
     r"\begin{tabular}{lrrrrr}", r"\toprule",
     " & ".join(hdr[c] for c in agg.columns) + r" \\", r"\midrule"]
for _, r in agg.iterrows():
    L.append(" & ".join(str(r[c]) for c in agg.columns) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "appendixC_divergence_agg.tex"), "w").write("\n".join(L))
print("\nwrote", os.path.join(TAB, "appendixC_divergence_agg.{csv,tex}"), flush=True)
