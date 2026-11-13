"""Robustness check FB2 — "are the digital traces just a demographic proxy?"

For one outcome, predict (a) the survey behavior and (b) each demographic attribute
from the trace blocks alone (no demographic features), using the SAME feature
construction and cross-validation protocol as that outcome's grid.py. If traces were
merely a demographic proxy, trace->behavior could not out-run trace->demographics; it
does, so the traces carry outcome-specific signal.

Blocks : Net (audience-lean), Auth (authored text), Feed (feed text), All traces (union).
Targets: behavior (positive class == 1); party Democrat/Republican/Independent-Other
         (one-vs-rest); gender Female/Male; age 18-24/25-44/45-64/65+ (one-vs-rest).
Output : <outcome>/figures/<outcome>_fb2_predict_demographics.csv
         (columns: target, prevalence, block, roc_auc, ci95). Consumed by Table 5.

Run per outcome (choose the outcome via PYTHONPATH):
  PYTHONPATH="<repo>/cand20"  python robustness/fb2_predict_demographics.py
  PYTHONPATH="<repo>/cov_vac" python robustness/fb2_predict_demographics.py   (etc.)
"""
import os, pickle, numpy as np, pandas as pd
import config as C
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

O     = C.OUTCOME_COL
ODIR  = os.path.join(C.EXPO_DIR, O)
R     = 25
CGRID = np.logspace(-2, 2, 9)


def merge(df, oof, emb, pfx):
    e = pd.read_pickle(emb); e["uid_str"] = e["uid_str"].astype(str); ec = [c for c in e.columns if c != "uid_str"]
    e = e.rename(columns={c: f"{pfx}e_{c}" for c in ec})
    s = pd.read_pickle(oof)[["uid_str", "ft_score"]].rename(columns={"ft_score": f"{pfx}_score"}); s["uid_str"] = s["uid_str"].astype(str)
    return df.merge(s, on="uid_str", how="left").merge(e[["uid_str"]+[f"{pfx}e_{c}" for c in ec]], on="uid_str", how="left"), [f"{pfx}e_{c}" for c in ec]


# ---- features: exactly as the outcome's grid.py builds them ----
full = pd.read_pickle(os.path.join(ODIR, f"{O}_full_df.pkl")); full["uid_str"] = full["uid_str"].astype(str)
meta = pickle.load(open(os.path.join(ODIR, f"{O}_meta.pkl"), "rb")); NET = meta["NET"]
full, AEMB = merge(full, C.CONTENT["authored"]["oof"], C.CONTENT["authored"]["emb"], "a")
full, SEMB = merge(full, C.CONTENT["feed"]["oof"],     C.CONTENT["feed"]["emb"],     "s")

def embp(cols, k): return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("p", PCA(n_components=k, random_state=C.RANDOM_STATE))]), cols
def netb():        return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NET
def transformers(sig):
    t = []
    if "N" in sig: t += [("N", *netb())]
    if "A" in sig: t += [("A", *embp(AEMB, C.NPCA_A))]
    if "S" in sig: t += [("S", *embp(SEMB, C.NPCA_S))]
    return t
def clf(sig): return Pipeline([("f", ColumnTransformer(transformers(sig), remainder="drop")),
                               ("c", LogisticRegressionCV(Cs=CGRID, cv=3, scoring="roc_auc",
                                                          class_weight="balanced", max_iter=2000, n_jobs=1))])

df = full[full["a_score"].notna()].reset_index(drop=True)   # covered pop (has authored content)
print(f"[{O}] covered modeled subjects: {len(df)}", flush=True)

# ---- targets (binary), exhaustive over the model's demographic attributes ----
TARGETS = [
    ("OUTCOME (behavior)",                (df["y"].astype(int) == 1).astype(int).values),
    ("party: Democrat vs rest",           (df["party"] == "Democrat").astype(int).values),
    ("party: Republican vs rest",         (df["party"] == "Republican").astype(int).values),
    ("party: Independent/Other vs rest",  (df["party"] == "Independent/Other").astype(int).values),
    ("gender: Female vs Male",            (df["gender"] == "Female").astype(int).values),
    ("gender: Male vs Female",            (df["gender"] == "Male").astype(int).values),
    ("age: 18-24 vs rest",                (df["age_cat_4"] == 0).astype(int).values),
    ("age: 25-44 vs rest",                (df["age_cat_4"] == 1).astype(int).values),
    ("age: 45-64 vs rest",                (df["age_cat_4"] == 2).astype(int).values),
    ("age: 65+ vs rest",                  (df["age_cat_4"] == 3).astype(int).values),
]
BLOCKS = [("Net", "N"), ("Auth", "A"), ("Feed", "S"), ("All traces", "NAS")]

rows = []
for tlab, y in TARGETS:
    if y.sum() < 25 or (len(y) - y.sum()) < 25:
        print(f"  skip {tlab} (too few in a class: pos={int(y.sum())}/{len(y)})", flush=True)
        continue
    prev = float(y.mean())
    splits = [list(StratifiedKFold(5, shuffle=True, random_state=s).split(df, y)) for s in range(R)]
    for blab, sig in BLOCKS:
        est = clf(sig); aucs = []
        for r in range(R):
            oof = cross_val_predict(est, df, y, cv=splits[r], method="predict_proba", n_jobs=-1)[:, 1]
            aucs.append(roc_auc_score(y, oof))
        a = np.asarray(aucs); m = a.mean(); ci = 1.96*a.std(ddof=1)/np.sqrt(len(a))
        rows.append({"target": tlab, "prevalence": round(prev, 3), "block": blab,
                     "roc_auc": round(float(m), 4), "ci95": round(float(ci), 4)})
        print(f"  {tlab:34s} {blab:11s} AUC={m:.4f} ci95={ci:.4f}", flush=True)

os.makedirs(os.path.join(ODIR, "figures"), exist_ok=True)
out = os.path.join(ODIR, "figures", f"{O}_fb2_predict_demographics.csv")
pd.DataFrame(rows)[["target", "prevalence", "block", "roc_auc", "ci95"]].to_csv(out, index=False)
print("saved", out, flush=True)
