# vote20 metrics grid — SINGLE OVERALL MODEL per feature set, sliced by subgroup.
# Mirrors presentation_grid.py: content = MIL+CT-BERT embeddings (PCA 15/20), network =
# audience-lean (on vote20), C tuned per fold (nested). One model fit on all covered people,
# out-of-fold predictions (25x 5-fold), metrics computed per subgroup. Covered pop = people
# with authored covid/vaccine tweets. PREREQ: run dataprep.py (and fine-tune, unless reusing).
import os, itertools, pickle, numpy as np, pandas as pd
import config as C
full = pd.read_pickle(os.path.join(C.CAND20_DIR, "cand20_full_df.pkl")); full["uid_str"] = full["uid_str"].astype(str)
meta = pickle.load(open(os.path.join(C.CAND20_DIR, "cand20_meta.pkl"), "rb"))
NET = meta["NET"]; ORD, NOM = C.DEMO_MODEL_ORDINAL, C.DEMO_MODEL_NOMINAL   # Demo = reported dims only (age/gender/party)

def merge(df, oof, emb, pfx):
    e = pd.read_pickle(emb); e["uid_str"] = e["uid_str"].astype(str); ec = [c for c in e.columns if c != "uid_str"]
    e = e.rename(columns={c: f"{pfx}e_{c}" for c in ec})
    s = pd.read_pickle(oof)[["uid_str", "ft_score"]].rename(columns={"ft_score": f"{pfx}_score"}); s["uid_str"] = s["uid_str"].astype(str)
    return df.merge(s, on="uid_str", how="left").merge(e[["uid_str"]+[f"{pfx}e_{c}" for c in ec]], on="uid_str", how="left"), [f"{pfx}e_{c}" for c in ec]
full, AEMB = merge(full, C.CONTENT["authored"]["oof"], C.CONTENT["authored"]["emb"], "a")
full, SEMB = merge(full, C.CONTENT["feed"]["oof"],     C.CONTENT["feed"]["emb"],     "s")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
CGRID = np.logspace(-2, 2, 9)
def embp(cols, k): return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("p", PCA(n_components=k, random_state=C.RANDOM_STATE))]), cols
def netb(): return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NET
def transformers(sig):
    t = []
    if "D" in sig: t += [("ord", StandardScaler(), ORD), ("nom", OneHotEncoder(handle_unknown="ignore"), NOM)]
    if "N" in sig: t += [("N", *netb())]
    if "A" in sig: t += [("A", *embp(AEMB, C.NPCA_A))]
    if "S" in sig: t += [("S", *embp(SEMB, C.NPCA_S))]
    return t
def clf(sig): return Pipeline([("f", ColumnTransformer(transformers(sig), remainder="drop")),
                               ("c", LogisticRegressionCV(Cs=CGRID, cv=3, scoring="roc_auc", class_weight="balanced", max_iter=2000, n_jobs=1))])

ORDER = ["D", "A", "S", "N"]; NAME = {"D": "Demo", "A": "Auth", "S": "Feed", "N": "Net"}
combos = []
for k in range(1, 5):
    for cb in itertools.combinations(ORDER, k): combos.append(list(cb))
combos.sort(key=lambda c: (len(c), [ORDER.index(x) for x in c]))
cname = lambda c: "+".join(NAME[x] for x in c)

df = full[full["a_score"].notna()].reset_index(drop=True)      # covered pop (has authored content)
y = df["y"].astype(int).values
SUBDEF = [("Overall", None, None), ("Male", "gender", "Male"), ("Female", "gender", "Female"),
          ("Democrat", "party", "Democrat"), ("Republican", "party", "Republican"), ("Independent/Other", "party", "Independent/Other"),
          ("Age 18-24", "age_cat_4", 0), ("Age 25-44", "age_cat_4", 1), ("Age 45-64", "age_cat_4", 2), ("Age 65+", "age_cat_4", 3)]
SUB = []
for lab, col, val in SUBDEF:
    mask = np.ones(len(df), bool) if col is None else (df[col] == val).values
    yy = y[mask]
    if yy.sum() >= 25 and (len(yy)-yy.sum()) >= 25: SUB.append((lab, mask))
    else: print(f"skip {lab} (n={mask.sum()}, pos={yy.sum()})", flush=True)

R = 25
splits = [list(StratifiedKFold(5, shuffle=True, random_state=s).split(df, y)) for s in range(R)]
def ci(a): a = np.asarray(a); m = a.mean(); s = a.std(ddof=1)/np.sqrt(len(a)); return m, m-1.96*s, m+1.96*s
RES = {(lab, cname(c)): {"roc": [], "pr": []} for lab, _ in SUB for c in combos}
rows = []
for c in combos:
    est = clf(c); oof_sum = np.zeros(len(df))
    for r in range(R):
        oof = cross_val_predict(est, df, y, cv=splits[r], method="predict_proba", n_jobs=-1)[:, 1]; oof_sum += oof
        for lab, mask in SUB:
            RES[(lab, cname(c))]["roc"].append(roc_auc_score(y[mask], oof[mask]))
            RES[(lab, cname(c))]["pr"].append(average_precision_score(y[mask], oof[mask]))
    oof_mean = oof_sum/R
    for lab, mask in SUB:
        RES[(lab, cname(c))]["roc"] = np.array(RES[(lab, cname(c))]["roc"]); RES[(lab, cname(c))]["pr"] = np.array(RES[(lab, cname(c))]["pr"])
        yy = y[mask]; pp = oof_mean[mask]; n = int(mask.sum()); prev = yy.mean(); floor = max(prev, 1-prev)
        rm, rlo, rhi = ci(RES[(lab, cname(c))]["roc"]); pm, plo, phi = ci(RES[(lab, cname(c))]["pr"])
        acc = accuracy_score(yy, (pp >= 0.5).astype(int)); k = max(1, int(round(0.10*n))); sel = np.argsort(-pp)[:k]
        rows.append({"subgroup": lab, "n": n, "prevalence": round(prev, 4), "combination": cname(c),
                     "roc_auc": round(rm, 4), "roc_lo": round(rlo, 4), "roc_hi": round(rhi, 4),
                     "pr_auc": round(pm, 4), "pr_lo": round(plo, 4), "pr_hi": round(phi, 4),
                     "exact_acc_0.5": round(acc, 4), "majority_floor": round(floor, 4),
                     "precision_top10pct": round(yy[sel].mean(), 4), "recall_top10pct": round(yy[sel].sum()/yy.sum(), 4)})
    print(f"done combo {cname(c)}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(C.FIG_DIR, "cand20_metrics_by_combination.csv"), index=False)
pickle.dump({"RES": RES, "SUB": [l for l, _ in SUB], "combos": [cname(c) for c in combos]}, open(os.path.join(C.CAND20_DIR, "cand20_grid.pkl"), "wb"))
print(f"\nsaved figures/cand20_metrics_by_combination.csv ({len(rows)} rows) + cand20_grid.pkl", flush=True)
print("DONE", flush=True)
