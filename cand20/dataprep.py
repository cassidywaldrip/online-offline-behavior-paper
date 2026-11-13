# cand20 data prep: concatenate the vote20 per-wave CSP_W*.csv files (cand20 is in W1-10),
# recover `handle` from `id` via the stacked file, and build the per-respondent table using each
# respondent's LATEST pre-election cand20 answer (value + date). Binary y (1=Biden vs 2=Trump;
# drop 3/4/5/missing) + network audience-lean on cand20 + per-respondent cutoff_date (= latest
# answer date, leak-free since answering intent implies not-yet-voted). Canonical set only.
# Saves cand20_full_df.pkl + cand20_meta.pkl.
import os, glob, pickle, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict
import config as C

NEEDED = ["id", "wave", "EndDate", C.OUTCOME_COL] + C.DEMO_COLS

def load_csp():
    files = sorted(glob.glob(C.CSP_GLOB))
    if not files: raise FileNotFoundError(f"no CSP_W*.csv files found at {C.CSP_GLOB}")
    parts = []
    for f in files:
        hdr = pd.read_csv(f, nrows=0).columns.tolist()
        cols = [c for c in NEEDED if c in hdr]
        if C.OUTCOME_COL not in cols:
            print(f"  skip {os.path.basename(f)} (no {C.OUTCOME_COL})", flush=True); continue
        parts.append(pd.read_csv(f, usecols=cols, low_memory=False))
        print(f"  loaded {os.path.basename(f)}: {len(parts[-1])} rows", flush=True)
    raw = pd.concat(parts, ignore_index=True)
    print(f"concatenated -> {len(raw)} rows", flush=True)
    return raw

def id_to_handle():
    h = pd.read_csv(C.HANDLES_SOURCE, usecols=["id", "handle"], low_memory=False)
    h = h[h["handle"].notna() & ~h["handle"].astype(str).isin(["", "NA", "NaN"])]
    return dict(zip(h["id"], h["handle"]))

def preprocess_survey(df):
    df = df.copy()
    if C.WAVES is not None: df = df[df["wave"].isin(C.WAVES)]
    df[C.OUTCOME_COL] = pd.to_numeric(df[C.OUTCOME_COL], errors="coerce")
    df = df[df[C.OUTCOME_COL].isin(C.POS_VALUES | C.NEG_VALUES)]   # keep Biden(1)/Trump(2); drop other/undecided/missing
    df["income_cat_10"] = df["income_cat_10"].map(C.INCOME_NUM_TO_STR)
    for col in C.DEMO_COLS: df = df[df[col].notna()]
    df["race"]          = df["race"].map(C.RACE_MAP)
    df["education_cat"] = df["education_cat"].map(C.EDUC_MAP)
    df["income_cat_10"] = df["income_cat_10"].map(C.INCOME_MAP)
    df["employ"]        = df["employ"].map(C.EMPLOY_MAP)
    df["party"]         = df["party"].map(C.PARTY_MAP)
    return df

def latest_response_map(df, col):
    # value at each id's LATEST valid answer date
    rec = {}
    for rid, val, date in zip(df["id"], df[col], df["EndDate"]):
        try: parsed = datetime.strptime(str(date)[:10], "%Y-%m-%d").date()
        except Exception: continue
        rec.setdefault(rid, []).append((parsed, val))
    return {rid: sorted(v, key=lambda x: x[0])[-1][1] for rid, v in rec.items()}   # LATEST

def latest_date_map(df):
    rec = {}
    for rid, date in zip(df["id"], df["EndDate"]):
        try: parsed = datetime.strptime(str(date)[:10], "%Y-%m-%d").date()
        except Exception: continue
        rec.setdefault(rid, []).append(parsed)
    return {rid: max(v).isoformat() for rid, v in rec.items()}

def apply_ordinal_encoding(df):
    df = df.copy()
    df["income_cat_10"] = df["income_cat_10"].map(C.INCOME_ORDER)
    df["age_cat_4"]     = df["age_cat_4"].map(C.AGE_ORDER)
    df["education_cat"] = df["education_cat"].map(C.EDUC_ORDER)
    return df

def build():
    raw = load_csp()
    survey = preprocess_survey(raw)
    id2h = id_to_handle()
    uh = pd.read_csv(C.UID_HANDLE_PATH); handle_to_uid = dict(zip(uh["handle"], uh["user_id"]))

    out_map   = latest_response_map(survey, C.OUTCOME_COL)       # candidate at latest answer
    demo_maps = {c: latest_response_map(survey, c) for c in C.DEMO_COLS}
    date_map  = latest_date_map(survey)                          # cutoff = latest answer date
    rids = list(out_map)
    demo = pd.DataFrame({
        "id":    rids,
        "y":     [1 if int(out_map[r]) in C.POS_VALUES else 0 for r in rids],   # positive = Biden
        "handle": [id2h.get(r) for r in rids],
        "cutoff_date": [date_map.get(r) for r in rids],
        **{c: [demo_maps[c][r] for r in rids] for c in C.DEMO_COLS},
    })
    demo["uid"] = demo["handle"].map(handle_to_uid)
    demo = apply_ordinal_encoding(demo)
    demo["uid_str"] = demo["uid"].apply(lambda x: str(int(x)) if pd.notna(x) else None)
    print(f"usable cand20 respondents: {len(demo)} | pos-rate {demo['y'].mean():.3f} | with UID {demo['uid_str'].notna().sum()}", flush=True)

    canon = set(pd.read_csv(C.INTERSECTION_PATH)["uid"].astype(str))
    full = demo[demo["uid_str"].isin(canon)].copy().reset_index(drop=True)
    print(f"after canonical-intersection filter: {len(full)}  (pos rate {full['y'].mean():.3f})", flush=True)
    print(f"cutoff_date range: {full['cutoff_date'].min()} .. {full['cutoff_date'].max()}", flush=True)

    # ---- NETWORK audience-lean on the cand20 BINARY outcome (external panelists only) ----
    will = {}
    for r in rids:
        uid = handle_to_uid.get(id2h.get(r))
        if uid is not None and pd.notna(uid):
            will[str(int(uid))] = 1 if int(out_map[r]) in C.POS_VALUES else 0
    foll = pickle.load(open(C.FOLLOW_PATH, "rb"))
    external = (set(will) & set(foll)) - set(full["uid_str"])
    asum = defaultdict(float); acnt = defaultdict(int)
    for u in external:
        b = will[u]
        for f in foll[u]:
            f = str(f); asum[f] += b; acnt[f] += 1
    MINF = 5
    lean = {f: asum[f]/acnt[f] for f in acnt if acnt[f] >= MINF}
    nr = [{"uid_str": str(u),
           "net_lean": (np.mean([lean[str(f)] for f in foll.get(str(u), []) if str(f) in lean]) if any(str(f) in lean for f in foll.get(str(u), [])) else np.nan),
           "net_nacc_log": np.log1p(sum(1 for f in foll.get(str(u), []) if str(f) in lean))}
          for u in full["uid_str"]]
    full = full.merge(pd.DataFrame(nr), on="uid_str", how="left")
    NET = ["net_lean", "net_nacc_log"]
    print(f"network coverage: {full['net_lean'].notna().sum()} / {len(full)}", flush=True)

    full.to_pickle(os.path.join(C.CAND20_DIR, "cand20_full_df.pkl"))
    pickle.dump({"NET": NET, "ORDINAL": C.ORDINAL_COLS, "NOMINAL": C.NOMINAL_COLS},
                open(os.path.join(C.CAND20_DIR, "cand20_meta.pkl"), "wb"))
    print("saved cand20_full_df.pkl + cand20_meta.pkl", flush=True)
    return full, NET

if __name__ == "__main__":
    build()
