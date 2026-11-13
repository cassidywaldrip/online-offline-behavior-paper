"""cov_vac (vaccine-refusal) data preparation.

PORTED from the original analysis notebook (14_summary_and_figures.ipynb) into a
standalone script, so the release contains no notebooks. It reproduces the exact
data-building the grid consumes:

  * sample  : COVID States respondents, waves 5-13 (pre-availability), with a non-missing
              cov_vac answer and complete demographics, restricted to the canonical
              Twitter-matched intersection;
  * outcome : y = each respondent's MOST-HESITANT (minimum) cov_vac value across those
              waves (refuser = y == 1 on the 1-5 scale);
  * demographics: age/gender/party (+ the full reported set, for subgroup parity);
  * network : leak-free audience-lean, built ONLY from EXTERNAL panelists (respondents
              NOT in the modeled set) target-encoded onto each modeled user's followees
              (minimum MINF=5 scored followees).

Content features (authored / feed text) are NOT built here; they come from the MIL+CT-BERT
caches produced by finetune_authored_mil.py / finetune_exposure_mil.py and are merged in
grid.py. (The original notebook also computed sentence-transformer and keyword-ideology
features used in earlier exploration; those are not used by the final model and are omitted.)

Writes config.FULL_DF_PATH (full_df) and config.META_PATH ({"NET": [...]}).
Run:  PYTHONPATH="<repo>/cov_vac" python cov_vac/dataprep.py
"""
import os, pickle
from datetime import datetime
from collections import defaultdict
import numpy as np, pandas as pd
import config as C

MINF = 5   # minimum scored followees for a usable audience-lean


def preprocess_survey(df):
    df = df[df["wave"].isin(C.RELEVANT_WAVES)].copy()
    df = df[df["cov_vac"].notna() & ~df["cov_vac"].isin([-99.0])]
    df["income_cat_10"] = df["income_cat_10"].map(C.INCOME_NUM_TO_STR)
    for col in C.DEMO_COLS:
        df = df[df[col].notna()]
    df["race"]          = df["race"].map(C.RACE_MAP)
    df["education_cat"] = df["education_cat"].map(C.EDUC_MAP)
    df["income_cat_10"] = df["income_cat_10"].map(C.INCOME_MAP)
    df["employ"]        = df["employ"].map(C.EMPLOY_MAP)
    df["party"]         = df["party"].map(C.PARTY_MAP)
    return df


def most_hesitant_map(df, col):
    """Per respondent, the MINIMUM value across waves (most-hesitant cov_vac answer).
    Sorting (value, date) tuples and taking the first element yields the min value."""
    rec = {}
    for rid, val, date in zip(df["id"], df[col], df["EndDate"]):
        parsed = datetime.strptime(str(date)[:10], "%Y-%m-%d").date()
        rec.setdefault(rid, []).append((val, parsed))
    return {rid: sorted(v)[0][0] for rid, v in rec.items()}


def apply_ordinal_encoding(df):
    df = df.copy()
    df["income_cat_10"] = df["income_cat_10"].map(C.INCOME_ORDER)
    df["age_cat_4"]     = df["age_cat_4"].map(C.AGE_ORDER)
    df["education_cat"] = df["education_cat"].map(C.EDUC_ORDER)
    return df


def build():
    raw_df    = pd.read_csv(C.SURVEY_PATH, low_memory=False)
    survey_df = preprocess_survey(raw_df)

    uid_handle_df = pd.read_csv(C.UID_HANDLE_PATH)
    handle_to_uid = dict(zip(uid_handle_df["handle"], uid_handle_df["user_id"]))

    vac_map    = most_hesitant_map(survey_df, "cov_vac")
    handle_map = most_hesitant_map(survey_df, "handle")   # one handle per respondent
    demo_maps  = {col: most_hesitant_map(survey_df, col) for col in C.DEMO_COLS}

    respondents = list(vac_map)
    demo_df = pd.DataFrame({
        "y":      [int(vac_map[r])    for r in respondents],
        "handle": [handle_map[r]      for r in respondents],
        **{col: [demo_maps[col][r] for r in respondents] for col in C.DEMO_COLS},
    })
    demo_df["uid"]     = demo_df["handle"].map(handle_to_uid)
    demo_df            = apply_ordinal_encoding(demo_df)
    demo_df["uid_str"] = demo_df["uid"].apply(lambda x: str(int(x)) if pd.notna(x) else None)
    print(f"respondents with UID: {demo_df['uid_str'].notna().sum():,} / {len(demo_df):,}", flush=True)

    canonical_uids = set(pd.read_csv(C.INTERSECTION_PATH)["uid"].astype(str))
    full_df = demo_df[demo_df["uid_str"].isin(canonical_uids)].copy().reset_index(drop=True)
    print(f"after canonical-intersection filter: {len(full_df):,}  (refuser rate {(full_df['y']==1).mean():.3f})", flush=True)

    # ---- leak-free NETWORK audience-lean (external panelists only) ----
    will = {}
    wdf = raw_df[raw_df["wave"].isin(C.RELEVANT_WAVES)]
    wdf = wdf[wdf["cov_vac"].notna() & ~wdf["cov_vac"].isin([-99.0])]
    byr = {}
    for rid, cv, h, d in zip(wdf["id"], wdf["cov_vac"], wdf["handle"], wdf["EndDate"]):
        try: dt = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        except Exception: continue
        byr.setdefault(rid, []).append((dt, cv, h))
    for rid, lst in byr.items():
        lst.sort(); cv, h = lst[0][1], lst[0][2]; uid = handle_to_uid.get(h)
        if uid is not None and pd.notna(cv): will[str(int(uid))] = int(cv)

    foll = pickle.load(open(C.FOLLOW_PATH, "rb"))
    canon = set(full_df["uid_str"])
    external = (set(will) & set(foll)) - canon     # panelists who are NOT modeled users
    asum = defaultdict(float); acnt = defaultdict(int); aref = defaultdict(int)
    for u in external:
        cv = will[u]
        for f in foll[u]:
            f = str(f); asum[f] += cv; acnt[f] += 1; aref[f] += (cv == 1)
    lean = {f: asum[f]/acnt[f] for f in acnt if acnt[f] >= MINF}
    refr = {f: aref[f]/acnt[f] for f in acnt if acnt[f] >= MINF}
    nr = []
    for uid in full_df["uid_str"]:
        fs = foll.get(str(uid), [])
        L = [lean[str(f)] for f in fs if str(f) in lean]
        R = [refr[str(f)] for f in fs if str(f) in refr]
        nr.append({"uid_str": str(uid),
                   "net_lean":     np.mean(L) if L else np.nan,
                   "net_refrate":  np.mean(R) if R else np.nan,
                   "net_nacc_log": np.log1p(len(L))})
    net = pd.DataFrame(nr); NET_COLS = ["net_lean", "net_refrate", "net_nacc_log"]
    full_df = full_df.merge(net, on="uid_str", how="left")
    print(f"network coverage: {full_df['net_lean'].notna().sum()} / {len(full_df)}", flush=True)

    full_df.to_pickle(C.FULL_DF_PATH)
    pickle.dump({"NET": NET_COLS}, open(C.META_PATH, "wb"))
    print(f"saved {C.FULL_DF_PATH} + {C.META_PATH}", flush=True)
    return full_df


if __name__ == "__main__":
    build()
