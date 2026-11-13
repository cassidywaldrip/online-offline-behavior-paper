# Build per-outcome demographic caches for the sample-characteristics table (Table 8).
# For each outcome we take the EXACT modeled population (the uid list in
# paper_assets/signal_cache/<out>_signal.csv, which matches the paper's reported N) and
# record each respondent's demographics AS THE MODEL SAW THEM, decoded to human labels.
#   every outcome : read <out>/<out>_full_df.pkl (written by that outcome's dataprep.py;
#                   already one row per modeled respondent).
# Writes paper_assets/demo_cache/<out>_demo.csv with columns:
#   uid_str,y,party,gender,age,education,race,employ,income
import os, sys, glob, json, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXPO = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "demo_cache"); os.makedirs(CACHE, exist_ok=True)
SIG   = os.path.join(HERE, "signal_cache")

# ---- decode maps: ordinal int -> human label (inverse of cov_vac/dataprep.py orders) --------
AGE_LAB   = {0:"18--24", 1:"25--44", 2:"45--64", 3:"65+"}
EDUC_LAB  = {0:"High school or less", 1:"Some college", 2:"College degree", 3:"Graduate degree"}
INCOME_LAB= {0:"<\\$15k", 1:"\\$15--25k", 2:"\\$25--35k", 3:"\\$35--50k",
             4:"\\$50--75k", 5:"\\$75--100k", 6:"\\$100--150k", 7:"\\$150k+"}

def decode(df):
    """df has model-encoded demo cols; return a frame of decoded human labels."""
    out = pd.DataFrame({"uid_str": df["uid_str"].astype(str).values})
    out["y"]      = df["y"].astype(int).values
    out["party"]  = df["party"].astype(str).values
    out["gender"] = df["gender"].astype(str).values
    out["race"]   = df["race"].astype(str).values
    out["employ"] = df["employ"].astype(str).values
    out["age"]       = pd.to_numeric(df["age_cat_4"], errors="coerce").map(AGE_LAB).values
    out["education"] = pd.to_numeric(df["education_cat"], errors="coerce").map(EDUC_LAB).values
    out["income"]    = pd.to_numeric(df["income_cat_10"], errors="coerce").map(INCOME_LAB).values
    return out

def modeled_uids(out):
    return set(pd.read_csv(os.path.join(SIG, f"{out}_signal.csv"), dtype={"uid_str":str})["uid_str"])

def build_simple(out):
    df = pd.read_pickle(os.path.join(EXPO, out, f"{out}_full_df.pkl"))
    df["uid_str"] = df["uid_str"].astype(str)
    keep = modeled_uids(out)
    df = df[df["uid_str"].isin(keep)].drop_duplicates("uid_str").reset_index(drop=True)
    return decode(df)

def main():
    for out in ["cand20", "mail_nov", "vote20", "cov_vac"]:
        d = build_simple(out)
        d.to_csv(os.path.join(CACHE, f"{out}_demo.csv"), index=False)
        print(f"[{out}] n={len(d)}  party={d['party'].value_counts().to_dict()}", flush=True)

if __name__ == "__main__":
    main()
