# Build the ELECTION simfeed from the decahose text source (decahose_samples/*.json.gz):
# for each modelable respondent, the election-relevant tweets authored by accounts they follow.
# Outputs: election_simfeed_text.pkl (tweet_id -> text) and uid_to_election_simfeed_tweet_ids.pkl.
# CPU, streams all daily decahose files (~30 min). Run after dataprep.py.
import os, gzip, json, glob, pickle, time, pandas as pd
from collections import defaultdict
import config as C

t0 = time.time()
full = pd.read_pickle(os.path.join(C.VOTE20_DIR, "vote20_full_df.pkl"))
uids = set(full["uid_str"].astype(str))
foll = pickle.load(open(C.FOLLOW_PATH, "rb"))
# inverted index: followee-author -> modelable respondents who follow them
author2resp = defaultdict(list)
for u in uids:
    for f in foll.get(u, []): author2resp[str(f)].append(u)
feed_authors = set(author2resp)
print(f"modelable respondents: {len(uids)} | followee-authors: {len(feed_authors):,}", flush=True)

TXT = {}; uid2t = defaultdict(list)
days = sorted(glob.glob(os.path.join(C.DECAHOSE_DIR, "*.json.gz")))
print(f"streaming {len(days)} decahose day-files...", flush=True)
for i, day in enumerate(days, 1):
    kept = 0
    try:
        with gzip.open(day, "rt") as f:
            for line in f:
                try: d = json.loads(line)
                except Exception: continue
                au = str((d.get("user") or {}).get("id_str") or "")
                if au not in feed_authors: continue
                tw = (d.get("extended_tweet") or {}).get("full_text") or d.get("text") or ""
                if not C.KWRE.search(tw): continue
                tid = str(d.get("id_str") or "")
                if not tid: continue
                TXT[tid] = tw
                for r in author2resp[au]: uid2t[r].append(tid)
                kept += 1
    except Exception as e:
        print(f"  WARN {os.path.basename(day)}: {e}", flush=True); continue
    if i % 20 == 0 or i == len(days):
        print(f"  [{i}/{len(days)}] {os.path.basename(day)} kept {kept} | total tweets {len(TXT):,} ({time.time()-t0:.0f}s)", flush=True)

pickle.dump(TXT, open(C.EL_SIMFEED_TEXT, "wb"))
pickle.dump({u: v for u, v in uid2t.items()}, open(C.EL_UID_TO_TWEETS, "wb"))
cov = sum(1 for u in uids if uid2t.get(u))
print(f"\nelection simfeed: {len(TXT):,} unique tweets | respondents with >=1 feed tweet: {cov}/{len(uids)}", flush=True)
import numpy as np
sizes = np.array([len(uid2t.get(u, [])) for u in uids])
print(f"per-respondent feed size — median {int(np.median(sizes))}, mean {int(sizes.mean())}, p90 {int(np.percentile(sizes,90))}", flush=True)
print("saved election_simfeed_text.pkl + uid_to_election_simfeed_tweet_ids.pkl", flush=True)
print("DONE", flush=True)
