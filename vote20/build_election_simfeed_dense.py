# Build the ELECTION simfeed from the DENSE full-decahose gnip.lz4 files
# (dense daily decahose gnip.lz4 archives, ~4.25 GB/day; see config_paths.DENSE_DECAHOSE_DIR).
# For each modelable respondent: the election-relevant tweets authored by accounts they follow,
# over the pre-election run-up window (config.DENSE_WINDOW, default 2020-09-01..2020-11-03).
# Format: lz4-compressed gnip TSV; fields id=0, author_uid=1, text=9 (verified on 2020-11-03).
# Outputs the SAME paths as the sparse builder, so finetune/grid pick them up unchanged:
#   election_simfeed_text.pkl (tweet_id -> text) and uid_to_election_simfeed_tweet_ids.pkl.
# CPU, streams each day with lz4.frame (no lz4 CLI needed). Run after dataprep.py.
import os, glob, pickle, time, datetime as dt, numpy as np, pandas as pd, lz4.frame
from collections import defaultdict
import config as C

t0 = time.time()
TID, AUID, TEXT = C.GNIP_COL_TID, C.GNIP_COL_AUID, C.GNIP_COL_TEXT
NCOL = max(TID, AUID, TEXT) + 1

# ---- respondents + inverted index (followee-author -> respondents who follow them) ----
full = pd.read_pickle(os.path.join(C.VOTE20_DIR, "vote20_full_df.pkl"))
uids = set(full["uid_str"].astype(str))
foll = pickle.load(open(C.FOLLOW_PATH, "rb"))
author2resp = defaultdict(list)
for u in uids:
    for f in foll.get(u, []): author2resp[str(f)].append(u)
feed_authors = set(author2resp)
print(f"modelable respondents: {len(uids)} | followee-authors: {len(feed_authors):,}", flush=True)

# ---- select the day-files inside the window ----
d0 = dt.date.fromisoformat(C.DENSE_WINDOW[0]); d1 = dt.date.fromisoformat(C.DENSE_WINDOW[1])
days = []
for p in sorted(glob.glob(os.path.join(C.DENSE_DECAHOSE_DIR, "tweets.*.gnip.lz4"))):
    try: d = dt.date.fromisoformat(os.path.basename(p).split(".")[1])
    except Exception: continue
    if d0 <= d <= d1: days.append((d, p))
days.sort()
print(f"window {C.DENSE_WINDOW[0]}..{C.DENSE_WINDOW[1]}: {len(days)} day-files", flush=True)
assert days, "no gnip.lz4 files in the window"

def stream_lines(path, chunk=1 << 22):
    """Yield decoded text lines from an lz4-framed file without loading it whole."""
    dctx = lz4.frame.LZ4FrameDecompressor(); tail = b""
    with open(path, "rb") as fh:
        while True:
            comp = fh.read(chunk)
            if not comp:
                if tail: yield tail.decode("utf-8", "replace")
                break
            tail += dctx.decompress(comp)
            parts = tail.split(b"\n"); tail = parts.pop()
            for ln in parts: yield ln.decode("utf-8", "replace")

TXT = {}; uid2t = defaultdict(list)
for i, (d, path) in enumerate(days, 1):
    kept = seen = 0
    try:
        for line in stream_lines(path):
            seen += 1
            f = line.split("\t")
            if len(f) < NCOL: continue
            au = f[AUID]
            if au not in feed_authors: continue
            tw = f[TEXT]
            if not C.KWRE.search(tw): continue
            tid = f[TID]
            if not tid: continue
            TXT[tid] = tw
            for r in author2resp[au]: uid2t[r].append(tid)
            kept += 1
    except Exception as e:
        print(f"  WARN {os.path.basename(path)}: {e}", flush=True); continue
    print(f"  [{i}/{len(days)}] {d} lines {seen:,} kept {kept:,} | total {len(TXT):,} ({time.time()-t0:.0f}s)", flush=True)

pickle.dump(TXT, open(C.EL_SIMFEED_TEXT, "wb"))
pickle.dump({u: v for u, v in uid2t.items()}, open(C.EL_UID_TO_TWEETS, "wb"))
cov = sum(1 for u in uids if uid2t.get(u))
sizes = np.array([len(uid2t.get(u, [])) for u in uids])
print(f"\nDENSE election simfeed: {len(TXT):,} unique tweets | respondents with >=1 feed tweet: {cov}/{len(uids)} ({cov/len(uids):.1%})", flush=True)
print(f"per-respondent feed size — median {int(np.median(sizes))}, mean {sizes.mean():.1f}, p90 {int(np.percentile(sizes,90))}", flush=True)
print("saved election_simfeed_text.pkl + uid_to_election_simfeed_tweet_ids.pkl", flush=True)
print("DONE", flush=True)
