# Re-fine-tune CT-BERT + attention-MIL on the vote20 target (faithful "same setup").
# CORPUS=authored  -> respondents' own covid/vaccine-keyword tweets
# CORPUS=feed      -> simfeed tweets they saw (covid/vaccine-keyword set)
# Leakage-safe: 5-fold split BY PERSON; train on train-fold people; out-of-fold per-person
# pooled embedding + P(definite voter). Writes to the config.CONTENT[...] paths.
# PREREQS: run dataprep.py first (needs cand20_full_df.pkl); GPU torch/transformers env.
#   usage:  CORPUS=authored python finetune_mil_vote20.py   (then CORPUS=feed ...)
import os, re, time, pickle, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import config as C

CORPUS = os.environ.get("CORPUS", "authored")   # "authored" | "feed"
assert CORPUS in ("authored", "feed")
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_TRAIN_LAYERS, TRAIN_CAP, SCORE_CAP = 3, 32, 64
PBATCH, MAXLEN, LR, EPOCHS_MAX, PATIENCE, FOLDS, SEED, ATTN_DIM = 8, 96, 2e-5, 6, 2, 5, 42, 128
if SMOKE: FOLDS, EPOCHS_MAX, TRAIN_CAP, SCORE_CAP, PBATCH = 2, 2, 12, 20, 6
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
print(f"device:{dev} CORPUS:{CORPUS} SMOKE:{SMOKE} BASE:{C.BASE_MODEL}", flush=True)

full = pd.read_pickle(os.path.join(C.CAND20_DIR, "cand20_full_df.pkl"))
full["uid_str"] = full["uid_str"].astype(str)
lab = dict(zip(full["uid_str"], full["y"].astype(int)))
CANON = set(full["uid_str"])
print(f"people: {len(full)}  pos-rate {full['y'].mean():.3f}", flush=True)

# ---- per-respondent PRE-EVENT cutoff: tweets must predate the respondent's answer date ----
# Twitter IDs are snowflake-timestamped; convert each cutoff_date to an ID floor and keep only
# tweets with id < floor (conservative: excludes the answer day itself). No datetime decode needed.
import datetime as _dt
TW_EPOCH = 1288834974657
def _sf_floor(date_str):
    y, m, d = (int(x) for x in str(date_str)[:10].split("-"))
    ts_ms = int(_dt.datetime(y, m, d, tzinfo=_dt.timezone.utc).timestamp() * 1000)
    return (ts_ms - TW_EPOCH) << 22   # smallest tweet id created at/after midnight UTC of that date
_FIXED = getattr(C, "FIXED_CUTOFF", None)
_FLOOR = getattr(C, "FLOOR_CUTOFF", None)
def _eff_cutoff(ans):
    if _FIXED: return _FIXED
    opts = [x for x in [(str(ans)[:10] if pd.notna(ans) else None), _FLOOR] if x]
    return max(opts) if opts else None   # ISO date strings sort chronologically
CUT = {u: (_sf_floor(e) if e else None)
       for u, e in ((uu, _eff_cutoff(cc)) for uu, cc in zip(full["uid_str"], full[C.CUTOFF_COL]))}
if _FIXED:   print(f"cutoff: uniform {_FIXED}", flush=True)
elif _FLOOR: print(f"cutoff: max(latest answer date, {_FLOOR}) per respondent", flush=True)
else:        print("cutoff: per-respondent answer date", flush=True)
def _before_cutoff(uid, tid):
    thr = CUT.get(uid)
    if thr is None: return True
    try: return int(tid) < thr
    except Exception: return False

# ---- build per-person tweet bags (election keyword-filtered; trimmed to pre-event tweets) ----
PT = {}; dropped_late = 0
if CORPUS == "authored":
    au = pd.read_parquet(C.AUTHORED_PARQUET, columns=["id_str", "userid", "eff_text"])
    au["userid"] = au["userid"].astype(str); au = au[au["userid"].isin(CANON)]
    au["eff_text"] = au["eff_text"].astype(str)
    au = au[au["eff_text"].str.lower().str.contains(C.KWRE, na=False)]
    for tid, u, t in zip(au["id_str"].values, au["userid"].values, au["eff_text"].values):
        if _before_cutoff(u, tid): PT.setdefault(u, []).append(t)
        else: dropped_late += 1
else:
    TXT = pickle.load(open(C.EL_SIMFEED_TEXT, "rb"))     # election simfeed tweet_id -> text
    u2t = pickle.load(open(C.EL_UID_TO_TWEETS, "rb"))    # uid_str -> [tweet_id]
    for u in CANON:
        f = [TXT[t] for t in u2t.get(u, []) if t in TXT and _before_cutoff(u, t)]
        dropped_late += sum(1 for t in u2t.get(u, []) if t in TXT and not _before_cutoff(u, t))
        if f: PT[u] = f
print(f"people with {CORPUS} tweets: {len(PT)} | pre-event tweets kept, {dropped_late:,} post-cutoff dropped ({time.time()-t0:.0f}s)", flush=True)

tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
class AttnMIL(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = AutoModel.from_pretrained(C.BASE_MODEL); H = self.enc.config.hidden_size
        for p in self.enc.parameters(): p.requires_grad = False
        for lyr in self.enc.encoder.layer[-N_TRAIN_LAYERS:]:
            for p in lyr.parameters(): p.requires_grad = True
        self.V = nn.Linear(H, ATTN_DIM); self.U = nn.Linear(H, ATTN_DIM); self.w = nn.Linear(ATTN_DIM, 1)
        self.drop = nn.Dropout(0.1); self.head = nn.Linear(H, 1); self.H = H
    def tweet_emb(self, texts):
        out = []
        for s in range(0, len(texts), 128):
            enc = tok(texts[s:s+128], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            h = self.enc(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1)
            out.append((h*m).sum(1)/m.sum(1).clamp(min=1))
        return torch.cat(out, 0)
    def pool_bag(self, hb):
        a = torch.softmax(self.w(torch.tanh(self.V(hb))*torch.sigmoid(self.U(hb))).squeeze(-1), 0)
        return (a.unsqueeze(-1)*hb).sum(0)
    def forward_people(self, bags):
        flat = [t for bag in bags for t in bag]; owner = []
        for i, bag in enumerate(bags): owner += [i]*len(bag)
        H = self.tweet_emb(flat); owner = torch.tensor(owner, device=dev)
        Z = torch.stack([self.pool_bag(H[owner == i]) for i in range(len(bags))], 0)
        return self.head(self.drop(Z)).squeeze(-1), Z

def build_bags(uids, cap):
    bags, keep = [], []
    for u in uids:
        f = PT.get(u, [])
        if not f: continue
        if len(f) > cap: f = random.sample(f, cap)
        bags.append(f); keep.append(u)
    return keep, bags
@torch.no_grad()
def eval_people(model, uids, cap):
    model.eval(); probs, embs = {}, {}
    keep, bags = build_bags(uids, cap)
    for s in range(0, len(keep), PBATCH):
        lg, Z = model.forward_people(bags[s:s+PBATCH])
        p = torch.sigmoid(lg).float().cpu().numpy(); z = Z.float().cpu().numpy()
        for j, u in enumerate(keep[s:s+PBATCH]): probs[u] = float(p[j]); embs[u] = z[j]
    return probs, embs
def auc_safe(y, p): y = np.asarray(y); return roc_auc_score(y, p) if len(set(y)) > 1 and len(y) > 3 else float("nan")

def train_fold(tr_uids, fold):
    tr_uids = [u for u in tr_uids if PT.get(u)]; rng = np.random.RandomState(SEED+fold); rng.shuffle(tr_uids)
    nval = max(PBATCH, int(0.15*len(tr_uids))); val, trn = tr_uids[:nval], tr_uids[nval:]
    ytr = np.array([lab[u] for u in trn]); posw = torch.tensor([(len(ytr)-ytr.sum())/max(ytr.sum(), 1)], dtype=torch.float, device=dev)
    model = AttnMIL().to(dev); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw); best_auc, best_state, bad = -1, None, 0
    for ep in range(EPOCHS_MAX):
        model.train(); order = list(range(len(trn))); random.shuffle(order); tl = nb = 0
        for s in range(0, len(order), PBATCH):
            us = [trn[i] for i in order[s:s+PBATCH]]; _, bags = build_bags(us, TRAIN_CAP)
            if len(bags) < 2: continue
            y = torch.tensor([lab[u] for u in us if PT.get(u)], dtype=torch.float, device=dev)
            lg, _ = model.forward_people(bags); loss = lossf(lg, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); tl += loss.item(); nb += 1
            if SMOKE and nb >= 4: break
        vp, _ = eval_people(model, val, SCORE_CAP); vauc = auc_safe([lab[u] for u in vp], [vp[u] for u in vp])
        print(f"  fold{fold} ep{ep+1}: loss {tl/max(nb,1):.4f} HELDOUT_auc {vauc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vauc > best_auc: best_auc, best_state, bad = vauc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    return model

uids = full["uid_str"].values; y = full["y"].astype(int).values
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_p, oof_e = {}, {}
for k, (tr, te) in enumerate(skf.split(uids, y)):
    print(f"\n=== FOLD {k+1}/{FOLDS} ===", flush=True)
    m = train_fold(list(uids[tr]), k); pp, ee = eval_people(m, list(uids[te]), SCORE_CAP)
    oof_p.update(pp); oof_e.update(ee); del m; torch.cuda.empty_cache()
    if SMOKE: break

H = len(next(iter(oof_e.values())))
res = pd.DataFrame([{"uid_str": u, "ft_score": oof_p[u]} for u in oof_p])
emb = pd.DataFrame([{"uid_str": u, **{f"e{j}": float(oof_e[u][j]) for j in range(H)}} for u in oof_e])
if not SMOKE:
    res.to_pickle(C.CONTENT[CORPUS]["oof"]); emb.to_pickle(C.CONTENT[CORPUS]["emb"])
    print(f"saved {C.CONTENT[CORPUS]['oof']} + emb (dim {H}, {len(emb)} people)", flush=True)
ev = res.merge(full[["uid_str", "y"]], on="uid_str")
print(f"\nSTANDALONE {CORPUS} head-prob vs vote20-positive: AUC={auc_safe(ev['y'], ev['ft_score']):.3f}", flush=True)
print("DONE", flush=True)
