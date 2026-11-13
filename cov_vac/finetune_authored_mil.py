# Attention-MIL fine-tuning: predict REFUSER (cov_vac==1) from a person's AUTHORED covid/vaccine
# tweets, trained END-TO-END at the PERSON level (bag of tweets -> gated-attention pool -> head).
# Goal: produce a task-shaped per-person EMBEDDING (the attention-pooled vector), not just a score.
#
# Leakage-safe: 5-fold split BY PERSON; train on train-fold people only; out-of-fold person
# embedding + head prob. Anti-overfit: freeze embeddings + all but top N_TRAIN_LAYERS encoder
# layers; within-fold val split for early stopping. Base model swappable (twitter-roberta-base
# default; set BASE to a covid model like digitalepidemiologylab/covid-twitter-bert-v2 to try CT-BERT).
import os, time, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import config as C

SMOKE = os.environ.get("SMOKE","0")=="1"
BASE  = os.environ.get("BASE", C.BASE_MODEL)   # CT-BERT (the committed config); override via BASE env
KWRE  = C.KWRE
N_TRAIN_LAYERS = 3          # unfreeze top-k encoder layers (+ attn pool + head); rest frozen
TRAIN_CAP, SCORE_CAP = 32, 64   # tweets/person for train (sampled) / scoring
PBATCH, MAXLEN, LR, EPOCHS_MAX, PATIENCE, FOLDS, SEED = 8, 96, 2e-5, 6, 2, 5, 42
ATTN_DIM = 128
if SMOKE: FOLDS, EPOCHS_MAX, TRAIN_CAP, SCORE_CAP, PBATCH = 2, 2, 12, 20, 6
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device:{dev} SMOKE:{SMOKE} BASE:{BASE}", flush=True); t0=time.time()

# ---------- person table (uid_str, refuser, party, gender) from dataprep's full_df ----------
demo=pd.read_pickle(C.FULL_DF_PATH); demo["uid_str"]=demo["uid_str"].astype(str)
demo["refuser"]=(demo["y"].astype(int)==1).astype(int)
demo=demo[["uid_str","refuser","party","gender"]].dropna(subset=["uid_str"]).reset_index(drop=True)
CANON=set(demo["uid_str"])
print(f"people:{len(demo)} refusers:{demo['refuser'].sum()} ({demo['refuser'].mean():.3f})", flush=True)

# ---------- authored covid/vaccine tweets per person ----------
au=pd.read_parquet(C.AUTHORED_PARQUET, columns=["userid","eff_text"])
au["userid"]=au["userid"].astype(str); au=au[au["userid"].isin(CANON)]
au["eff_text"]=au["eff_text"].astype(str)
au=au[au["eff_text"].str.lower().str.contains(KWRE,na=False)]
PT={}
for u,t in zip(au["userid"].values, au["eff_text"].values): PT.setdefault(u,[]).append(t)
lab=dict(zip(demo["uid_str"],demo["refuser"]))
print(f"authored kw tweets:{len(au):,} from {len(PT):,} people ({time.time()-t0:.0f}s)", flush=True)

tok=AutoTokenizer.from_pretrained(BASE)

# ---------- model: encoder (partly frozen) + gated-attention MIL pool + head ----------
class AttnMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=AutoModel.from_pretrained(BASE)
        H=self.enc.config.hidden_size
        # freeze everything, then unfreeze top N encoder layers
        for p in self.enc.parameters(): p.requires_grad=False
        layers=self.enc.encoder.layer
        for lyr in layers[-N_TRAIN_LAYERS:]:
            for p in lyr.parameters(): p.requires_grad=True
        self.V=nn.Linear(H,ATTN_DIM); self.U=nn.Linear(H,ATTN_DIM); self.w=nn.Linear(ATTN_DIM,1)
        self.drop=nn.Dropout(0.1); self.head=nn.Linear(H,1); self.H=H
    def tweet_emb(self, texts):
        # masked mean-pool of last hidden state -> (n_tweets, H)
        out=[]
        for s in range(0,len(texts),128):
            enc=tok(texts[s:s+128],truncation=True,max_length=MAXLEN,padding=True,return_tensors="pt").to(dev)
            h=self.enc(**enc).last_hidden_state; m=enc["attention_mask"].unsqueeze(-1)
            out.append((h*m).sum(1)/m.sum(1).clamp(min=1))
        return torch.cat(out,0)                                  # (S,H)
    def pool_bag(self, hb):
        a=self.w(torch.tanh(self.V(hb))*torch.sigmoid(self.U(hb))).squeeze(-1)   # (n,)
        a=torch.softmax(a,0)
        return (a.unsqueeze(-1)*hb).sum(0)                        # (H,)
    def forward_people(self, bags):
        # bags: list of lists of tweet strings; returns logits (B,), person embeddings (B,H)
        flat=[t for bag in bags for t in bag]; owner=[]
        for i,bag in enumerate(bags): owner += [i]*len(bag)
        H=self.tweet_emb(flat); owner=torch.tensor(owner,device=dev)
        Z=torch.stack([self.pool_bag(H[owner==i]) for i in range(len(bags))],0)   # (B,H)
        return self.head(self.drop(Z)).squeeze(-1), Z

def build_bags(uids, cap):
    bags=[]; keep=[]
    for u in uids:
        f=PT.get(u,[])
        if not f: continue
        if len(f)>cap: f=random.sample(f,cap)
        bags.append(f); keep.append(u)
    return keep, bags

@torch.no_grad()
def eval_people(model, uids, cap):
    model.eval(); probs={}; embs={}
    keep,bags=build_bags(uids,cap)
    for s in range(0,len(keep),PBATCH):
        bb=bags[s:s+PBATCH]; lg,Z=model.forward_people(bb)
        p=torch.sigmoid(lg).float().cpu().numpy(); z=Z.float().cpu().numpy()
        for j,u in enumerate(keep[s:s+PBATCH]): probs[u]=float(p[j]); embs[u]=z[j]
    return probs, embs

def auc_safe(y,p):
    y=np.asarray(y);
    return roc_auc_score(y,p) if len(set(y))>1 and len(y)>3 else float("nan")

def train_fold(tr_uids, fold):
    # internal person-level val split for early stopping
    tr_uids=[u for u in tr_uids if PT.get(u)]
    rng=np.random.RandomState(SEED+fold); rng.shuffle(tr_uids)
    nval=max(PBATCH, int(0.15*len(tr_uids))); val=tr_uids[:nval]; trn=tr_uids[nval:]
    ytr=np.array([lab[u] for u in trn]); posw=torch.tensor([(len(ytr)-ytr.sum())/max(ytr.sum(),1)],dtype=torch.float,device=dev)
    model=AttnMIL().to(dev)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    lossf=nn.BCEWithLogitsLoss(pos_weight=posw)
    best_auc=-1; best_state=None; bad=0
    for ep in range(EPOCHS_MAX):
        model.train(); order=list(range(len(trn))); random.shuffle(order); tl=0; nb=0
        for s in range(0,len(order),PBATCH):
            idx=order[s:s+PBATCH]; us=[trn[i] for i in idx]
            _,bags=build_bags(us,TRAIN_CAP)
            if len(bags)<2: continue
            y=torch.tensor([lab[u] for u in us if PT.get(u)],dtype=torch.float,device=dev)
            lg,_=model.forward_people(bags); loss=lossf(lg,y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step(); tl+=loss.item(); nb+=1
            if SMOKE and nb>=4: break
        vp,_=eval_people(model, val, SCORE_CAP)
        vauc=auc_safe([lab[u] for u in vp], [vp[u] for u in vp])
        tp,_=eval_people(model, trn[:min(len(trn),400)], SCORE_CAP)
        tauc=auc_safe([lab[u] for u in tp],[tp[u] for u in tp])
        print(f"  fold{fold} ep{ep+1}: train_loss {tl/max(nb,1):.4f} | TRAIN_auc {tauc:.3f} HELDOUT_auc {vauc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vauc>best_auc: best_auc=vauc; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=PATIENCE: print(f"  early stop (fold{fold})",flush=True); break
    if best_state: model.load_state_dict(best_state)
    return model

uids=demo["uid_str"].values; y=demo["refuser"].values
skf=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
oof_p={}; oof_e={}
for k,(tr,te) in enumerate(skf.split(uids,y)):
    print(f"\n=== FOLD {k+1}/{FOLDS} === train {len(tr)} test {len(te)} ({time.time()-t0:.0f}s)", flush=True)
    m=train_fold(list(uids[tr]), k)
    pp,ee=eval_people(m, list(uids[te]), SCORE_CAP)
    oof_p.update(pp); oof_e.update(ee)
    del m; torch.cuda.empty_cache()
    if SMOKE: break

H=len(next(iter(oof_e.values())))
res=pd.DataFrame([{"uid_str":u,"ft_score":oof_p[u],"n_scored":len(PT.get(u,[]))} for u in oof_p]).merge(demo,on="uid_str",how="left")
emb=pd.DataFrame([{"uid_str":u, **{f"e{j}":float(oof_e[u][j]) for j in range(H)}} for u in oof_e])
if not SMOKE:
    res.to_pickle(C.CONTENT["authored"]["oof"]); emb.to_pickle(C.CONTENT["authored"]["emb"])
    print(f"saved OOF scores ({len(res)}) + person embeddings ({len(emb)}, dim {H})", flush=True)

# ---------- quick STANDALONE (content-only) eval — the trace-only regime ----------
ev=res.dropna(subset=["ft_score"])
print(f"\nscored {len(ev)}/{len(demo)} people with authored content", flush=True)
def rep(name,mask):
    d=ev[mask]
    if d["refuser"].sum()<10: return
    print(f"  {name:18s} n={len(d):5d} prev={d['refuser'].mean():.3f}  AUC={auc_safe(d['refuser'],d['ft_score']):.3f}  PR={average_precision_score(d['refuser'],d['ft_score']):.3f}", flush=True)
print("STANDALONE MIL head-prob vs refuser (content only, no demographics):")
rep("Overall", np.ones(len(ev),bool))
for p in ["Democrat","Republican","Independent/Other"]: rep(p,(ev["party"]==p).values)
for g in ["Male","Female"]: rep(g,(ev["gender"]==g).values)
print("DONE", flush=True)
