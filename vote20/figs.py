# vote20 final figures (NO titles) from vote20_grid.pkl. Per demographic dimension:
#   1_nodemo_by_<dim>.png : Network/Authored/Feed + best-trace-combo bars; demographics = baseline line.
#   2_lift_by_<dim>.png    : Delta ROC-AUC each signal (and best combo) adds over demographics (95% CI).
# Mirrors make_presentation_figs.py. Positive class = "definite voter" (vote20==1).
import os, pickle, numpy as np
import config as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#888","axes.linewidth":0.8,"figure.dpi":200})
BLUE="#0072B2"; ORANGE="#E69F00"; PURP="#CC79A7"; GREEN="#009E73"; INK="#222"
RES=pickle.load(open(os.path.join(C.VOTE20_DIR,"vote20_grid.pkl"),"rb"))["RES"]
have=lambda sub,c:(sub,c) in RES
def roc(sub,c): return RES[(sub,c)]["roc"].mean()
def paired(sub,c,base="Demo"):
    d=RES[(sub,c)]["roc"]-RES[(sub,base)]["roc"]; se=d.std(ddof=1)/np.sqrt(len(d))
    return d.mean(),1.96*se,(d.mean()-1.96*se>0 or d.mean()+1.96*se<0)
YL=f"ROC-AUC (predicting {C.POS_LABEL})"
DIMS={"gender":[("Overall","Overall"),("Male","Male"),("Female","Female")],
      "party":[("Overall","Overall"),("Democrat","Democrat"),("Republican","Republican"),("Independent/Other","Indep/Other")],
      "age":[("Overall","Overall"),("Age 18-24","18-24"),("Age 25-44","25-44"),("Age 45-64","45-64")]}
TRACE=["Net","Auth","Feed","Auth+Net","Feed+Net","Auth+Feed","Auth+Feed+Net"]
ADDON=["Demo+Net","Demo+Auth","Demo+Feed","Demo+Auth+Net","Demo+Feed+Net","Demo+Auth+Feed","Demo+Auth+Feed+Net"]
ABBR={"Net":"N","Auth":"A","Feed":"F"}
def present(subs): return [(s,l) for s,l in subs if have(s,"Demo")]   # drop subgroups the grid skipped

def fig_nodemo(dim,subs):
    subs=present(subs)
    if not subs: return
    fixed=[("Network","Net",BLUE),("Authored tweets","Auth",ORANGE),("Feed tweets","Feed",PURP)]
    x=np.arange(len(subs)); w=0.2; fig,ax=plt.subplots(figsize=(1.9*len(subs)+2.5,5.2))
    for i,(nm,cb,cl) in enumerate(fixed):
        v=[roc(s,cb) for s,_ in subs]; b=ax.bar(x+(i-1.5)*w,v,w,color=cl,label=nm,zorder=3)
        for bar in b: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.004,f"{bar.get_height():.2f}",ha="center",va="bottom",fontsize=7,color="#444")
    bvals=[];blab=[]
    for s,_ in subs:
        vv={c:roc(s,c) for c in TRACE}; bc=max(vv,key=vv.get); bvals.append(vv[bc]); blab.append("+".join(ABBR[p] for p in bc.split("+")))
    bb=ax.bar(x+1.5*w,bvals,w,color=GREEN,label="Best trace combo",zorder=3)
    for bar,vv,lb in zip(bb,bvals,blab):
        ax.text(bar.get_x()+bar.get_width()/2,vv+0.004,f"{vv:.2f}",ha="center",va="bottom",fontsize=7,color="#444")
        ax.text(bar.get_x()+bar.get_width()/2,vv+0.024,lb,ha="center",va="bottom",fontsize=6.5,color=GREEN,fontstyle="italic",fontweight="bold")
    for j,(s,_) in enumerate(subs):
        dv=roc(s,"Demo"); ax.plot([j-0.44,j+0.44],[dv,dv],color=INK,ls="--",lw=2,zorder=5)
        ax.text(j,dv+0.006,f"Demo {dv:.2f}",ha="center",va="bottom",fontsize=7.5,color=INK,fontweight="bold")
    ax.axhline(0.5,color="#aaa",ls=":",lw=1.1,zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in subs]); ax.set_ylim(0.5,0.82); ax.set_ylabel(YL)
    ax.grid(axis="y",ls="--",alpha=0.3,zorder=0)
    h=[Line2D([0],[0],color=INK,ls="--",lw=2)]+[plt.Rectangle((0,0),1,1,color=c) for c in (BLUE,ORANGE,PURP,GREEN)]
    ax.legend(h,["Demographics (baseline)","Network","Authored tweets","Feed tweets","Best trace combo (N=Net A=Auth F=Feed)"],
              ncol=3,frameon=False,fontsize=9,loc="upper center",bbox_to_anchor=(0.5,1.16))
    fig.tight_layout(); fn=os.path.join(C.FIG_DIR,f"1_nodemo_by_{dim}.png"); fig.savefig(fn,bbox_inches="tight"); print("saved",fn,flush=True); plt.close(fig)

def fig_lift(dim,subs):
    subs=present(subs)
    if not subs: return
    single=[("Network","Demo+Net",BLUE),("Authored tweets","Demo+Auth",ORANGE),("Feed tweets","Demo+Feed",PURP)]
    x=np.arange(len(subs)); w=0.2; fig,ax=plt.subplots(figsize=(1.9*len(subs)+2.5,5.2))
    for i,(nm,cb,cl) in enumerate(single):
        vals=[];errs=[];sig=[]
        for s,_ in subs: dm,er,ss=paired(s,cb); vals.append(dm);errs.append(er);sig.append(ss)
        b=ax.bar(x+(i-1.5)*w,vals,w,color=cl,label=nm,zorder=3,yerr=errs,capsize=2,error_kw={"lw":0.8,"ecolor":"#666"})
        for bar,ss in zip(b,sig):
            if ss: h=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,h+(0.0008 if h>=0 else -0.003),"*",ha="center",va="bottom" if h>=0 else "top",fontsize=11,color=INK)
    bvals=[];berrs=[];bsig=[];blab=[]
    for s,_ in subs:
        best=None
        for cb in ADDON:
            dm,er,ss=paired(s,cb)
            if best is None or dm>best[1]: best=(cb,dm,er,ss)
        bvals.append(best[1]);berrs.append(best[2]);bsig.append(best[3]); blab.append("+".join(ABBR[p] for p in best[0].replace("Demo+","").split("+")))
    b=ax.bar(x+1.5*w,bvals,w,color=GREEN,label="Best combo added",zorder=3,yerr=berrs,capsize=2,error_kw={"lw":0.8,"ecolor":"#666"})
    for bar,ss,lb,be in zip(b,bsig,blab,berrs):
        h=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,h+be+0.004,lb,ha="center",va="bottom",fontsize=6.5,color=GREEN,fontstyle="italic",fontweight="bold")
        if ss: ax.text(bar.get_x()+bar.get_width()/2,h+(0.0008 if h>=0 else -0.003),"*",ha="center",va="bottom" if h>=0 else "top",fontsize=11,color=INK)
    ax.axhline(0,color=INK,lw=1,zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in subs]); ax.set_ylabel("Δ ROC-AUC vs demographics")
    ax.grid(axis="y",ls="--",alpha=0.3,zorder=0)
    ax.legend(ncol=4,frameon=False,fontsize=9,loc="upper center",bbox_to_anchor=(0.5,1.10))
    fig.tight_layout(); fn=os.path.join(C.FIG_DIR,f"2_lift_by_{dim}.png"); fig.savefig(fn,bbox_inches="tight"); print("saved",fn,flush=True); plt.close(fig)

for dim,subs in DIMS.items():
    fig_nodemo(dim,subs); fig_lift(dim,subs)
print("DONE",flush=True)
