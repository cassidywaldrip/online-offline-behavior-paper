# Paper-ready figures + tables for the Network Science submission.
# Spine: ATTRIBUTE baselines (demographics) predict the modal partisan;
#        RELATIONAL position (network audience-lean) predicts the within-party deviant.
# Reads the four 3-demo results CSVs (+ the surviving 7-demo cov_vac CSV for the caveat table),
# writes static PDF+PNG figures and LaTeX+CSV tables into paper_assets/figs and paper_assets/tables.
import os, csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE   = os.path.dirname(os.path.abspath(__file__))
EXPO   = os.path.dirname(HERE)
FIGDIR = os.path.join(HERE, "figs");   os.makedirs(FIGDIR, exist_ok=True)
TABDIR = os.path.join(HERE, "tables"); os.makedirs(TABDIR, exist_ok=True)

# Okabe-Ito (colorblind-safe, publication standard)
OI = {"orange":"#E69F00","skyblue":"#56B4E9","green":"#009E73","yellow":"#F0E442",
      "blue":"#0072B2","vermillion":"#D55E00","purple":"#CC79A7","gray":"#7F7F7F"}
BASE_C = "#9A9A9A"   # attribute baseline (Demo)
BEST_C = OI["blue"]  # relational / best model
BLOCK_C = {"Demo":"#9A9A9A", "Net":OI["blue"], "Feed":OI["green"], "Auth":OI["purple"]}

plt.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["DejaVu Sans"],
    "font.size":10, "axes.titlesize":11, "axes.labelsize":10,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.edgecolor":"#444444", "axes.linewidth":0.8,
    "xtick.color":"#444444", "ytick.color":"#444444",
    "figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight",
})

# ---- outcome display + descriptive metadata (from PAPER_DESIGN_NOTES) -------------
OUTCOMES = [
    ("cand20",  "Candidate choice",      "Biden vs. Trump (intent)",        "partisan"),
    ("cov_vac", "Vaccine refusal",       "Refuse vs. accept (intent)",      "health"),
    ("mail_nov","Mail-voting intent",    "Will mail-vote vs. not",          "partisan"),
    ("vote20",  "Turnout intent",        "Will vote vs. not (control)",     "control"),
]
CSV = {
 "cand20":  os.path.join(EXPO,"cand20","figures","cand20_metrics_by_combination.csv"),
 "cov_vac": os.path.join(EXPO,"cov_vac","figures","refuser_metrics_by_combination.csv"),
 "mail_nov":os.path.join(EXPO,"mail_nov","figures","mail_nov_metrics_by_combination.csv"),
 "vote20":  os.path.join(EXPO,"vote20","figures","vote20_metrics_by_combination.csv"),
}
DESC = {  # construct / item / positive class / cutoff+window (for Table 1)
 "cand20": dict(construct="Candidate-choice intent", item="cand20 (latest pre-election answer)",
                pos="Biden = 1 (vs. Trump = 2)", cutoff="max(answer, 2020-09-04)", window="May 1 – Sep 27 2020"),
 "cov_vac":dict(construct="Vaccination refusal",     item="cov_vac 1–5, min across W5–13",
                pos="Refuser = 1 (vs. 2–5)",     cutoff="pre-2020-12-11 (EUA)",    window="Sep 9 – Dec 9 2020"),
 "mail_nov":dict(construct="Mail-voting intent",     item="mail_nov 1–4 (latest answer)",
                pos="Will mail-vote = {3,4} (vs. {1,2})", cutoff="max(answer, 2020-09-04)", window="Jun 1 – Sep 30 2020"),
 "vote20": dict(construct="Turnout intent (control)",item="vote20 (definite-voter)",
                pos="Definite voter = 1 (vs. 2–3)", cutoff="max(answer, 2020-09-04)", window="Sep 1 – Nov 3 2020"),
}
TITLE = {k:v for k,v,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]}
SUBTITLE = {o[0]:o[2] for o in OUTCOMES}

def load(path):
    return list(csv.DictReader(open(path)))
DATA = {k:load(v) for k,v in CSV.items()}

def rows_for(out, sub):
    return [r for r in DATA[out] if r["subgroup"]==sub]
def demo_row(out, sub):
    rr=[r for r in rows_for(out,sub) if r["combination"]=="Demo"]
    return rr[0] if rr else None
def best_row(out, sub):
    rr=rows_for(out,sub)
    return max(rr, key=lambda r:float(r["roc_auc"])) if rr else None
def best_trace_row(out, sub):
    # best combination that uses NO demographics (trace-only: Auth/Feed/Net and their unions)
    rr=[r for r in rows_for(out,sub) if "Demo" not in r["combination"]]
    return max(rr, key=lambda r:float(r["roc_auc"])) if rr else None
def single_block(out, sub, block):
    rr=[r for r in rows_for(out,sub) if r["combination"]==block]
    return rr[0] if rr else None

# ---- Shapley decomposition of the BEST model's ROC-AUC above chance ----------------
# For each subgroup the BEST model is the single highest-AUC combination (same model fig1
# plots as its endpoint). Players are ONLY the blocks in that best combination -- a block the
# best model drops is not credited at all. The empty model (no features) scores ROC-AUC = 0.5,
# so v(emptyset) = 0.5 and v(best) = AUC(best combo). Each block's Shapley value is its marginal
# AUC contribution averaged over all orders; the values sum EXACTLY to AUC(best) - 0.5. Every
# sub-coalition of the best model's blocks is already in the grid, so this is exact, no refit.
# Two-way grouping downstream: INDIVIDUAL = Demo + Auth (own attributes + own tweets),
# RELATIONAL = Feed + Net (network's feed text + its revealed audience-lean).
from math import factorial
from itertools import combinations as _combos
_SHAP_BLOCKS=["Demo","Auth","Feed","Net"]   # canonical order for building combination names
def _auc_coalition(out, sub, blockset):
    if not blockset: return 0.5   # empty model = chance
    parts=[b for b in _SHAP_BLOCKS if b in blockset]
    r=[x for x in rows_for(out,sub) if x["combination"]=="+".join(parts)]
    return float(r[0]["roc_auc"]) if r else None
def shapley_best(out, sub):
    # returns (best_combo_str, best_auc, {block: shapley}) over ONLY the best model's blocks,
    # or None. Blocks absent from the best model do not appear in the dict (contribution = 0).
    b=best_row(out,sub)
    if b is None: return None
    blocks=b["combination"].split("+")
    n=len(blocks); phi={bk:0.0 for bk in blocks}
    for i in blocks:
        others=[bk for bk in blocks if bk!=i]
        for ss in range(len(others)+1):
            for S in _combos(others, ss):
                S=frozenset(S)
                vwith=_auc_coalition(out,sub,S|{i}); vwo=_auc_coalition(out,sub,S)
                if vwith is None or vwo is None: return None
                w=factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
                phi[i]+=w*(vwith-vwo)
    return b["combination"], float(b["roc_auc"]), phi

PARTIES=["Democrat","Republican","Independent/Other"]
PLAB={"Democrat":"Democrats","Republican":"Republicans","Independent/Other":"Independents"}

# =====================================================================================
# FIGURE 1 — headline dumbbell: attribute baseline (Demo) -> best model, within party.
# Grouped by outcome; Overall + each estimable party. Chance line at 0.5.
# =====================================================================================
def fig1():
    rows=[]   # (outcome_key, label, is_group_header, demo, best, best_combo, blo, bhi, dlo, dhi)
    for key,_,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
        rows.append((key,"HDR",True,None,None,None,None,None,None,None))
        order=["Overall"]+[p for p in PARTIES if rows_for(key,p)]
        for sub in order:
            d=demo_row(key,sub); b=best_row(key,sub)
            if d is None or b is None: continue
            lab=("Overall" if sub=="Overall" else PLAB[sub])
            rows.append((key,lab,False,float(d["roc_auc"]),float(b["roc_auc"]),
                         b["combination"],float(b["roc_lo"]),float(b["roc_hi"]),
                         float(d["roc_lo"]),float(d["roc_hi"])))
    # layout
    n=len(rows); fig,ax=plt.subplots(figsize=(8.0,0.46*n+1.0))
    y=np.arange(n)[::-1]
    yticklab=[]; grouplines=[]
    for yi,(key,lab,hdr,d,b,combo,blo,bhi,dlo,dhi) in zip(y,rows):
        if hdr:
            # anchor the group title in the LEFT margin (x in axes fraction, y in data),
            # right-aligned so it grows leftward and never crosses into the plot / chance line.
            trans=ax.get_yaxis_transform()
            ax.text(-0.02, yi, f"{TITLE[key]}", transform=trans, fontweight="bold",
                    va="center", ha="right", fontsize=10.5, clip_on=False)
            if dict((o[0],o[3]) for o in OUTCOMES)[key]=="control":
                ax.text(0.812, yi, "negative control", va="center", ha="right", fontsize=8.5,
                        style="italic", color="#888888")
            yticklab.append("")
            grouplines.append(yi+0.5)
            continue
        # connector
        ax.plot([d,b],[yi,yi], color="#C7C7C7", lw=2.4, zorder=1, solid_capstyle="round")
        # 95% CI whiskers (capped error bars), drawn above the connector so they read clearly.
        # slight vertical offset keeps the two intervals from overlapping each other / the connector.
        ax.errorbar([d],[yi-0.26], xerr=[[d-dlo],[dhi-d]], fmt="none", ecolor=BASE_C,
                    elinewidth=0.8, alpha=0.9, capsize=1.6, capthick=0.7, zorder=2)
        ax.errorbar([b],[yi+0.26], xerr=[[b-blo],[bhi-b]], fmt="none", ecolor=BEST_C,
                    elinewidth=0.9, alpha=0.95, capsize=1.6, capthick=0.7, zorder=2)
        # endpoints: Demo = open gray, Best = filled blue (shape = secondary encoding)
        ax.scatter([d],[yi], s=46, facecolors="white", edgecolors=BASE_C, linewidths=1.6, zorder=3)
        ax.scatter([b],[yi], s=52, facecolors=BEST_C, edgecolors="white", linewidths=1.0, zorder=3)
        lift=b-d
        # lift label above connector midpoint (just clear of the best-model CI bar at yi+0.26)
        ax.text((d+b)/2, yi+0.36, f"+{lift:.2f}", ha="center", va="bottom", fontsize=8, color="#333333")
        # best-combo label just right of the best dot (never on the connector)
        ax.text(bhi+0.012, yi, combo, ha="left", va="center", fontsize=7.8, color=BEST_C, clip_on=False)
        yticklab.append(lab)
    ax.axvline(0.5, color="#B00020", lw=1.0, ls=(0,(4,3)), zorder=0)
    ax.text(0.507, y.max()+0.9, "chance", color="#B00020", fontsize=8, ha="left", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(yticklab, fontsize=9)
    ax.set_ylim(y.min()-0.8, y.max()+1.4)
    ax.set_xlim(0.40,1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    # vertical reference lines so dot values line up with the axis ticks (behind the data)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which="major", color="#DADADA", lw=0.7, zorder=0)
    ax.xaxis.grid(True, which="minor", color="#EFEFEF", lw=0.5, zorder=0)
    ax.set_xlabel("ROC-AUC  (out-of-fold, single model sliced by subgroup)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
    # legend
    from matplotlib.lines import Line2D
    leg=[Line2D([0],[0],marker="o",color="w",markerfacecolor="white",markeredgecolor=BASE_C,
                markeredgewidth=1.6,markersize=8,label="Demographics baseline (age, gender, party)"),
         Line2D([0],[0],marker="o",color="w",markerfacecolor=BEST_C,markeredgecolor="white",
                markersize=8,label="Best model (adds digital traces)"),
         Line2D([0],[0],color="#888888",lw=1.5,marker="|",markersize=9,markeredgewidth=1.3,
                label="95% CI (out-of-fold, 25 repeats)")]
    # park the legend in the empty block on the right of the mail-voting group
    ax.legend(handles=leg, loc="center left", bbox_to_anchor=(0.60,0.43), frameon=False,
              fontsize=8.3, handletextpad=0.4)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(FIGDIR,f"fig1_within_party_lift.{ext}"))
    plt.close(fig); print("fig1 done")

# =====================================================================================
# FIGURE 2 — structure beats text: marginal single-block Overall ROC-AUC per outcome.
# Blocks: Demo (attributes), Net (relational), Feed (text), Auth (text).
# =====================================================================================
def fig2():
    blocks=["Demo","Net","Feed","Auth"]
    blabel={"Demo":"Demographics","Net":"Network","Feed":"Feed text","Auth":"Authored text"}
    keys=[o[0] for o in OUTCOMES]
    vals={blk:[float(single_block(k,"Overall",blk)["roc_auc"]) for k in keys] for blk in blocks}
    los ={blk:[float(single_block(k,"Overall",blk)["roc_lo"]) for k in keys] for blk in blocks}
    his ={blk:[float(single_block(k,"Overall",blk)["roc_hi"]) for k in keys] for blk in blocks}
    x=np.arange(len(keys)); w=0.19
    fig,ax=plt.subplots(figsize=(8.0,4.1))
    for i,blk in enumerate(blocks):
        xs=x+(i-1.5)*w
        yerr=np.array([[v-lo for v,lo in zip(vals[blk],los[blk])],
                       [hi-v for v,hi in zip(vals[blk],his[blk])]])
        bars=ax.bar(xs, [v-0.5 for v in vals[blk]], bottom=0.5, width=w*0.92,
                    color=BLOCK_C[blk], edgecolor="white", linewidth=0.6,
                    label=blabel[blk], zorder=3,
                    yerr=yerr, error_kw=dict(elinewidth=0.8, capsize=1.6, capthick=0.7,
                                             ecolor="#333333", alpha=0.75, zorder=4))
        for xi,v,hi in zip(xs,vals[blk],his[blk]):
            ax.text(xi, hi+0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2, color="#333333")
    ax.axhline(0.5, color="#B00020", lw=1.0, ls=(0,(4,3)), zorder=2)
    ax.text(len(keys)-0.5, 0.503, "chance", color="#B00020", fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([TITLE[k]+("\n(control)" if k=="vote20" else "") for k in keys], fontsize=9)
    ax.set_ylim(0.5,0.98); ax.set_ylabel("ROC-AUC  (Overall, single block alone)")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    ax.text(0.003, 0.975, "error bars: 95% CI (out-of-fold, 25 repeats)", transform=ax.transAxes,
            fontsize=7.6, style="italic", color="#666666", va="top", ha="left")
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(FIGDIR,f"fig2_structure_beats_text.{ext}"))
    plt.close(fig); print("fig2 done")

# =====================================================================================
# FIGURE 3 — operational value (anchor outcome cov_vac): precision @ top-10% flagged,
# across feature sets, vs the base rate.
# =====================================================================================
def fig3():
    sub="Overall"; base=float(rows_for("cov_vac",sub)[0]["prevalence"])
    combos=["Demo","Auth","Feed","Net","Demo+Net","Demo+Auth+Net"]
    clab={"Demo":"Demographics","Auth":"Authored text","Feed":"Feed text","Net":"Network",
          "Demo+Net":"Demo + Network","Demo+Auth+Net":"Demo + Auth + Network (best)"}
    prec=[]
    for c in combos:
        r=[x for x in rows_for("cov_vac",sub) if x["combination"]==c][0]
        prec.append(float(r["precision_top10pct"]))
    y=np.arange(len(combos))[::-1]
    cols=[BLOCK_C["Demo"],BLOCK_C["Auth"],BLOCK_C["Feed"],BLOCK_C["Net"],OI["blue"],OI["blue"]]
    fig,ax=plt.subplots(figsize=(8.0,3.6))
    ax.barh(y, prec, color=cols, edgecolor="white", linewidth=0.6, height=0.68, zorder=3)
    for yi,p in zip(y,prec):
        ax.text(p+0.006, yi, f"{p:.0%}", va="center", ha="left", fontsize=8.5, color="#333333")
    ax.axvline(base, color="#B00020", lw=1.1, ls=(0,(4,3)), zorder=4)
    ax.text(base, y.max()+0.75, f"base rate {base:.0%}", color="#B00020", fontsize=8.3, ha="center", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([clab[c] for c in combos], fontsize=9)
    ax.set_xlim(0,0.46); ax.set_xlabel("Precision among the top 10% flagged as likely refusers")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
    ax.set_title("Vaccine refusal: who you'd actually flag (anchor outcome, n=2,402)",
                 fontsize=10.5, loc="left")
    ax.set_ylim(y.min()-0.6, y.max()+1.1)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(FIGDIR,f"fig3_precision_at_topk.{ext}"))
    plt.close(fig); print("fig3 done")

# =====================================================================================
# TABLES
# =====================================================================================
def _wcsv(path, header, rows):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def table1():
    # descriptive: outcome, construct, positive class, n (covered), prevalence, cutoff, feed window
    hdr=["outcome","construct","positive class","n (modeled)","prevalence","tweet cutoff","feed window"]
    rows=[]
    for k,title,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
        ov=rows_for(k,"Overall")[0]
        d=DESC[k]
        rows.append([title, d["construct"], d["pos"], ov["n"], f'{float(ov["prevalence"]):.3f}',
                     d["cutoff"], d["window"]])
    _wcsv(os.path.join(TABDIR,"table1_outcomes.csv"), hdr, rows)
    # latex
    L=[r"\begin{tabular}{lllrrll}", r"\toprule",
       "Outcome & Construct & Positive class & $n$ & Prev. & Tweet cutoff & Feed window \\\\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(str(x).replace("&","\\&").replace("–","--") for x in r)+" \\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table1_outcomes.tex"),"w").write("\n".join(L))
    print("table1 done")

def table2():
    # main within-party result: outcome, subgroup, n, prev, Demo [CI], best TRACE-ONLY combo [CI],
    # lift of best trace model over Demo alone, and whether traces alone beat demographics.
    hdr=["outcome","subgroup","n","prevalence","demo_roc","demo_ci",
         "best_trace_combo","best_trace_roc","best_trace_ci","lift_over_demo","beats_demo"]
    rows=[]
    for k,title,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
        for sub in ["Overall"]+[p for p in PARTIES if rows_for(k,p)]:
            d=demo_row(k,sub); b=best_trace_row(k,sub)
            if not d or not b: continue
            dr,br=float(d["roc_auc"]),float(b["roc_auc"])
            rows.append([title, ("Overall" if sub=="Overall" else PLAB[sub]), d["n"], f'{float(d["prevalence"]):.3f}',
                         f"{dr:.3f}", f'[{float(d["roc_lo"]):.3f}, {float(d["roc_hi"]):.3f}]',
                         b["combination"], f"{br:.3f}", f'[{float(b["roc_lo"]):.3f}, {float(b["roc_hi"]):.3f}]',
                         f"{br-dr:+.3f}", ("Yes" if br>dr else "No")])
    _wcsv(os.path.join(TABDIR,"table2_within_party.csv"), hdr, rows)
    L=[r"% Best TRACE-ONLY model (no demographics) per outcome x subgroup; last column = does it beat Demo alone.",
       r"\begin{tabular}{llrrcccc}", r"\toprule",
       "Outcome & Subgroup & $n$ & Prev. & Demo AUC & Best trace model (AUC) & Lift & Beats demo? \\\\", r"\midrule"]
    last=None
    for r in rows:
        oc = r[0] if r[0]!=last else ""
        last=r[0]
        best=f"{r[6]} ({r[7]})"
        L.append(f"{oc} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {best} & {r[9]} & {r[10]} \\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table2_within_party.tex"),"w").write("\n".join(L))
    print("table2 done")

def table3():
    # anchor outcome full combination grid (cov_vac Overall): ROC, PR, prec@top10
    hdr=["combination","roc_auc","roc_ci","pr_auc","precision_top10pct","accuracy_0.5"]
    rr=rows_for("cov_vac","Overall")
    order=["Demo","Auth","Feed","Net","Demo+Auth","Demo+Feed","Demo+Net","Auth+Feed","Auth+Net",
           "Feed+Net","Demo+Auth+Feed","Demo+Auth+Net","Demo+Feed+Net","Auth+Feed+Net","Demo+Auth+Feed+Net"]
    idx={r["combination"]:r for r in rr}
    rows=[]
    for c in order:
        if c not in idx: continue
        r=idx[c]
        rows.append([c, f'{float(r["roc_auc"]):.3f}', f'[{float(r["roc_lo"]):.3f}, {float(r["roc_hi"]):.3f}]',
                     f'{float(r["pr_auc"]):.3f}', f'{float(r["precision_top10pct"]):.3f}', f'{float(r["exact_acc_0.5"]):.3f}'])
    _wcsv(os.path.join(TABDIR,"table3_covvac_grid.csv"), hdr, rows)
    base=float(rr[0]["prevalence"])
    L=[f"% cov_vac Overall n={rr[0]['n']}, refuser base rate={base:.3f}; majority-accuracy floor={float(rr[0]['majority_floor']):.3f}",
       r"\begin{tabular}{lccccc}", r"\toprule",
       "Signal set & ROC-AUC & 95\\% CI & PR-AUC & Prec@10\\% & Acc@0.5 \\\\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(r)+" \\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table3_covvac_grid.tex"),"w").write("\n".join(L))
    print("table3 done")

def table4():
    # caveat/robustness: cov_vac Demo baseline under 3-demo vs 7-demo, Overall + per party
    old=os.path.join(EXPO,"figures","refuser_metrics_by_combination.csv")
    if not os.path.exists(old):
        print("table4 skipped (no 7-demo CSV)"); return
    oldrows=list(csv.DictReader(open(old)))
    def od(sub):
        rr=[r for r in oldrows if r["subgroup"]==sub and r["combination"]=="Demo"]
        return float(rr[0]["roc_auc"]) if rr else None
    def ob(sub):  # old best
        rr=[r for r in oldrows if r["subgroup"]==sub]
        return max(rr,key=lambda r:float(r["roc_auc"])) if rr else None
    hdr=["subgroup","demo_3dim","demo_7dim","best_trace_roc","lift_over_3dim","lift_over_7dim"]
    rows=[]
    for sub in ["Overall"]+PARTIES:
        d3=demo_row("cov_vac",sub); b3=best_row("cov_vac",sub)
        if not d3: continue
        d3r=float(d3["roc_auc"]); b3r=float(b3["roc_auc"]); d7=od(sub)
        lab=("Overall" if sub=="Overall" else PLAB[sub])
        rows.append([lab, f"{d3r:.3f}", (f"{d7:.3f}" if d7 is not None else "—"),
                     f"{b3r:.3f}", f"{b3r-d3r:+.3f}", (f"{b3r-d7:+.3f}" if d7 is not None else "—")])
    _wcsv(os.path.join(TABDIR,"table4_baseline_caveat.csv"), hdr, rows)
    L=[r"% cov_vac: trace lift depends on how rich the demographic baseline is.",
       r"\begin{tabular}{lccccc}", r"\toprule",
       "Subgroup & Demo (3-dim) & Demo (7-dim) & Best trace & Lift / 3-dim & Lift / 7-dim \\\\", r"\midrule"]
    for r in rows: L.append(" & ".join(r).replace("—","--")+" \\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table4_baseline_caveat.tex"),"w").write("\n".join(L))
    print("table4 done")

FB2CSV = {k: os.path.join(EXPO, k, "figures", f"{k}_fb2_predict_demographics.csv") for k in CSV}
FB2_TARGETS = [  # (csv target label, display label)
    ("OUTCOME (behavior)",                "Behavior (the outcome)"),
    ("party: Democrat vs rest",           "Party: Democrat"),
    ("party: Republican vs rest",         "Party: Republican"),
    ("party: Independent/Other vs rest",  "Party: Independent/Other"),
    ("gender: Female vs Male",            "Gender: female"),
    ("gender: Male vs Female",            "Gender: male"),
    ("age: 18-24 vs rest",                "Age 18--24"),
    ("age: 25-44 vs rest",                "Age 25--44"),
    ("age: 45-64 vs rest",                "Age 45--64"),
    ("age: 65+ vs rest",                  "Age 65+"),
]

def table5():
    # FB2 robustness: can traces predict demographics? ("are we just inferring demographics?")
    # All-traces (Net+Auth+Feed) AUC for the behavior + each demographic axis, per outcome.
    if not all(os.path.exists(p) for p in FB2CSV.values()):
        print("table5 skipped (missing FB2 CSVs)"); return
    d = {k: {r["target"]: r for r in load(p) if r["block"]=="All traces"} for k,p in FB2CSV.items()}
    cols = [o[0] for o in OUTCOMES]
    hdr = ["target"] + [TITLE[c] for c in cols]
    rows = []
    for tgt, disp in FB2_TARGETS:
        row = [disp]
        for c in cols:
            row.append(f'{float(d[c][tgt]["roc_auc"]):.3f}' if tgt in d[c] else "--")
        rows.append(row)
    _wcsv(os.path.join(TABDIR,"table5_predict_demographics.csv"), hdr, rows)
    L=[r"% FB2: traces DO encode demographics (party AUC ~.64-.75), but demographic legibility is flat",
       r"% across outcomes while behavior-AUC swings .57-.80 -> traces are not merely a demographic proxy.",
       r"\begin{tabular}{lcccc}", r"\toprule",
       "Predicted from traces & " + " & ".join(TITLE[c] for c in cols) + r" \\", r"\midrule"]
    for i,r in enumerate(rows):
        L.append(" & ".join(r)+r" \\")
        if i==0: L.append(r"\midrule")   # rule under the behavior row
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table5_predict_demographics.tex"),"w").write("\n".join(L))
    print("table5 done")

FB1CSV = {k: os.path.join(EXPO, k, "figures", f"{k}_fb1_neighbor_demo_grid.csv") for k in CSV}

def table6():
    # FB1: does the audience-lean (Net) add predictive value OVER neighbors' demographics (NbrDemo)?
    # Per outcome x subgroup: Demo, Demo+NbrDemo, Demo+NbrDemo+Net, and the marginal lift of adding Net.
    if not all(os.path.exists(p) for p in FB1CSV.values()):
        print("table6 skipped (missing FB1 grids)"); return
    D1 = {k: load(p) for k,p in FB1CSV.items()}
    def g(k, sub, combo):
        rr=[r for r in D1[k] if r["subgroup"]==sub and r["combination"]==combo]
        return float(rr[0]["roc_auc"]) if rr else None
    subs = ["Overall","Democrat","Republican","Independent/Other"]
    hdr=["outcome","subgroup","n","Demo","Demo+NbrDemo","Demo+NbrDemo+Net","Net_adds_over_NbrDemo"]
    rows=[]
    for k,title,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
        for sub in subs:
            dn=[r for r in D1[k] if r["subgroup"]==sub and r["combination"]=="Demo"]
            if not dn: continue
            n=dn[0]["n"]
            demo=g(k,sub,"Demo"); dnd=g(k,sub,"Demo+NbrDemo"); dndn=g(k,sub,"Demo+NbrDemo+Net")
            if None in (demo,dnd,dndn): continue
            rows.append([title, ("Overall" if sub=="Overall" else PLAB.get(sub,sub)), n,
                         f"{demo:.3f}", f"{dnd:.3f}", f"{dndn:.3f}", f"{dndn-dnd:+.3f}"])
    _wcsv(os.path.join(TABDIR,"table6_neighbor_demo.csv"), hdr, rows)
    L=[r"% FB1: NbrDemo = relational block built from NEIGHBORS' demographics (leak-free, external panelists).",
       r"% Last column = audience-lean's marginal AUC beyond demographics AND neighbor demographics.",
       r"\begin{tabular}{llrcccc}", r"\toprule",
       r"Outcome & Subgroup & $n$ & Demo & +NbrDemo & +NbrDemo+Net & Net adds \\", r"\midrule"]
    last=None
    for r in rows:
        oc=r[0] if r[0]!=last else ""; last=r[0]
        L.append(f"{oc} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} \\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table6_neighbor_demo.tex"),"w").write("\n".join(L))
    print("table6 done")

SIGCACHE = os.path.join(HERE, "signal_cache")

def table7():
    # Per-person signal volume (median/mean/IQR of authored topic-tweets + scored followees) over the
    # covered modeled population, from cached per-person counts (signal_cache/<out>_signal.csv), which
    # are produced once by scratchpad/signal_volume.py (needs the authored parquet). Rebuilds from cache.
    import statistics as st
    paths = {k: os.path.join(SIGCACHE, f"{k}_signal.csv") for k in CSV}
    if not all(os.path.exists(p) for p in paths.values()):
        print("table7 skipped (missing signal_cache/*_signal.csv)"); return
    def stats(vals):
        v = sorted(int(x) for x in vals)
        n = len(v)
        med = v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2
        med = int(med) if float(med).is_integer() else med
        q1 = v[int(0.25*(n-1))]; q3 = v[int(0.75*(n-1))]
        return med, round(sum(v)/n, 1), q1, q3
    hdr = ["outcome","n_modeled","auth_median","auth_mean","auth_iqr","foll_median","foll_mean","foll_iqr"]
    rows = []
    for k,title,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
        d = load(paths[k])
        am, aM, aq1, aq3 = stats([r["n_auth"] for r in d])
        fm, fM, fq1, fq3 = stats([r["n_foll"] for r in d])
        rows.append([title, len(d), am, aM, f"[{aq1}, {aq3}]", fm, fM, f"[{fq1}, {fq3}]"])
    _wcsv(os.path.join(TABDIR,"table7_signal_volume.csv"), hdr, rows)
    L=[r"% Per-person signal volume over the covered modeled population.",
       r"% authored = topic-keyword-filtered authored tweets fed to the content model;",
       r"% followees = scored followees (>=5 external-panelist followers) backing the audience-lean.",
       r"\begin{tabular}{lrcccc}", r"\toprule",
       r"Outcome & $n$ & Authored tweets (med) & mean & Scored followees (med) & mean \\", r"\midrule"]
    for r in rows:
        L.append(f"{r[0]} & {r[1]} & {r[2]} {r[4]} & {r[3]} & {r[5]} {r[7]} & {r[6]} \\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table7_signal_volume.tex"),"w").write("\n".join(L))
    print("table7 done")

DEMOCACHE = os.path.join(HERE, "demo_cache")
# (group heading, cache column, ordered levels) — levels carry LaTeX-ready labels.
DEMO_GROUPS = [
    ("Party",            "party",     ["Democrat","Republican","Independent/Other"]),
    ("Gender",           "gender",    ["Female","Male"]),
    ("Age",              "age",       ["18--24","25--44","45--64","65+"]),
    ("Race / ethnicity", "race",      ["White","African American","Hispanic","Asian American","Other"]),
    ("Education",        "education", ["High school or less","Some college","College degree","Graduate degree"]),
    ("Household income", "income",    ["<\\$15k","\\$15--25k","\\$25--35k","\\$35--50k",
                                       "\\$50--75k","\\$75--100k","\\$100--150k","\\$150k+"]),
    ("Employment",       "employ",    ["Full-time","Part-time/Gig/Contract","Self-employed/Home-maker",
                                       "Unemployed","Retired","Student"]),
]

def table8():
    # Sample characteristics: demographic breakdown of the modeled population for each outcome.
    # Column percentages (denominator = that outcome's modeled N); first rows give N and the
    # positive-class rate. Built from paper_assets/demo_cache/<out>_demo.csv (build_demo_cache.py).
    import collections
    paths = {k: os.path.join(DEMOCACHE, f"{k}_demo.csv") for k in CSV}
    if not all(os.path.exists(p) for p in paths.values()):
        print("table8 skipped (run paper_assets/build_demo_cache.py first)"); return
    cols = [o[0] for o in OUTCOMES]                      # cand20, cov_vac, mail_nov, vote20
    data = {k: load(paths[k]) for k in CSV}
    N    = {k: len(data[k]) for k in CSV}
    def cnt(k, field):  return collections.Counter(r[field] for r in data[k])
    def pct(k, field, lvl):
        c = cnt(k, field); n = N[k]
        return 100.0*c.get(lvl,0)/n if n else 0.0

    hdr = ["variable","level"] + [TITLE[c] for c in cols]
    rows = [["Sample","N (modeled)"] + [str(N[c]) for c in cols]]
    pv = {}
    for c in cols:
        ov = rows_for(c,"Overall")[0]; pv[c] = float(ov["prevalence"])
    rows.append(["Sample","Positive-class rate (\\%)"] + [f"{100*pv[c]:.1f}" for c in cols])
    for gname, field, levels in DEMO_GROUPS:
        for i,lvl in enumerate(levels):
            rows.append([gname if i==0 else "", lvl] + [f"{pct(c,field,lvl):.1f}" for c in cols])
    _wcsv(os.path.join(TABDIR,"table8_sample_characteristics.csv"), hdr, rows)

    # LaTeX: grouped, column percentages
    L=[r"% Sample characteristics: demographic composition of each outcome's modeled population.",
       r"% Cells are column percentages (denominator = modeled N); first rows give N and positive-class rate.",
       r"\begin{tabular}{ll" + "r"*len(cols) + "}", r"\toprule",
       "Variable & Level & " + " & ".join(TITLE[c] for c in cols) + r" \\", r"\midrule",
       r"\multicolumn{2}{l}{\textit{Modeled $n$}} & " + " & ".join(str(N[c]) for c in cols) + r" \\",
       r"\multicolumn{2}{l}{\textit{Positive-class rate (\%)}} & " + " & ".join(f"{100*pv[c]:.1f}" for c in cols) + r" \\",
       r"\midrule"]
    for gi,(gname, field, levels) in enumerate(DEMO_GROUPS):
        if gi>0: L.append(r"\addlinespace")
        L.append(r"\multicolumn{" + str(2+len(cols)) + r"}{l}{\textbf{" + gname + r"}} \\")
        for lvl in levels:
            L.append(" & " + lvl + " & " + " & ".join(f"{pct(c,field,lvl):.1f}" for c in cols) + r" \\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table8_sample_characteristics.tex"),"w").write("\n".join(L))
    print("table8 done")

def _read_npca(outcome):
    # read NPCA_A, NPCA_S straight from the outcome's config.py (no import side-effects)
    import re
    cfg = os.path.join(EXPO, outcome, "config.py")
    txt = open(cfg).read()
    m = re.search(r"NPCA_A\s*,\s*NPCA_S\s*=\s*(\d+)\s*,\s*(\d+)", txt)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def table9():
    # Selected content-embedding PCA dimensionality per outcome (retuned per outcome, 2026-11-09).
    # Chosen to maximize cross-validated ROC-AUC of the content block added to the 3-dim demographic
    # baseline, swept separately for authored and feed over {3,5,10,15,20,30,50,100,200,full}.
    hdr = ["outcome","authored_dims","feed_dims"]
    rows = []
    for k,title,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
        a,s = _read_npca(k)
        rows.append([title, a if a is not None else "--", s if s is not None else "--"])
    _wcsv(os.path.join(TABDIR,"table9_pca_dims.csv"), hdr, rows)
    L=[r"% Content-embedding PCA dimensionality, selected per outcome by cross-validated ROC-AUC",
       r"% (content block added to the 3-dim demographic baseline); swept over {3,5,10,15,20,30,50,100,200,full}.",
       r"% Raw pooled CT-BERT embeddings are 1024-dimensional for both authored and feed.",
       r"\begin{tabular}{lcc}", r"\toprule",
       r"Outcome & Authored dims & Feed dims \\", r"\midrule"]
    for r in rows:
        L.append(f"{r[0]} & {r[1]} & {r[2]} \\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table9_pca_dims.tex"),"w").write("\n".join(L))
    print("table9 done")

def table10():
    # Shapley decomposition of the BEST model's ROC-AUC above chance (empty model = 0.5), per
    # outcome x subgroup (Overall + every estimable party). Only the blocks IN the best model
    # get a value; blocks the best model drops are shown as "--". top_feature = block with the
    # largest Shapley value (bolded in the LaTeX table). Individual = Demo+Auth, Relational = Feed+Net.
    BLK=["Demo","Auth","Feed","Net"]
    hdr=["outcome","subgroup","best_model","best_roc","above_chance",
         "shap_demo","shap_auth","shap_feed","shap_net","top_feature",
         "individual","relational","relational_share"]
    rows=[]
    for k,title,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
        for sub in ["Overall"]+[p for p in PARTIES if rows_for(k,p)]:
            res=shapley_best(k,sub)
            if res is None: continue
            combo, aucb, phi = res; tot=aucb-0.5
            ind=phi.get("Demo",0.0)+phi.get("Auth",0.0); rel=phi.get("Feed",0.0)+phi.get("Net",0.0)
            top=max(phi, key=lambda b: phi[b])  # argmax among blocks actually in the best model
            share_str=(f"{rel/tot:.0%}" if (tot>=0.05 and ind>=0) else "--")
            cell=lambda b: (f"{phi[b]:+.3f}" if b in phi else "--")
            rows.append([title, ("Overall" if sub=="Overall" else PLAB[sub]),
                         combo, f"{aucb:.3f}", f"{tot:+.3f}",
                         cell("Demo"), cell("Auth"), cell("Feed"), cell("Net"),
                         top, f"{ind:+.3f}", f"{rel:+.3f}", share_str])
    _wcsv(os.path.join(TABDIR,"table10_lift_decomposition.csv"), hdr, rows)
    L=[r"% Shapley decomposition of the BEST model's ROC-AUC above chance (empty model = 0.5).",
       r"% Only blocks in the best model are credited; dropped blocks shown as --. Values sum",
       r"% EXACTLY to (best AUC - 0.5). Largest contributor per row is bold. Individual = Demo + Auth",
       r"% (own attributes + own tweets); Relational = Feed + Net (feed text + audience-lean).",
       r"\begin{tabular}{lllccccccccc}", r"\toprule",
       r"Outcome & Subgroup & Best model & AUC & $>$chance & Demo & Auth & Feed & Net & Indiv.\ & Rel.\ & Rel.\ share \\",
       r"\midrule"]
    last=None
    for r in rows:
        oc=r[0] if r[0]!=last else ""; last=r[0]
        top=r[9]; blkcol={"Demo":5,"Auth":6,"Feed":7,"Net":8}
        cells={b:r[blkcol[b]] for b in blkcol}
        if cells[top]!="--": cells[top]=r"\textbf{"+cells[top]+"}"  # bold the top contributor
        combo_tex=r[2].replace("+",r"+\allowbreak ")
        share_tex=r[12].replace("%",r"\%")  # % is a LaTeX comment char
        L.append(f"{oc} & {r[1]} & {combo_tex} & {r[3]} & {r[4]} & {cells['Demo']} & {cells['Auth']} & "
                 f"{cells['Feed']} & {cells['Net']} & {r[10]} & {r[11]} & {share_tex} \\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"table10_lift_decomposition.tex"),"w").write("\n".join(L))
    print("table10 done")

# =====================================================================================
# FIGURE 4 — where the BEST model's predictive power comes from: Shapley shares of its
# ROC-AUC above chance, collapsed to INDIVIDUAL (Demo + Auth) vs RELATIONAL exposure
# (Feed + Net). Only blocks in the best model contribute. Additive (waterfall) stacked bars:
# the bar's far end is the best model's total AUC above chance. Grouped by outcome like fig1.
# =====================================================================================
def fig4():
    AUTH_C=BLOCK_C["Auth"]; REL_C=OI["blue"]
    rows=[]  # (key, label, is_hdr, individual, relational)
    for key,_,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
        rows.append((key,"HDR",True,None,None))
        for sub in ["Overall"]+[p for p in PARTIES if rows_for(key,p)]:
            res=shapley_best(key,sub)
            if res is None: continue
            _,_,phi=res
            lab=("Overall" if sub=="Overall" else PLAB[sub])
            rows.append((key,lab,False,phi.get("Demo",0.0)+phi.get("Auth",0.0),
                         phi.get("Feed",0.0)+phi.get("Net",0.0)))
    n=len(rows); fig,ax=plt.subplots(figsize=(8.0,0.46*n+1.0))
    y=np.arange(n)[::-1]; yticklab=[]
    for yi,(key,lab,hdr,indiv,rel) in zip(y,rows):
        if hdr:
            trans=ax.get_yaxis_transform()
            is_ctrl=dict((o[0],o[3]) for o in OUTCOMES)[key]=="control"
            ax.text(-0.04, yi+(0.12 if is_ctrl else 0.0), f"{TITLE[key]}", transform=trans,
                    fontweight="bold", va="center", ha="right", fontsize=10.5, clip_on=False)
            if is_ctrl:
                ax.text(-0.04, yi-0.32, "negative control", transform=trans, va="center",
                        ha="right", fontsize=8.0, style="italic", color="#888888", clip_on=False)
            yticklab.append(""); continue
        # additive (waterfall) stack: each segment starts where the last ended, so the bar's
        # far end lands exactly on the total AUC above chance. Individual (Demo+Auth) is drawn
        # first (it is the piece that can go negative within party) so a negative value just
        # starts the bar left of zero rather than overshooting on the opposite side.
        base=0.0
        for val,col in [(indiv,AUTH_C),(rel,REL_C)]:
            ax.barh(yi,val,left=base,color=col,edgecolor="white",linewidth=0.6,height=0.62,zorder=3)
            base+=val
        tot=base  # == indiv + rel == AUC(full) - 0.5
        ax.text(tot+(0.005 if tot>=0 else -0.005), yi, f"{tot:+.2f}",
                ha=("left" if tot>=0 else "right"), va="center", fontsize=7.6, color="#333333")
        yticklab.append(lab)
    ax.axvline(0.0, color="#888888", lw=1.0, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(yticklab, fontsize=9)
    ax.set_ylim(y.min()-0.8, y.max()+1.4)
    xmax=max((a if a else 0)+(r if r else 0) for _,_,h,a,r in rows if not h)+0.045
    xmin=min(0.0, min((a if a else 0)+(r if r else 0) for _,_,h,a,r in rows if not h))-0.03
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_axisbelow(True); ax.xaxis.grid(True, which="major", color="#EAEAEA", lw=0.6, zorder=0)
    ax.set_xlabel("Shapley share of the best model's ROC-AUC above chance (empty model = 0.5)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
    from matplotlib.patches import Patch
    leg=[Patch(facecolor=AUTH_C,edgecolor="white",label="Individual (demographics + authored text)"),
         Patch(facecolor=REL_C,edgecolor="white",label="Relational (feed text + audience-lean)")]
    ax.legend(handles=leg, loc="lower right", frameon=False, fontsize=8.3)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(FIGDIR,f"fig4_lift_decomposition.{ext}"))
    plt.close(fig); print("fig4 done")

# =====================================================================================
# APPENDIX A1 — age and gender subgroup results (parallels the party splits of Table 7).
# One table per non-party demographic. Mirrors the traces-vs-demographics table: within each
# subgroup, the demographic-baseline AUC, the best demographic-free trace model and its AUC,
# and the lift of traces alone over demographics. Supports the Section 5.3 claim that, unlike
# party, slicing on age or gender does NOT collapse the demographic baseline.
# =====================================================================================
def appendix_demo_subgroups():
    GROUPS=[("age","Age",["Age 18-24","Age 25-44","Age 45-64","Age 65+"]),
            ("gender","Gender",["Female","Male"])]
    SUBLAB={"Age 18-24":"18--24","Age 25-44":"25--44","Age 45-64":"45--64","Age 65+":"65+",
            "Female":"Female","Male":"Male"}
    for tag,gtitle,subs in GROUPS:
        hdr=["outcome","subgroup","n","prevalence","demo_roc","best_model","best_roc","lift_over_demo"]
        rows=[]
        for k,title,_,_ in [(o[0],o[1],o[2],o[3]) for o in OUTCOMES]:
            for sub in subs:
                d=demo_row(k,sub); bm=best_row(k,sub)
                if not d or not bm: continue
                dr=float(d["roc_auc"]); br=float(bm["roc_auc"])
                rows.append([title, SUBLAB[sub], d["n"], f'{float(d["prevalence"]):.3f}',
                             f"{dr:.3f}", bm["combination"], f"{br:.3f}", f"{br-dr:+.3f}"])
        _wcsv(os.path.join(TABDIR,f"appendixA_{tag}_subgroups.csv"), hdr, rows)
        L=[f"% Appendix: {gtitle} subgroup results, parallel to the party splits of Table~7.",
           r"% Demo AUC = demographic-baseline out-of-fold ROC-AUC within the subgroup; Best model =",
           r"% highest-AUC combination over ALL blocks (demographics included as an option); Best AUC its",
           r"% ROC-AUC; Lift = Best - Demo.",
           r"\begin{tabular}{llrrclrc}", r"\toprule",
           rf"Outcome & {gtitle} & $n$ & Prev. & Demo AUC & Best model & Best AUC & Lift \\",
           r"\midrule"]
        last=None
        for r in rows:
            oc=r[0] if r[0]!=last else ""; last=r[0]
            L.append(f"{oc} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} & {r[7]} \\\\")
        L+=[r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TABDIR,f"appendixA_{tag}_subgroups.tex"),"w").write("\n".join(L))
        print(f"appendixA {tag} done")

# =====================================================================================
# APPENDIX B — topic keyword lists used to filter authored and feed tweets (referenced in 3.3.3).
# =====================================================================================
def appendix_keywords():
    covid=["covid","corona","vaccin","vax","pfizer","moderna","astrazeneca","biontech","pandemic",
           "lockdown","quarantin","mask","fauci","cdc","sars-cov","wuhan","mrna","warp speed",
           "social distan","anti-vax"]
    election=["vote","voting","voter(s)","ballot","election","electoral","poll(s)","turnout",
              "mail-in / mail in","absentee","early voting","register(ed) to vote","trump","biden",
              "pence","harris","maga","democrat(s) / democratic","republican(s)","gop","campaign",
              "swing state","debate","electionday","#election2020"]
    with open(os.path.join(TABDIR,"appendixB_keywords.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["topic","keywords (case-insensitive substring/stem match)"])
        w.writerow(["COVID-19 (vaccine-refusal outcome)", "; ".join(covid)])
        w.writerow(["Elections (candidate / mail-voting / turnout outcomes)", "; ".join(election)])
    def esc(s): return s.replace("#",r"\#").replace("_",r"\_").replace("&",r"\&")
    L=[r"% Appendix: topic keyword lists (case-insensitive; matched as substrings/word stems).",
       r"\begin{tabular}{p{0.22\textwidth}p{0.70\textwidth}}", r"\toprule",
       r"Topic & Keywords \\", r"\midrule",
       r"COVID-19 \newline (vaccine-refusal outcome) & "+esc(", ".join(covid))+r" \\",
       r"\midrule",
       r"Elections \newline (candidate, mail-voting, turnout outcomes) & "+esc(", ".join(election))+r" \\",
       r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR,"appendixB_keywords.tex"),"w").write("\n".join(L))
    print("appendixB keywords done")

if __name__=="__main__":
    fig1(); fig2(); fig3(); fig4()
    table1(); table2(); table3(); table4(); table5(); table6(); table7(); table8(); table9(); table10()
    appendix_demo_subgroups(); appendix_keywords()
    print("\nALL ASSETS WRITTEN to", HERE)
