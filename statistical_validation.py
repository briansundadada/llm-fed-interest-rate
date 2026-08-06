from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(r"C:\Users\HUAWEI\Documents\xwechat_files\wxid_p5actilp89y722_75af\msg\file\2026-07\q1_synthetic_deepseek_only_new.csv")
WEIGHT_DATA = Path(r"D:\Team2\新结果\q1_weights_180.csv")
OUT = Path(r"D:\Team2\1-2检验\统计检验结果.txt")
RNG = np.random.default_rng(20260802)
df = pd.read_csv(DATA)
p = df[df["group"].eq("policy_persona")].copy()
p["persona"] = pd.Categorical(p["persona"], ["dove", "centrist", "hawk"], ordered=True)

def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    out = np.empty_like(ps)
    running = 0.0
    m = len(ps)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank)*ps[idx])
        out[idx] = min(1.0, running)
    return out

def perm_order_test(sub, n_perm=200000):
    y = sub["rate_decision_bps"].to_numpy(float)
    labels = sub["persona"].astype(str).to_numpy()
    obs = y[labels == "hawk"].mean() - y[labels == "dove"].mean()
    ge = 0
    for _ in range(n_perm):
        z = RNG.permutation(labels)
        t = y[z == "hawk"].mean() - y[z == "dove"].mean()
        ge += t >= obs - 1e-12
    return obs, (ge + 1) / (n_perm + 1)

def bootstrap_diff(a, b, n_boot=100000):
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = RNG.choice(a, len(a), replace=True).mean() - RNG.choice(b, len(b), replace=True).mean()
    return np.quantile(vals, [0.025, 0.975])

lines = []
lines.append(f"Raw rows={len(df)}; policy-persona rows={len(p)}")
lines.append("\nDESCRIPTIVE MEANS (bps)")
means = p.groupby(["scenario_id", "persona"], observed=True)["rate_decision_bps"].agg(["count", "mean", "std"])
lines.append(means.to_string())

lines.append("\nCONCLUSION 1: persona ordering within each scenario")
for sc in ["sc1", "sc2", "sc3"]:
    sub = p[p.scenario_id.eq(sc)]
    groups = [sub.loc[sub.persona.eq(x), "rate_decision_bps"].to_numpy() for x in ["dove", "centrist", "hawk"]]
    kw = stats.kruskal(*groups)
    eps2 = max(0, (kw.statistic - 3 + 1) / (len(sub) - 3))
    contrast, pp = perm_order_test(sub)
    lines.append(f"{sc}: Kruskal-Wallis H={kw.statistic:.4f}, p={kw.pvalue:.6g}, epsilon^2={eps2:.4f}; hawk-dove mean contrast={contrast:.2f}, one-sided permutation p={pp:.6g}")
    pairs = list(itertools.combinations(["dove", "centrist", "hawk"], 2))
    raw = []
    vals = []
    for a,b in pairs:
        xa = sub.loc[sub.persona.eq(a), "rate_decision_bps"]
        xb = sub.loc[sub.persona.eq(b), "rate_decision_bps"]
        u = stats.mannwhitneyu(xa, xb, alternative="two-sided", method="asymptotic")
        raw.append(u.pvalue); vals.append((a,b,u.statistic))
    adj = holm(raw)
    for (a,b,u), rp, ap in zip(vals, raw, adj):
        lines.append(f"  {a} vs {b}: Mann-Whitney U={u:.1f}, raw p={rp:.6g}, Holm p={ap:.6g}")

grand = p.rate_decision_bps.mean()
persona_means = p.groupby("persona", observed=True).rate_decision_bps.mean()
scenario_means = p.groupby("scenario_id").rate_decision_bps.mean()
cell_means = p.groupby(["persona", "scenario_id"], observed=True).rate_decision_bps.mean()
n_cell = 15; a = b = 3
ss_persona = b*n_cell*sum((persona_means-grand)**2)
ss_scenario = a*n_cell*sum((scenario_means-grand)**2)
ss_inter = n_cell*sum((cell_means.loc[i,j]-persona_means.loc[i]-scenario_means.loc[j]+grand)**2 for i in persona_means.index for j in scenario_means.index)
ss_error = sum((row.rate_decision_bps-cell_means.loc[row.persona,row.scenario_id])**2 for row in p.itertuples())
df_error = a*b*(n_cell-1); ms_error=ss_error/df_error
lines.append("\nTWO-WAY ANOVA (supporting sensitivity analysis)")
for name,ss,dfx in [("persona",ss_persona,a-1),("scenario",ss_scenario,b-1),("interaction",ss_inter,(a-1)*(b-1))]:
    f=(ss/dfx)/ms_error; pv=stats.f.sf(f,dfx,df_error); eta=ss/(ss+ss_error)
    lines.append(f"{name}: F({dfx},{df_error})={f:.4f}, p={pv:.6g}, partial eta^2={eta:.4f}")

lines.append("\nCONCLUSION 2: Scenario 3 has the largest hawk-dove gap")
gaps = {}
samples = {}
for sc in ["sc1", "sc2", "sc3"]:
    sub = p[p.scenario_id.eq(sc)]
    h = sub.loc[sub.persona.eq("hawk"), "rate_decision_bps"].to_numpy(float)
    d = sub.loc[sub.persona.eq("dove"), "rate_decision_bps"].to_numpy(float)
    gaps[sc] = h.mean()-d.mean(); samples[sc]=(h,d)
    ci = bootstrap_diff(h,d)
    lines.append(f"{sc}: hawk-dove gap={gaps[sc]:.2f} bps; bootstrap 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")

obs = gaps["sc3"] - max(gaps["sc1"], gaps["sc2"])
ge = 0
n_perm = 200000
# Under the null of equal gap size across scenarios, permute scenario labels
# within each persona, preserving the overall hawk/dove main effect.
wide = {pers: p.loc[p.persona.eq(pers), ["scenario_id", "rate_decision_bps"]].copy() for pers in ["hawk", "dove"]}
for _ in range(n_perm):
    pm = {}
    for pers in ["hawk", "dove"]:
        vals = wide[pers]["rate_decision_bps"].to_numpy()
        labs = RNG.permutation(wide[pers]["scenario_id"].to_numpy())
        pm[pers] = {sc: vals[labs == sc].mean() for sc in ["sc1", "sc2", "sc3"]}
    pg = {sc: pm["hawk"][sc]-pm["dove"][sc] for sc in ["sc1", "sc2", "sc3"]}
    t=pg["sc3"]-max(pg["sc1"],pg["sc2"])
    ge += t >= obs - 1e-12
lines.append(f"Observed sc3 - max(sc1,sc2) gap difference={obs:.2f} bps; one-sided stratified permutation p={(ge+1)/(n_perm+1):.6g}")

# Bootstrap the two direct gap differences.
for other in ["sc1", "sc2"]:
    vals=np.empty(100000)
    h3,d3=samples["sc3"]; ho,do=samples[other]
    for i in range(len(vals)):
        vals[i]=(RNG.choice(h3,len(h3),True).mean()-RNG.choice(d3,len(d3),True).mean())-(RNG.choice(ho,len(ho),True).mean()-RNG.choice(do,len(do),True).mean())
    ci=np.quantile(vals,[.025,.975])
    lines.append(f"sc3 gap - {other} gap={gaps['sc3']-gaps[other]:.2f} bps; bootstrap 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")

# Weight experiment: use the precomputed inflation minus employment difference.
w_all = pd.read_csv(WEIGHT_DATA)
w = w_all[w_all.persona.isin(["hawk", "centrist", "dove"])].copy()
lines.append("\nWEIGHT EXPERIMENT QUALITY CONTROL")
lines.append(f"Raw rows={len(w_all)}; three-persona rows={len(w)}; duplicate persona/scenario/run keys={w_all.duplicated(['persona','scenario_id','run_id']).sum()}")
lines.append(f"Weight-sum range=[{w_all.weights_sum.min():.6f}, {w_all.weights_sum.max():.6f}]; missing values={int(w_all.isna().sum().sum())}")
wm = w.groupby(["scenario_id","persona"]).weight_difference.agg(["count","mean","std"])
lines.append("\nWEIGHT-DIFFERENCE DESCRIPTIVES (inflation minus employment; proportion units)")
lines.append(wm.to_string())

lines.append("\nWEIGHT PERSONA DIFFERENCES WITHIN EACH SCENARIO")
for sc in ["sc1","sc2","sc3"]:
    sub=w[w.scenario_id.eq(sc)]
    groups=[sub.loc[sub.persona.eq(x),"weight_difference"].to_numpy() for x in ["dove","centrist","hawk"]]
    kw=stats.kruskal(*groups)
    eps=max(0,(kw.statistic-2)/(len(sub)-3))
    lines.append(f"{sc}: Kruskal-Wallis H={kw.statistic:.4f}, p={kw.pvalue:.6g}, epsilon^2={eps:.4f}")
    pairs=list(itertools.combinations(["dove","centrist","hawk"],2)); raw=[]; vals=[]
    for a1,b1 in pairs:
        xa=sub.loc[sub.persona.eq(a1),"weight_difference"]; xb=sub.loc[sub.persona.eq(b1),"weight_difference"]
        u=stats.mannwhitneyu(xa,xb,alternative="two-sided",method="asymptotic")
        raw.append(u.pvalue); vals.append((a1,b1,u.statistic))
    for (a1,b1,u),rp,ap in zip(vals,raw,holm(raw)):
        lines.append(f"  {a1} vs {b1}: U={u:.1f}, raw p={rp:.6g}, Holm p={ap:.6g}")

# Balanced two-way ANOVA on weight difference.
grand=w.weight_difference.mean(); pm=w.groupby("persona").weight_difference.mean(); smn=w.groupby("scenario_id").weight_difference.mean(); cm=w.groupby(["persona","scenario_id"]).weight_difference.mean()
ss_p=3*15*sum((pm-grand)**2); ss_s=3*15*sum((smn-grand)**2)
ss_i=15*sum((cm.loc[i,j]-pm.loc[i]-smn.loc[j]+grand)**2 for i in pm.index for j in smn.index)
ss_e=sum((row.weight_difference-cm.loc[row.persona,row.scenario_id])**2 for row in w.itertuples()); dfe=126; mse=ss_e/dfe
lines.append("\nTWO-WAY ANOVA ON WEIGHT DIFFERENCE (supporting sensitivity analysis)")
for name,ss,dfx in [("persona",ss_p,2),("scenario",ss_s,2),("interaction",ss_i,4)]:
    f=(ss/dfx)/mse; pv=stats.f.sf(f,dfx,dfe); eta=ss/(ss+ss_e)
    lines.append(f"{name}: F({dfx},{dfe})={f:.4f}, p={pv:.6g}, partial eta^2={eta:.4f}")

lines.append("\nHAWK-DOVE WEIGHT-DIFFERENCE GAPS")
wg={}
for sc in ["sc1","sc2","sc3"]:
    hs=w.loc[(w.persona.eq('hawk')) & (w.scenario_id.eq(sc)),'weight_difference'].to_numpy()
    ds=w.loc[(w.persona.eq('dove')) & (w.scenario_id.eq(sc)),'weight_difference'].to_numpy()
    wg[sc]=hs.mean()-ds.mean(); ci=bootstrap_diff(hs,ds)
    lines.append(f"{sc}: gap={100*wg[sc]:.2f} percentage points; bootstrap 95% CI [{100*ci[0]:.2f}, {100*ci[1]:.2f}]")

cent3=w.loc[(w.persona.eq('centrist')) & (w.scenario_id.eq('sc3')),'weight_difference'].to_numpy()
ci3=np.quantile([RNG.choice(cent3,len(cent3),True).mean() for _ in range(100000)],[.025,.975])
lines.append(f"sc3 centrist mean={100*cent3.mean():.2f} percentage points; bootstrap 95% CI [{100*ci3[0]:.2f}, {100*ci3[1]:.2f}]")

# Cell-level association between the separate weight and decision experiments.
rate_cells=p.groupby(["scenario_id","persona"],observed=True).rate_decision_bps.mean().rename('rate')
weight_cells=w.groupby(["scenario_id","persona"]).weight_difference.mean().rename('weight_diff')
joined=pd.concat([rate_cells,weight_cells],axis=1).dropna()
pear=stats.pearsonr(joined.weight_diff,joined.rate); spear=stats.spearmanr(joined.weight_diff,joined.rate)
lines.append("\nCELL-LEVEL ASSOCIATION BETWEEN SEPARATE EXPERIMENTS (9 persona-scenario cells)")
lines.append(f"Pearson r={pear.statistic:.4f}, p={pear.pvalue:.6g}; Spearman rho={spear.statistic:.4f}, p={spear.pvalue:.6g}")
lines.append("Interpretation caveat: this is cross-cell consistency, not run-level mediation or causal identification.")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
