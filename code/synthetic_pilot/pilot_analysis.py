"""Analysis of the synthetic pilot. Usage: python pilot_analysis.py <pilot_out> <brats_sizeonly.csv> <volume_robust_features.txt> <fig_dir>"""
import sys, os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

P, SZB, ROB, FIG = sys.argv[1:5]
d = pd.read_csv(f"{P}/synthetic_features.csv.gz")
rob = set(l.strip() for l in open(ROB) if l.strip())
feats = [c for c in d.columns if c not in ("texture", "rep", "N")]
FAMS = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]

def size_auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.nanstd(np.r_[a, b]) == 0: return .5
    u = mannwhitneyu(b, a).statistic / (len(a) * len(b)); return max(u, 1 - u)

def paired_auc(a, b):
    diff = np.asarray(b, float) - np.asarray(a, float); p = (np.sum(diff > 0) + .5 * np.sum(diff == 0)) / len(diff)
    return max(p, 1 - p)

rows = []
for tex, g in d.groupby("texture"):
    s, l = g[g.N == 432].sort_values("rep"), g[g.N == 1728].sort_values("rep")
    med = g.groupby("N")[feats].median()
    for f in feats:
        y = med[f]; slope = np.polyfit(np.log(y.index), np.log(y.values), 1)[0] if (y > 0).all() else np.nan
        rows.append(dict(texture=tex, feature=f, family=f.split("_")[1], size_auc=size_auc(s[f], l[f]),
                         size_auc_paired=paired_auc(s[f].values, l[f].values), size_slope=slope, robust=f in rob))
R = pd.DataFrame(rows); R.to_csv(f"{P}/pilot_sizeonly.csv", index=False)

B = pd.read_csv(SZB); B = B[(B.family != "shape") & (B.seq == "flair")].set_index("feature")
summ = {"n_rep": int(d.rep.nunique()), "n_features": len(feats)}
for tex, g in R.groupby("texture"):
    rb = g[g.robust]; j = g.set_index("feature").join(B[["size_auc"]], rsuffix="_brats").dropna(subset=["size_auc_brats"])
    summ[tex] = dict(high=float((g.size_auc >= .75).mean()), high_n=int((g.size_auc >= .75).sum()),
                     tex_high=float((g[g.family.isin(FAMS[2:])].size_auc >= .75).mean()),
                     fo_high=float((g[g.family.isin(FAMS[:2])].size_auc >= .75).mean()),
                     robust_max=float(rb.size_auc.max()), robust_median=float(rb.size_auc.median()),
                     robust_medium=int((rb.size_auc >= .65).sum()),
                     robust_paired_median=float(rb.size_auc_paired.median()), robust_paired_max=float(rb.size_auc_paired.max()),
                     robust_paired_high=int((rb.size_auc_paired >= .75).sum()),
                     all_paired_median=float(g.size_auc_paired.median()),
                     rho_vs_brats=float(spearmanr(j.size_auc, j.size_auc_brats)[0]),
                     robust_absslope_max=float(rb.size_slope.abs().max()))
json.dump(summ, open(f"{P}/pilot_summary.json", "w"), indent=1)

# table rows by family for the supplementary
tab = []
for f in FAMS + ["All", "Volume-robust set"]:
    r = {"family": f}
    for tex in ["fine", "coarse"]:
        g = R[R.texture == tex]
        g = g if f == "All" else (g[g.robust] if f == "Volume-robust set" else g[g.family == f])
        r.update({f"{tex}_n": len(g), f"{tex}_med": g.size_auc.median(), f"{tex}_high": int((g.size_auc >= .75).sum()),
                  f"{tex}_paired": g.size_auc_paired.median()})
    tab.append(r)
pd.DataFrame(tab).to_csv(f"{P}/pilot_table.csv", index=False)

# figure
INK, MUTED, GRID, BLUE, WARM = "#1f2328", "#6b7280", "#e5e7eb", "#2a78d6", "#d97706"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "pdf.fonttype": 42})
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey=True)
rng = np.random.default_rng(0)
for i, tex in enumerate(["fine", "coarse"]):
    for j, (col, lab) in enumerate([("size_auc", "independent realizations (unpaired)"), ("size_auc_paired", "same realization (paired)")]):
        ax = axs[i, j]; g = R[R.texture == tex]
        for k, f in enumerate(FAMS):
            h = g[g.family == f]; xj = k + rng.uniform(-.28, .28, len(h))
            ax.scatter(xj[~h.robust.values], h[col].values[~h.robust.values], s=3, color=BLUE, alpha=.45, edgecolor="none")
            ax.scatter(xj[h.robust.values], h[col].values[h.robust.values], s=9, color=WARM, edgecolor="none", zorder=3)
            ax.plot([k - .32, k + .32], [h[col].median()] * 2, color=INK, lw=1.4)
        for t in (.65, .75): ax.axhline(t, color=MUTED, lw=.7, ls=":")
        ax.set_xticks(range(6)); ax.set_xticklabels(FAMS, fontsize=7.5); ax.grid(axis="x", visible=False)
        ax.set_title(f"{'Fine' if tex == 'fine' else 'Coarse'} texture, {lab}", fontsize=8.5, color=INK)
    axs[i, 0].set_ylabel("Size-only ROC AUC, 432 vs 1,728 voxels")
axs[0, 1].scatter([], [], s=9, color=WARM, label="volume-robust configuration"); axs[0, 1].legend(frameon=False, fontsize=7, loc="lower right")
axs[0, 0].set_ylim(.45, 1.02)
fig.tight_layout(); os.makedirs(FIG, exist_ok=True); os.makedirs(FIG.replace("/fig", "/fig_pdf"), exist_ok=True)
fig.savefig(f"{FIG}/s_pilot.png", dpi=300); fig.savefig(f"{FIG.replace('/fig', '/fig_pdf')}/s_pilot.pdf")
print(json.dumps(summ, indent=1))
