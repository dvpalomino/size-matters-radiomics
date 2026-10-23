"""BraTS figures. Usage: python brats_figures.py <out_dir> <brats_features.csv.gz>"""
import sys, os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT, SRC = sys.argv[1:3]
D = f"{OUT}/fig"; os.makedirs(D, exist_ok=True)
S = json.load(open(f"{OUT}/brats_summary.json"))
DEP = pd.read_csv(f"{OUT}/brats_dependence.csv"); SZ = pd.read_csv(f"{OUT}/brats_sizeonly.csv"); PR = pd.read_csv(f"{OUT}/brats_paired.csv")
C = pd.read_csv(f"{OUT}/rho_consistency_matrix.csv", index_col=0)
BLUE, DARK, LIGHT, INK, MUTED, GRID = "#2a78d6", "#0b3d91", "#9cc3ef", "#1f2328", "#6b7280", "#e5e7eb"
WARM, WARM2 = "#b45309", "#f2b46d"
PC = {"Robust to volume": BLUE, "Volume-explained": WARM, "Masked by volume": WARM2, "Inconclusive": "#9ca3af", "Not discriminative": "#d9dde3"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "savefig.dpi": 300})
fams = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
SEQN = {"flair": "FLAIR", "t1ce": "T1gd", "t1": "T1"}
RN = {"et": "ET", "ncr": "NCR", "ed": "ED"}

# B1 volumes
d = pd.read_csv(SRC, usecols=["case", "seq", "roi", "n_voxels", "status"])
v = d[(d.seq == "flair") & (d.status == "ok")].pivot(index="case", columns="roi", values="n_voxels")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
bins = np.logspace(1.5, 5.5, 33)
for r, c in [("ed", WARM2), ("et", BLUE), ("ncr", DARK)]:
    axs[0].hist(v[r].dropna(), bins=bins, histtype="step", lw=1.6, color=c, label=RN[r])
axs[0].set_xscale("log"); axs[0].set_xlabel("Region volume (mm³)"); axs[0].set_ylabel("Cases"); axs[0].legend(frameon=False, fontsize=8)
for k, (num, den, c) in enumerate([("ed", "et", WARM), ("ncr", "et", DARK)]):
    rr = (v[num] / v[den]).dropna()
    axs[1].hist(rr, bins=np.logspace(-2, 3.2, 40), histtype="step", lw=1.6, color=c, label=f"{RN[num]} / {RN[den]}")
axs[1].axvline(1, color=INK, lw=.8, ls="--")
axs[1].set_xscale("log"); axs[1].set_xlabel("Within-patient volume ratio"); axs[1].set_ylabel("Cases"); axs[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{D}/b1_volumes.png"); plt.close()

# B2 heatmap family x (seq, roi)
ns = DEP[DEP.family != "shape"]
cols = [(s, r) for s in ["flair", "t1ce", "t1"] for r in ["et", "ncr", "ed"]]
H = ns.pivot_table(index="family", columns=["seq", "roi"], values="abs_rho", aggfunc="median").loc[fams, cols]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
im = ax.imshow(H.values, cmap="Blues", vmin=0, vmax=1, aspect="auto"); ax.grid(False)
ax.set_xticks(range(9)); ax.set_xticklabels([f"{SEQN[s]}\n{RN[r]}" for s, r in cols], fontsize=7.5)
ax.set_yticks(range(6)); ax.set_yticklabels(fams)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        val = H.values[i, j]; ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color="white" if val > .5 else INK)
cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02); cb.set_label("median |ρ|"); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(f"{D}/b2_heatmap.png"); plt.close()

# B3 consistency matrix
lab = [f"{SEQN[c.split('_')[0]]} {RN[c.split('_')[1]]}" for c in C.index]
fig, ax = plt.subplots(figsize=(5.0, 4.2))
im = ax.imshow(C.values, cmap="Blues", vmin=.5, vmax=1); ax.grid(False)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=7.5); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7.5)
for i in range(len(lab)):
    for j in range(len(lab)):
        ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if C.values[i, j] > .85 else INK)
cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02); cb.set_label("Spearman correlation of feature-wise ρ"); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(f"{D}/b3_consistency.png"); plt.close()

# B4 ladder in healthy brain, three sequences
SIZES = [108, 216, 432, 864, 1728, 3456, 6912]
ex = {"original_firstorder_Energy": "FO Energy", "original_glszm_GrayLevelNonUniformity": "GLSZM GLNU",
      "original_ngtdm_Coarseness": "NGTDM Coarseness", "original_gldm_LowGrayLevelEmphasis": "GLDM LowGrayLevelEmphasis",
      "original_glcm_JointEntropy": "GLCM JointEntropy", "original_firstorder_Variance": "FO Variance",
      "original_glcm_Contrast": "GLCM Contrast", "original_firstorder_Median": "FO Median"}
cl = [DARK, BLUE, LIGHT, WARM2, WARM, MUTED, "#111827", "#7c3aed"]
full = pd.read_csv(SRC, low_memory=False, usecols=["case", "seq", "roi", "status"] + list(ex))
full = full[(full.status == "ok") & full.roi.str.startswith("healthy")]
full["N"] = full.roi.str.split("_").str[1].astype(int)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
for ax, sq in zip(axs, ["flair", "t1ce", "t1"]):
    g = full[full.seq == sq]
    W = {k: g.pivot(index="case", columns="N", values=k) for k in ex}
    for (k, lb), c in zip(ex.items(), cl):
        R = W[k].div(W[k][6912], axis=0)
        ax.plot(SIZES, R.median().loc[SIZES].values, marker="o", ms=2.5, lw=1.2, color=c, label=lb)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(1, color=INK, lw=.7); ax.set_title(SEQN[sq], fontsize=8.5, color=INK)
    ax.set_xlabel("ROI size N (voxels)")
axs[0].set_ylabel("Median value relative to N = 6912")
h, lb = axs[0].get_legend_handles_labels(); fig.legend(h, lb, frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(.5, 0))
fig.tight_layout(rect=(0, .14, 1, 1)); fig.savefig(f"{D}/b4_ladder.png"); plt.close()

# B5 healthy size-only AUC by family (FLAIR), unpaired and paired
g = SZ[(SZ.seq == "flair") & (SZ.family != "shape")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
rng = np.random.default_rng(0)
for ax, col, t in zip(axs, ["size_auc", "size_auc_paired"], ["Between patients (unpaired)", "Within patient (paired)"]):
    for i, fm in enumerate(fams):
        h = g[g.family == fm]
        ax.scatter(i + rng.uniform(-.2, .2, len(h)), h[col], s=4, color=np.where(h.safe, WARM, BLUE), alpha=.55, edgecolor="none")
        ax.hlines(h[col].median(), i - .32, i + .32, color=INK, lw=1.4)
    for y in (.65, .75): ax.axhline(y, color=MUTED, lw=.7, ls=":")
    ax.set_xticks(range(6)); ax.set_xticklabels(fams, fontsize=7.5); ax.set_title(t, fontsize=8.5, color=INK); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Size-only AUC, 432 vs 1728 voxels"); axs[0].set_ylim(.45, 1.02)
axs[1].scatter([], [], s=12, color=WARM, label="volume-robust set"); axs[1].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(f"{D}/b5_size_auc.png"); plt.close()

# B8 healthy-tissue scaling slope vs between-patient exponent in tumor regions
mm = SZ[(SZ.family != "shape")].merge(DEP[["feature", "seq", "roi", "exponent"]], on=["feature", "seq"]).dropna(subset=["size_slope", "exponent"])
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
for ax, r in zip(axs, ["et", "ncr", "ed"]):
    h = mm[(mm.seq == "t1ce") & (mm.roi == r)]
    ax.scatter(h.size_slope.clip(-1.6, 1.6), h.exponent.clip(-1.6, 1.6), s=4, color=BLUE, alpha=.5, edgecolor="none")
    ax.plot([-1.6, 1.6], [-1.6, 1.6], color=INK, lw=.7, ls="--"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_title(f"T1gd {RN[r]}  ρ = {spearmanr(h.size_slope, h.exponent)[0]:.2f}", fontsize=8.5, color=INK)
    ax.set_xlabel("Healthy-tissue slope")
axs[0].set_ylabel("Between-patient exponent b")
fig.tight_layout(); fig.savefig(f"{D}/b8_exponents.png"); plt.close()

# B6 naive selection: proxies among top-50 by mismatch tertile
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, pair, title in zip(axs, ["et_ed", "ncr_et"], ["ET vs ED  (edema systematically larger)", "NCR vs ET  (no systematically larger region)"]):
    x = np.arange(3); w = .26
    for k, (s, c) in enumerate([("t1", WARM2), ("flair", DARK), ("t1ce", BLUE)]):
        st = S[f"strata_{pair}_{s}"]
        vals = [st[t]["top50_proxies"] for t in ["low", "mid", "high"]]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=c, label=SEQN[s])
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + .8, f"{v}", ha="center", va="bottom", fontsize=6.3, color=INK)
    st = S[f"strata_{pair}_flair"]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nratio ×{st[t]['median_ratio']:.1f}\nAUC of volume {st[t]['auc_logV']:.2f}"
                        for lab, t in zip(["small", "medium", "large"], ["low", "mid", "high"])], fontsize=6.8)
    ax.set_title(title, fontsize=8.2, color=INK); ax.grid(axis="x", visible=False); ax.set_ylim(0, 56)
    ax.set_xlabel("Within-patient volume difference (tertile)", fontsize=7.5)
axs[0].set_ylabel("Volume proxies among the\n50 best-ranked features"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{D}/b6_naive.png"); plt.close()

# B7 paired ET vs ED: AUC vs |rho| (T1 and FLAIR)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
for ax, s in zip(axs, ["flair", "t1"]):
    g = PR[(PR.pair == "et_vs_ed") & (PR.seq == s) & (PR.family != "shape")].merge(
        DEP[(DEP.seq == s) & (DEP.roi == "et")][["feature", "abs_rho"]], on="feature")
    for c in ["Not discriminative", "Inconclusive", "Robust to volume", "Masked by volume", "Volume-explained"]:
        h = g[g.paired_class == c]
        ax.scatter(h.abs_rho, h.auc_abs, s=6 if c != "Volume-explained" else 10, color=PC[c], label=f"{c} ({len(h)})", edgecolor="none", alpha=.8)
    al = S[f"pair_et_ed_{s}"]["auc_logV"]; ax.axhline(al, color=INK, lw=.8, ls="--"); ax.text(.01, al + .01, "log V alone", fontsize=7)
    ax.set_title(f"ET vs ED, {SEQN[s]}", fontsize=8.5, color=INK); ax.set_xlabel("|ρ| with log V (ET)"); ax.set_xlim(0, 1.01); ax.set_ylim(.5, 1.0)
axs[0].set_ylabel("ET-vs-ED AUC")
h, lb = [], []
for c in ["Robust to volume", "Volume-explained", "Masked by volume", "Inconclusive", "Not discriminative"]:
    h.append(plt.Line2D([], [], marker="o", ls="", color=PC[c], ms=5)); lb.append(c)
fig.legend(h, lb, frameon=False, fontsize=7.5, ncol=5, loc="lower center", bbox_to_anchor=(.5, 0))
fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(f"{D}/b7_paired.png"); plt.close()

# Appendix: Figure 2 for T1gd and T1
g2 = SZ[SZ.family != "shape"]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
rng = np.random.default_rng(0)
for r_, sq in enumerate(["t1ce", "t1"]):
    for c_, (col, t) in enumerate([("size_auc", "between patients"), ("size_auc_paired", "within patient")]):
        ax = axs[r_, c_]; g = g2[g2.seq == sq]
        for i, fm in enumerate(fams):
            h = g[g.family == fm]
            ax.scatter(i + rng.uniform(-.2, .2, len(h)), h[col], s=4, color=np.where(h.safe, WARM, BLUE), alpha=.55, edgecolor="none")
            ax.hlines(h[col].median(), i - .32, i + .32, color=INK, lw=1.4)
        for y in (.65, .75): ax.axhline(y, color=MUTED, lw=.7, ls=":")
        ax.set_xticks(range(6)); ax.set_xticklabels(fams, fontsize=7); ax.set_title(f"{SEQN[sq]}, {t}", fontsize=8.5, color=INK); ax.grid(axis="x", visible=False)
    axs[r_, 0].set_ylabel("Size-only AUC")
axs[0, 0].set_ylim(.45, 1.02)
fig.tight_layout(); fig.savefig(f"{D}/a_size_auc_other.png"); plt.close()

# Appendix: Figure 4 for all sequences x regions
mm = SZ[(SZ.family != "shape")].merge(DEP[["feature", "seq", "roi", "exponent"]], on=["feature", "seq"]).dropna(subset=["size_slope", "exponent"])
fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.0), sharex=True, sharey=True)
for i, sq in enumerate(["flair", "t1ce", "t1"]):
    for j, r in enumerate(["et", "ncr", "ed"]):
        ax = axs[i, j]; h = mm[(mm.seq == sq) & (mm.roi == r)]
        ax.scatter(h.size_slope.clip(-1.6, 1.6), h.exponent.clip(-1.6, 1.6), s=3, color=BLUE, alpha=.5, edgecolor="none")
        ax.plot([-1.6, 1.6], [-1.6, 1.6], color=INK, lw=.7, ls="--"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_title(f"{SEQN[sq]} {RN[r]}  ρ = {spearmanr(h.size_slope, h.exponent)[0]:.2f}", fontsize=8, color=INK)
        if i == 2: ax.set_xlabel("Healthy-tissue slope")
        if j == 0: ax.set_ylabel("Volume exponent")
fig.tight_layout(); fig.savefig(f"{D}/a_exponents_all.png"); plt.close()
print("ok")
