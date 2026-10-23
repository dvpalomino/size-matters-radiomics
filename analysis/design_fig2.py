import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, DARK, LIGHT, WARM, WARM2 = "#2a78d6", "#0b3d91", "#9cc3ef", "#b45309", "#f2b46d"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})

# ---------- source slices from QC image (case BRATS_001) ----------
qc = np.asarray(Image.open("data/qc/qc_BRATS_001.png").convert("RGB"))
X = [(15, 272), (293, 549), (570, 826), (847, 1104)]
pan = [qc[14:269, a:b + 1] for a, b in X]
gray = [p.mean(2) for p in pan[:3]]
# crop to brain bounding box (from FLAIR)
m = gray[0] > 12; ys, xs = np.where(m); y0, y1, x0, x1 = ys.min() - 4, ys.max() + 4, xs.min() - 4, xs.max() + 4
crop = lambda a: a[y0:y1, x0:x1]
flair, t1ce, t1 = [crop(g) for g in gray]; overlay = crop(pan[3])
# centre of the green healthy ROI
g = overlay.astype(int); green = (g[..., 1] > 150) & (g[..., 0] < 120) & (g[..., 2] < 120)
gy, gx = [v.mean() for v in np.where(green)]
px_per_mm = flair.shape[1] / (x1 - x0) * (257 / 240)  # panel is 240 mm wide

fig = plt.figure(figsize=(7.4, 6.6))
gsT = fig.add_gridspec(1, 2, left=.02, right=.98, top=.945, bottom=.71, width_ratios=[2.15, 1], wspace=.08)
def label(ax, s, x=-.02, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight="bold", color=INK, va="bottom", ha="left")

# (a) data
ga = gsT[0].subgridspec(1, 4, wspace=.05)
titles = ["FLAIR", "T1gd", "T1", "Labels"]
for i, (im, t) in enumerate(zip([flair, t1ce, t1, overlay], titles)):
    ax = fig.add_subplot(ga[i]); ax.axis("off")
    if i < 3: ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    else: ax.imshow(im)
    ax.set_title(t, fontsize=7.5, color=INK, pad=3)
    if i == 0: label(ax, "a", y=1.12); ax.text(-.02, 1.12, "      MRI data (484 gliomas, MSD/BraTS)", transform=ax.transAxes, fontsize=8, color=INK, va="bottom")
    if i == 3: ax_last = ax
    if i == 0: ax_first = ax
items = [("#e05a5a", "ET  enhancing tumor"), ("#3b6fe0", "NCR  necrosis"), ("#f2c230", "ED  edema"), ("#22c55e", "healthy-tissue control")]
from matplotlib.patches import Patch
ax_first.legend(handles=[Patch(color=c, label=n) for c, n in items], loc="upper left", bbox_to_anchor=(0, -0.03), ncol=4,
                frameon=False, fontsize=6.4, handlelength=1, handleheight=.8, columnspacing=1.4)
# (b) healthy-tissue control with zoom
gb = gsT[1].subgridspec(1, 2, wspace=.04, width_ratios=[1, 1])
ax = fig.add_subplot(gb[0]); ax.imshow(flair, cmap="gray", vmin=0, vmax=255); ax.axis("off")
label(ax, "b", y=1.05); ax.text(-.02, 1.05, "      Healthy-tissue control", transform=ax.transAxes, fontsize=8, color=INK, va="bottom")
Z = 16
ax.add_patch(Rectangle((gx - Z, gy - Z), 2 * Z, 2 * Z, fill=False, ec="#22c55e", lw=.9))
cap_b1 = ax
az = fig.add_subplot(gb[1]); az.imshow(flair, cmap="gray", vmin=0, vmax=255, interpolation="bicubic")
az.set_xlim(gx - Z, gx + Z); az.set_ylim(gy + Z, gy - Z); az.set_xticks([]); az.set_yticks([])
for sp in az.spines.values(): sp.set_visible(True); sp.set_color("#22c55e"); sp.set_linewidth(.9)
sizes = [108, 216, 432, 864, 1728, 3456, 6912]
cmap = plt.get_cmap("Greens")
for k, n in enumerate(sizes):
    r = (3 * n / (4 * np.pi)) ** (1 / 3) * px_per_mm
    az.add_patch(Circle((gx, gy), r, fill=False, ec=cmap(.35 + .09 * k), lw=.9))
fig.canvas.draw()
for a_, t_ in [(cap_b1, "contralateral,\n≥ 10 mm from tumor"), (az, "7 nested ROIs,\n108 → 6 912 voxels")]:
    bb = a_.get_position(); fig.text((bb.x0 + bb.x1) / 2, .715, t_, ha="center", va="top", fontsize=6.2, color=MUTED)
fig.add_artist(FancyArrowPatch((.0, 0), (0, 0)))  # placeholder no-op

# (c) feature extraction strip
gsM = fig.add_gridspec(1, 9, left=.02, right=.98, top=.56, bottom=.45, wspace=.06, width_ratios=[1, 1, 1, 1, 1, 1, .25, 2.2, .1])
f = flair.astype(float)
mask = f > 12
filt = [("Original", f),
        ("LoG σ=2 mm", -ndimage.gaussian_laplace(f, 2.0)),
        ("LoG σ=3 mm", -ndimage.gaussian_laplace(f, 3.0)),
        ("LoG σ=5 mm", -ndimage.gaussian_laplace(f, 5.0)),
        ("Wavelet LLL", ndimage.gaussian_filter(f, 2.0)),
        ("Wavelet HHH", f - ndimage.gaussian_filter(f, 1.0))]
for i, (t, im) in enumerate(filt):
    ax = fig.add_subplot(gsM[i]); ax.axis("off")
    v = np.where(mask, im, np.nan); lo, hi = np.nanpercentile(v, [2, 98])
    ax.imshow(np.where(mask, im, lo), cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(t, fontsize=6.5, color=INK, pad=2)
    if i == 0: label(ax, "c", y=1.25); ax.text(-.02, 1.25, "      PyRadiomics extraction per ROI and sequence: 12 images", transform=ax.transAxes, fontsize=8, color=INK, va="bottom")
ax = fig.add_subplot(gsM[6]); ax.axis("off"); ax.annotate("", xy=(1, .5), xytext=(0, .5), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
ax = fig.add_subplot(gsM[7]); ax.axis("off")
ax.add_patch(Rectangle((0, .02), 1, .96, transform=ax.transAxes, fc="#f3f7fd", ec="none"))
txt = [("1 116 features", 10, "bold"), ("first-order · GLCM · GLRLM", 6.8, None), ("GLSZM · GLDM · NGTDM", 6.8, None), ("+ 66-feature volume-robust set", 6.5, None)]
for k, (s_, fs, w) in enumerate(txt):
    ax.text(.5, .8 - k * .22, s_, ha="center", va="center", fontsize=fs, weight=w or "normal", color=DARK if k == 0 else INK, transform=ax.transAxes)

# (d-f) analyses with real data
SZ = pd.read_csv("out/brats_sizeonly.csv"); DEP = pd.read_csv("out/brats_dependence.csv"); S = json.load(open("out/brats_summary.json"))
src = pd.read_csv("data/brats_features.csv.gz", low_memory=False,
                  usecols=["case", "seq", "roi", "status", "original_firstorder_Energy", "original_glszm_GrayLevelNonUniformity",
                           "original_ngtdm_Coarseness", "original_firstorder_Median", "original_glcm_JointEntropy"])
src = src[(src.status == "ok") & (src.seq == "flair") & src.roi.str.startswith("healthy")]
src["N"] = src.roi.str.split("_").str[1].astype(int)
gsB = fig.add_gridspec(1, 3, left=.075, right=.985, top=.255, bottom=.065, wspace=.48)
qs = [("d", "Q1. Does a feature change when\nonly the ROI size changes?", "Yes: about 1 in 5 features does"),
      ("e", "Q2. Is this a property of the\nfeature or of the tissue?", "Of the feature: healthy and tumor\ntissue show the same scaling"),
      ("f", "Q3. When does it change a\nstudy's conclusions?", "When compared regions differ in\nsize and tissue contrast is low")]
# Q1
ax = fig.add_subplot(gsB[0])
for k, (col, lab) in zip(["original_firstorder_Energy", "original_glszm_GrayLevelNonUniformity", "original_ngtdm_Coarseness", "original_glcm_JointEntropy", "original_firstorder_Median"],
                       [(DARK, "Energy"), (BLUE, "GLSZM GLNU"), (LIGHT, "Coarseness"), (WARM, "JointEntropy"), (MUTED, "Median")]):
    W = src.pivot(index="case", columns="N", values=k); R = W.div(W[6912], axis=0).median()
    ax.plot(sizes, R.loc[sizes].values, marker="o", ms=2, lw=1, color=col, label=lab)
ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(1, color=INK, lw=.5); ax.set_ylim(.007, 600)
ax.set_xlabel("ROI size (voxels)"); ax.set_ylabel("Value / value at 6 912 vx")
ax.legend(frameon=False, fontsize=5.4, loc="upper right", ncol=2, handlelength=1.0, labelspacing=.2, columnspacing=.8, bbox_to_anchor=(1.02, 1.04))
# Q2
ax2 = fig.add_subplot(gsB[1])
mm = SZ[(SZ.family != "shape") & (SZ.seq == "t1ce")].merge(DEP[(DEP.seq == "t1ce") & (DEP.roi == "et")][["feature", "exponent"]], on="feature").dropna(subset=["size_slope", "exponent"])
ax2.scatter(mm.size_slope.clip(-1.5, 1.5), mm.exponent.clip(-1.5, 1.5), s=2.5, color=BLUE, alpha=.5, edgecolor="none")
ax2.plot([-1.5, 1.5], [-1.5, 1.5], color=INK, lw=.6, ls="--"); ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel("Size slope, healthy tissue"); ax2.set_ylabel("Volume exponent, tumor (ET)")
ax2.text(-1.4, 1.25, f"ρ = {spearmanr(mm.size_slope, mm.exponent)[0]:.2f}", fontsize=6.8, color=INK)
# Q3
ax3 = fig.add_subplot(gsB[2]); x = np.arange(3); w = .26
for k, (sq, c, lab) in enumerate([("t1", WARM2, "T1"), ("flair", DARK, "FLAIR"), ("t1ce", BLUE, "T1gd")]):
    st = S[f"strata_et_ed_{sq}"]; ax3.bar(x + (k - 1) * w, [st[t]["top50_proxies"] for t in ["low", "mid", "high"]], w, color=c, label=lab)
st = S["strata_et_ed_flair"]
ax3.set_xticks(x); ax3.set_xticklabels([f"×{st[t]['median_ratio']:.1f}" for t in ["low", "mid", "high"]])
ax3.set_xlabel("ED / ET volume ratio (tertile)"); ax3.set_ylabel("Volume proxies in top 50")
ax3.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1)
for a_, (l, t, ans) in zip([ax, ax2, ax3], qs):
    a_.text(-.30, 1.60, l, transform=a_.transAxes, fontsize=10, weight="bold", color=INK, va="top")
    a_.text(-.20, 1.60, t, transform=a_.transAxes, fontsize=7.3, color=INK, va="top", weight="bold", linespacing=1.15)
    a_.text(-.20, 1.33, ans, transform=a_.transAxes, fontsize=6.6, color=DARK, va="top", style="italic", linespacing=1.15)
    a_.grid(True, color=GRID, lw=.4); a_.set_axisbelow(True)
# section separators
for y in (.635, .415):
    fig.add_artist(plt.Line2D([.02, .98], [y, y], color=GRID, lw=.8))
fig.savefig("out/fig/b0_design.png", dpi=300)
print("ok")
