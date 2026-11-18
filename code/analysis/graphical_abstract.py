"""Graphical abstract (Physica Medica: >=1328 px wide, readable at 13 cm)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.stats import spearmanr
INK, MUTED = "#1f2328", "#6b7280"; BLUE, DARK, WARM = "#2a78d6", "#0b3d91", "#b45309"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "pdf.fonttype": 42})
SZ = pd.read_csv("out/brats_sizeonly.csv"); DEP = pd.read_csv("out/brats_dependence.csv")
fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(1, 3, left=.035, right=.985, top=.80, bottom=.30, wspace=.33)
for x, t in zip([.035, .365, .695], ["1  Only size changes", "2  Same behavior in tumors", "3  Larger group: volume selected"]):
    fig.text(x, .93, t, fontsize=15, weight="bold", color=DARK, va="top")
a = fig.add_subplot(gs[0]); a.set_aspect("equal"); a.axis("off")
a.add_patch(Circle((0, 0), 1.05, color="#e8e8e8"))
for k, r in enumerate(np.sqrt(np.array([108, 216, 432, 864, 1728, 3456, 6912]) / 6912)):
    a.add_patch(Circle((0, 0), r, fill=False, lw=2, color=plt.cm.Blues(.35 + k * .09)))
a.text(1.25, .35, "7 nested ROIs\nsame healthy\ntissue", fontsize=12.5, color=INK, va="center")
a.text(1.25, -.55, "≈1 in 5\nfeatures change\nwith size alone", fontsize=13.5, color=WARM, weight="bold", va="center")
a.set_xlim(-1.1, 3.2); a.set_ylim(-1.1, 1.1)
b = fig.add_subplot(gs[1])
mm = SZ[SZ.family != "shape"].merge(DEP[["feature", "seq", "roi", "exponent"]], on=["feature", "seq"]).dropna(subset=["size_slope", "exponent"])
h = mm[(mm.seq == "t1ce") & (mm.roi == "et")]
b.scatter(h.size_slope.clip(-1.6, 1.6), h.exponent.clip(-1.6, 1.6), s=9, color=BLUE, alpha=.45, edgecolor="none")
b.plot([-1.6, 1.6], [-1.6, 1.6], color=INK, lw=1, ls="--"); b.set_xlim(-1.6, 1.6); b.set_ylim(-1.6, 1.6)
b.set_xlabel("Size slope, healthy tissue", fontsize=12); b.set_ylabel("Volume exponent, tumor", fontsize=12)
b.text(1.5, -1.45, "T1gd, enhancing tumor", fontsize=10.5, color=MUTED, ha="right")
b.text(-1.5, 1.35, f"ρ = {spearmanr(h.size_slope, h.exponent)[0]:.2f}", fontsize=14, color=INK, weight="bold")
c = fig.add_subplot(gs[2])
x = np.arange(3); w = .38; ed = [20, 13, 0]
c.bar(x - w / 2, ed, w, color=WARM, label="ED vs ET (ED larger)")
c.bar(x + w / 2, [0] * 3, w, color=BLUE, label="NCR vs ET (no larger region)")
for xi, v in zip(x - w / 2, ed): c.text(xi, v + .5, str(v), ha="center", fontsize=12, color=INK)
for xi in x + w / 2: c.text(xi, .5, "0", ha="center", fontsize=12, color=INK)
c.set_xticks(x); c.set_xticklabels(["T1", "FLAIR", "T1gd"], fontsize=12); c.set_ylim(0, 24)
c.set_ylabel("Volume proxies in top 20", fontsize=12); c.legend(frameon=False, fontsize=10.5, loc="upper right")
fig.patches.append(FancyBboxPatch((.035, .03), .95, .15, boxstyle="round,pad=.01,rounding_size=.015", transform=fig.transFigure,
                                  fc="#eef4fb", ec=BLUE, lw=1.2))
fig.text(.51, .135, "Standardized features are reproducible, but not yet comparable across ROI sizes", ha="center", fontsize=15.5, weight="bold", color=DARK)
fig.text(.51, .07, "Compare with volume alone  ·  run a size-only control  ·  fixed discretization lower bound  ·  extend IBSI", ha="center", fontsize=13, color=INK)
fig.savefig("out/fig/graphical_abstract.png", dpi=200); fig.savefig("out/fig_pdf/graphical_abstract.pdf")
