"""
BraTS (MSD Task01) volume-dependence analysis.
Usage: python brats_analysis.py <brats_features.csv.gz> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, theilslopes, wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

SRC, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
NBOOT = 1000
rng = np.random.default_rng(2026)
SEQS = ["flair", "t1ce", "t1"]
TUM = ["et", "ncr", "ed"]
SIZES = [108, 216, 432, 864, 1728, 3456, 6912]

d = pd.read_csv(SRC, low_memory=False)
d = d[d.status == "ok"].copy()
FEATS = [c for c in d.columns if c.startswith(("original_", "log-sigma", "wavelet"))]
FEATS_NS = [c for c in FEATS if "_shape_" not in c]
d[FEATS] = d[FEATS].apply(pd.to_numeric, errors="coerce")

img_of = lambda c: {"original": "Original"}.get(c.split("_")[0], c.split("_")[0].replace("log-sigma-", "LoG ").replace("-0-mm-3D", " mm").replace("wavelet-", "Wavelet "))
fam_of = lambda c: c.split("_")[1]
name_of = lambda c: c.split("_", 2)[2]

# volume-robust configuration
main_imgs = ["original", "log-sigma-2-0-mm-3D", "log-sigma-3-0-mm-3D"]
main_feats = ["firstorder_Mean", "firstorder_Median", "firstorder_10Percentile", "firstorder_90Percentile", "firstorder_Variance",
              "firstorder_MeanAbsoluteDeviation", "firstorder_RobustMeanAbsoluteDeviation", "firstorder_InterquartileRange",
              "glcm_Contrast", "glcm_Idm", "glcm_DifferenceVariance"]
SAFE = {f"{i}_{x}" for i in main_imgs for x in main_feats} - {"log-sigma-3-0-mm-3D_firstorder_Variance", "log-sigma-3-0-mm-3D_firstorder_MeanAbsoluteDeviation"}
SAFE |= {f"wavelet-{s}_firstorder_{x}" for s in ["LLH", "LHL", "HLL", "LHH", "HLH", "HHL", "HHH"]
         for x in ["Variance", "MeanAbsoluteDeviation", "RobustMeanAbsoluteDeviation", "InterquartileRange", "RootMeanSquared"]}




def spearman_matrix(M, y):
    Rm = rankdata(M, axis=0, nan_policy="omit") if False else rankdata(M, axis=0); ry = rankdata(y)
    Rm = Rm - Rm.mean(0); ry = ry - ry.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Rm * ry[:, None]).sum(0) / np.sqrt((Rm ** 2).sum(0) * (ry ** 2).sum())


def dependence(g):
    """g: one row per case for one roi/seq. Returns per-feature table."""
    X = g[FEATS].values; V = g.n_voxels.values.astype(float); logV = np.log(V); n = len(g)
    keep = np.isfinite(X).all(1); X, logV, comp = X[keep], logV[keep], g.n_components.values[keep]; n = len(X)
    rho = spearman_matrix(X, logV)
    pv = np.array([spearmanr(X[:, j], logV)[1] if np.std(X[:, j]) > 0 else 1.0 for j in range(X.shape[1])])
    boot = np.empty((NBOOT, X.shape[1]))
    for b in range(NBOOT):
        i = rng.integers(0, n, n); boot[b] = spearman_matrix(X[i], logV[i])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    Z = np.column_stack([np.ones(n), rankdata(np.log(comp))]); H = Z @ np.linalg.pinv(Z)
    Rx = rankdata(X, axis=0); Rx = Rx - H @ Rx; rv = rankdata(logV); rv = rv - H @ rv
    with np.errstate(invalid="ignore", divide="ignore"):
        rpar = (Rx * rv[:, None]).sum(0) / np.sqrt((Rx ** 2).sum(0) * (rv ** 2).sum())
    expo = np.array([theilslopes(np.log(X[:, j]), logV)[0] if np.all(X[:, j] > 0) else np.nan for j in range(X.shape[1])])
    T = pd.DataFrame(dict(feature=FEATS, rho=rho, p=pv, ci_lo=lo, ci_hi=hi, rho_partial=rpar, exponent=expo, n=n))
    T["q"] = multipletests(np.nan_to_num(T.p, nan=1), method="fdr_bh")[1]
    a = T.rho.abs(); T["abs_rho"] = a
    T["category"] = np.where(T.q >= .05, "Not significant", np.where(a >= .7, "Strong", np.where(a >= .4, "Moderate", "Weak")))
    return T


def paired(A, B):
    """A, B: aligned per-case tables (two ROIs of the same patients). Volume-adjusted paired comparison."""
    Va, Vb = A.n_voxels.values.astype(float), B.n_voxels.values.astype(float)
    dlogV = np.log(Va / Vb); n = len(A)
    lv = np.r_[np.log(Va), np.log(Vb)]
    auc_logV = mannwhitneyu(lv[:n], lv[n:]).statistic / n ** 2
    rows = []
    for c in FEATS:
        a, b = A[c].values, B[c].values
        ok = np.isfinite(a) & np.isfinite(b)
        a, b, dl = a[ok], b[ok], dlogV[ok]; m = len(a)
        dd = a - b
        if m < 20 or np.all(dd == 0) or np.std(np.r_[a, b]) == 0:
            rows.append(dict(feature=c)); continue
        auc = mannwhitneyu(a, b).statistic / m ** 2
        pw = wilcoxon(a, b).pvalue
        positive = np.all(a > 0) and np.all(b > 0)
        ds = np.log(a / b) if positive else dd / np.std(np.r_[a, b], ddof=1)
        fit = sm.OLS(ds, sm.add_constant(dl)).fit(cov_type="HC3")
        rows.append(dict(feature=c, auc=auc, auc_abs=max(auc, 1 - auc), p_wilcoxon=pw, scale="logratio" if positive else "std_diff",
                         mean_diff=ds.mean(), adj_diff=fit.params[0], p_adj=fit.pvalues[0], slope=fit.params[1], p_slope=fit.pvalues[1],
                         rho_delta=spearmanr(ds, dl)[0]))
    P = pd.DataFrame(rows)
    for col in ["p_wilcoxon", "p_adj", "p_slope"]:
        P["q" + col[1:]] = multipletests(P[col].fillna(1), method="fdr_bh")[1]
    def cl(r):
        if not np.isfinite(r.get("auc", np.nan)): return "Not evaluable"
        if r.q_wilcoxon >= .05: return "Masked by volume" if (r.q_adj < .05 and r.q_slope < .05) else "Not discriminative"
        if r.q_adj < .05 and np.sign(r.adj_diff) == np.sign(r.mean_diff): return "Robust to volume"
        if r.q_slope < .05: return "Volume-explained"
        return "Inconclusive"
    P["paired_class"] = P.apply(cl, axis=1)
    return P, dict(n=n, auc_logV=float(max(auc_logV, 1 - auc_logV)), ratio_pct=np.percentile(Va / Vb, [10, 25, 50, 75, 90]).round(2).tolist(),
                   absdlogV=np.abs(dlogV))


S = {"n_cases": int(d.case.nunique())}
dep, pair_tabs, size_tabs = [], [], []
for s in SEQS:
    ds_ = d[d.seq == s]
    # ---------- A. cross-sectional per tumor region ----------
    for r in TUM:
        g = ds_[ds_.roi == r].sort_values("case")
        T = dependence(g); T["seq"] = s; T["roi"] = r; dep.append(T)
        print(s, r, len(g), flush=True)
    # ---------- B. real-tissue size-only test ----------
    H = {n: ds_[ds_.roi == f"healthy_{n}"].set_index("case") for n in SIZES}
    common = sorted(set.intersection(*[set(h.index) for h in H.values()]))
    H = {n: h.loc[common] for n, h in H.items()}
    rows = []
    logN = np.log(SIZES)
    for c in FEATS:
        a, b = H[432][c].values, H[1728][c].values
        auc = mannwhitneyu(b, a).statistic / len(a) ** 2               # unpaired, as in the synthetic experiment
        pa = np.mean(np.sign(b - a) > 0) + .5 * np.mean(b == a)          # paired: P(f1728 > f432) within patient
        M = np.column_stack([H[n][c].values for n in SIZES])
        rel = M / M[:, -1:]                                             # relative to N = 6912, per patient
        if np.all(M > 0):
            sl = np.median([np.polyfit(logN, np.log(M[i]), 1)[0] for i in range(len(M))])
        else:
            sl = np.nan
        rows.append(dict(feature=c, size_auc=max(auc, 1 - auc), size_auc_paired=max(pa, 1 - pa), size_slope=sl,
                         rel_bias_432=np.nanmedian(rel[:, 2]) - 1 if np.all(M[:, -1] != 0) else np.nan,
                         p_size=wilcoxon(a, b).pvalue if np.any(a != b) else 1.0))
    St = pd.DataFrame(rows); St["seq"] = s; St["q_size"] = multipletests(St.p_size, method="fdr_bh")[1]
    size_tabs.append(St)
    # ladder curves for the figure
    curves = {c: [float(np.median(H[n][c].values / H[6912][c].values)) for n in SIZES] for c in
              ["original_firstorder_Energy", "original_glszm_GrayLevelNonUniformity", "original_ngtdm_Coarseness",
               "original_glcm_JointEntropy", "original_firstorder_Variance", "original_glcm_Contrast", "original_firstorder_Mean"]}
    S[f"ladder_{s}"] = curves; S[f"n_healthy_{s}"] = len(common)
    # ---------- C. paired region comparisons ----------
    for A_, B_ in [("et", "ed"), ("ncr", "et")]:
        A = ds_[ds_.roi == A_].set_index("case"); B = ds_[ds_.roi == B_].set_index("case")
        cc = sorted(set(A.index) & set(B.index)); A, B = A.loc[cc], B.loc[cc]
        P, info = paired(A, B); P["seq"] = s; P["pair"] = f"{A_}_vs_{B_}"
        # mismatch strata: proportion of volume-explained among discriminative features, by tertile of |log ratio|
        pair_tabs.append(P)
        S[f"pair_{A_}_{B_}_{s}"] = {k: v for k, v in info.items() if k != "absdlogV"}
        # naive selection by AUC within strata of volume mismatch
        q = pd.qcut(info["absdlogV"], 3, labels=["low", "mid", "high"])
        strata = {}
        for t in ["low", "mid", "high"]:
            m = np.asarray(q == t)
            aa, bb = A[m], B[m]
            au = {}
            for c in FEATS_NS:
                x, y = aa[c].values, bb[c].values
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 10 or np.std(np.r_[x[ok], y[ok]]) == 0: continue
                u = mannwhitneyu(x[ok], y[ok]).statistic / ok.sum() ** 2; au[c] = max(u, 1 - u)
            lv = np.r_[np.log(aa.n_voxels.values), np.log(bb.n_voxels.values)]; k = m.sum()
            ul = mannwhitneyu(lv[:k], lv[k:]).statistic / k ** 2
            strata[t] = dict(n=int(k), median_ratio=float(np.median(np.exp(info["absdlogV"][m]))), auc_logV=float(max(ul, 1 - ul)),
                             top50=sorted(au, key=au.get, reverse=True)[:50])
        S[f"strata_{A_}_{B_}_{s}"] = strata

DEP = pd.concat(dep); SZ = pd.concat(size_tabs); PR = pd.concat(pair_tabs)
for X in (DEP, SZ, PR):
    X["image"] = X.feature.map(img_of); X["family"] = X.feature.map(fam_of); X["name"] = X.feature.map(name_of)
    X["safe"] = X.feature.isin(SAFE); 
DEP.to_csv(f"{OUT}/brats_dependence.csv", index=False); SZ.to_csv(f"{OUT}/brats_sizeonly.csv", index=False); PR.to_csv(f"{OUT}/brats_paired.csv", index=False)

# ---------- summaries ----------
ns = DEP[DEP.family != "shape"]
W = ns.pivot_table(index="feature", columns=["seq", "roi"], values="rho")
W.columns = [f"{a}_{b}" for a, b in W.columns]
C = W.corr(method="spearman")
C.to_csv(f"{OUT}/rho_consistency_matrix.csv")
S["consistency_min"] = float(C.values[np.triu_indices(len(C), 1)].min()); S["consistency_median"] = float(np.median(C.values[np.triu_indices(len(C), 1)]))
S["dep"] = {f"{s}_{r}": dict(n=int(g.n.iloc[0]), sig=float((g.q < .05).mean()), cats=g.category.value_counts().to_dict(),
                             proxies=int((g.abs_rho >= .9).sum()), med_abs=float(g.abs_rho.median()),
                             safe_cats=g[g.safe].category.value_counts().to_dict(), safe_med=float(g[g.safe].abs_rho.median()),
                             safe_max=float(g[g.safe].abs_rho.max()),
                             partial_diff=float((g.rho_partial - g.rho).abs().median()))
            for (s, r), g in ns.groupby(["seq", "roi"])}
prox = ns[ns.abs_rho >= .9].groupby("feature").size()
S["proxy_any"] = int(len(prox)); S["proxy_all9"] = int((prox == 9).sum())
S["proxy_families"] = ns[ns.abs_rho >= .9].drop_duplicates("feature").family.value_counts().to_dict()
S["proxy_exponent_median"] = float(ns[ns.abs_rho >= .9].exponent.abs().median())
S["by_family"] = ns.groupby("family").apply(lambda g: dict(sig=float((g.q < .05).mean()), strong=float((g.category == "Strong").mean()),
                                                           med=float(g.abs_rho.median()))).to_dict()
# size-only test in healthy tissue
szn = SZ[SZ.family != "shape"]
S["size"] = {}
for s, g in szn.groupby("seq"):
    S["size"][s] = dict(n=int(len(g)), high=int((g.size_auc >= .75).sum()), mid=int(((g.size_auc >= .65) & (g.size_auc < .75)).sum()),
                        low=int((g.size_auc < .65).sum()), sig=float((g.q_size < .05).mean()),
                        safe_high=int((g[g.safe].size_auc >= .75).sum()), safe_med=float(g[g.safe].size_auc.median()),
                        safe_max=float(g[g.safe].size_auc.max()),
                        fam_median=g.groupby("family").size_auc.median().round(2).to_dict())
# real-tissue slope vs between-patient exponent in tumor regions
mm = szn.merge(ns[["feature", "seq", "roi", "exponent"]], on=["feature", "seq"]).dropna(subset=["size_slope", "exponent"])
S["slope_real_vs_tumor_exponent"] = {f"{s}_{r}": float(spearmanr(g.size_slope, g.exponent)[0]) for (s, r), g in mm.groupby(["seq", "roi"])}
# paired summaries
prn = PR[PR.family != "shape"]
S["paired"] = {}
for (s, pair), g in prn.groupby(["seq", "pair"]):
    proxies = set(ns[(ns.seq == s) & (ns.abs_rho >= .9)].feature)
    top = g.sort_values("auc_abs", ascending=False)
    S["paired"][f"{s}_{pair}"] = dict(classes=g.paired_class.value_counts().to_dict(), max_auc=float(g.auc_abs.max()),
                                      top20_proxies=int(top.head(20).feature.isin(proxies).sum()),
                                      top50_proxies=int(top.head(50).feature.isin(proxies).sum()),
                                      top50_volexpl=int((top.head(50).paired_class == "Volume-explained").sum()),
                                      safe_classes=g[g.safe].paired_class.value_counts().to_dict(),
                                      proxy_slope_median=float(g[g.feature.isin(proxies) & (g.scale == "logratio")].slope.abs().median()))
    for t, st in S[f"strata_{pair.replace('_vs_', '_')}_{s}"].items():
        st["top50_proxies"] = int(sum(f in proxies for f in st["top50"])); st["top20_proxies"] = int(sum(f in proxies for f in st["top50"][:20]))
        del st["top50"]
# volume summary
v = d[d.seq == "flair"].pivot(index="case", columns="roi", values="n_voxels")
S["vol"] = {r: np.percentile(v[r].dropna(), [0, 10, 25, 50, 75, 90, 100]).round(0).tolist() for r in TUM}
S["ratio_ed_et"] = np.percentile((v.ed / v.et).dropna(), [0, 10, 25, 50, 75, 90, 100]).round(2).tolist()
S["ratio_ncr_et"] = np.percentile((v.ncr / v.et).dropna(), [0, 10, 25, 50, 75, 90, 100]).round(2).tolist()
S["ncomp_median"] = d[d.seq == "flair"].groupby("roi").n_components.median().to_dict()
S["n_roi"] = {r: int(v[r].notna().sum()) for r in TUM}
json.dump(S, open(f"{OUT}/brats_summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in S.items() if not k.startswith(("ladder", "strata"))}, indent=1, default=float)[:12000])
for k in S:
    if k.startswith("strata"): print(k, json.dumps(S[k], default=float))
