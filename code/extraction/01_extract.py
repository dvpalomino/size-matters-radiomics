"""
BraTS volume-dependence study - step 1: ROI construction and PyRadiomics extraction.

ROIs per case (native BraTS sub-regions + negative control), all on the same brain-normalised image:
  et          enhancing tumor
  ncr         necrosis / non-enhancing tumor core
  ed          peritumoral edema
  healthy_N   compact ROIs of exactly N voxels (N = 108, 216, 432, 864, 1728, 3456, 6912), nested, in
              normal-appearing brain of the contralateral hemisphere (mirror of the tumor centroid, >= 10 mm
              from the whole tumor): same tissue at different sizes = real-tissue size-only test

Supported layouts (--format):
  msd       Medical Segmentation Decathlon Task01_BrainTumour: imagesTr/BRATS_xxx.nii.gz (4D: FLAIR, T1w, T1gd, T2w)
            + labelsTr/BRATS_xxx.nii.gz (1 edema, 2 non-enhancing tumor, 3 enhancing tumor)
  brats2021 one folder per case: *_flair, *_t1, *_t1ce, *_t2, *_seg.nii.gz (1 necrosis, 2 edema, 4 enhancing)
  brats2023 one folder per case: *-t2f, *-t1n, *-t1c, *-t2w, *-seg.nii.gz (1 necrosis, 2 edema, 3 enhancing)

Usage:
  python 01_extract.py --data /path/Task01_BrainTumour --format msd --out results_brats --jobs 8
Resumable: each case is written to <out>/cases/<case>.csv; finished cases are skipped.
Outputs to send back: <out>/brats_features.csv.gz (or brats_features_<seq>.csv.gz if > 380 MB) and <out>/qc_images.zip
"""
import argparse, glob, os, sys, time, logging, traceback
import numpy as np, pandas as pd, SimpleITK as sitk
from scipy import ndimage

LABELS = {"msd": dict(et=3, ncr=2, ed=1), "brats2021": dict(et=4, ncr=1, ed=2), "brats2023": dict(et=3, ncr=1, ed=2)}
SIZES = [108, 216, 432, 864, 1728, 3456, 6912]
MSD_CH = {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}
SUFFIX = {"brats2021": {"flair": "_flair", "t1": "_t1", "t1ce": "_t1ce", "t2": "_t2", "seg": "_seg"},
          "brats2023": {"flair": "-t2f", "t1": "-t1n", "t1ce": "-t1c", "t2": "-t2w", "seg": "-seg"}}


def list_cases(data, fmt):
    if fmt == "msd":
        imgs = sorted(glob.glob(os.path.join(data, "imagesTr", "*.nii.gz")))
        imgs = [p for p in imgs if not os.path.basename(p).startswith(".")]
        return [(os.path.basename(p).replace(".nii.gz", ""), p) for p in imgs]
    dirs = sorted(d for d in glob.glob(os.path.join(data, "*")) if os.path.isdir(d))
    return [(os.path.basename(d), d) for d in dirs]


def load_case(case, path, fmt, seqs):
    """Returns {seq: sitk image 3D}, label array (z,y,x), reference image."""
    if fmt == "msd":
        lab = sitk.ReadImage(path.replace("imagesTr", "labelsTr"))
        img4 = sitk.ReadImage(path)
        arr4 = sitk.GetArrayFromImage(img4)            # (4, z, y, x)
        out = {}
        for s in seqs:
            im = sitk.GetImageFromArray(arr4[MSD_CH[s]].astype(np.float32)); im.CopyInformation(lab); out[s] = im
        return out, sitk.GetArrayFromImage(lab).astype(np.int16), lab
    suf = SUFFIX[fmt]
    def f(key):
        c = glob.glob(os.path.join(path, f"*{suf[key]}.nii*"))
        if not c: raise FileNotFoundError(f"{case}: missing {key}")
        return c[0]
    lab = sitk.ReadImage(f("seg"))
    out = {s: sitk.Cast(sitk.ReadImage(f(s)), sitk.sitkFloat32) for s in seqs}
    for s in out: out[s].CopyInformation(lab)
    return out, sitk.GetArrayFromImage(lab).astype(np.int16), lab


def brain_normalise(img):
    """z-score inside the brain (non-zero voxels) x 100. Background is left at the value of 0 after normalisation."""
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    brain = a != 0
    mu, sd = a[brain].mean(), a[brain].std()
    b = np.where(brain, (a - mu) / sd * 100.0, (0 - mu) / sd * 100.0).astype(np.float32)
    o = sitk.GetImageFromArray(b); o.CopyInformation(img)
    return o, brain


def ball_mm(r_mm, spacing_zyx):
    rz, ry, rx = [int(np.ceil(r_mm / s)) for s in spacing_zyx]
    z, y, x = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return (z * spacing_zyx[0]) ** 2 + (y * spacing_zyx[1]) ** 2 + (x * spacing_zyx[2]) ** 2 <= r_mm ** 2


def nearest_n(mask, centre_zyx, n, spacing_zyx):
    """Exactly n voxels of `mask` nearest to centre (compact ROI). None if mask too small."""
    idx = np.argwhere(mask)
    if len(idx) < n: return None
    d = (((idx - np.asarray(centre_zyx)) * np.asarray(spacing_zyx)) ** 2).sum(1)
    sel = idx[np.argpartition(d, n - 1)[:n]]
    out = np.zeros(mask.shape, np.uint8); out[tuple(sel.T)] = 1
    return out


def build_rois(lab, brain, spacing_zyx, fmt, healthy_margin_mm):
    L = LABELS[fmt]
    whole = lab > 0
    rois, info = {}, {}
    if whole.sum() == 0: return rois, info
    for k in ("et", "ncr", "ed"):
        rois[k] = lab == L[k]
    excl = ndimage.binary_dilation(whole, structure=ball_mm(healthy_margin_mm, spacing_zyx))
    healthy = brain & ~excl
    healthy = ndimage.binary_erosion(healthy, structure=ball_mm(2, spacing_zyx))   # away from the brain surface
    cz, cy, cx = ndimage.center_of_mass(whole)
    bx = np.where(brain.any(axis=(0, 1)))[0]; mid = (bx.min() + bx.max()) / 2
    centre = (cz, cy, 2 * mid - cx)                                                # mirror across the x midline
    hz = np.argwhere(healthy)
    if len(hz):
        d = (((hz - np.asarray(centre)) * np.asarray(spacing_zyx)) ** 2).sum(1); centre = tuple(int(v) for v in hz[d.argmin()])
        for n in SIZES:
            m = nearest_n(healthy, centre, n, spacing_zyx)
            if m is not None: rois[f"healthy_{n}"] = m.astype(bool)
    info["centre_healthy"] = centre
    info["whole_voxels"] = int(whole.sum())
    return rois, info


def qc_png(case, imgs, rois, centre, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    seqs = list(imgs)
    tum = rois["et"] | rois["ncr"] | rois["ed"]
    zt = int(np.argmax(tum.sum(axis=(1, 2))))
    fig, axs = plt.subplots(1, len(seqs) + 1, figsize=(4 * (len(seqs) + 1), 4))
    for ax, s in zip(axs, seqs):
        a = sitk.GetArrayFromImage(imgs[s])[zt]; ax.imshow(a, cmap="gray"); ax.set_title(f"{case} {s} z={zt}"); ax.axis("off")
    ov = np.zeros(tum.shape[1:] + (4,))
    for k, c in [("ed", (1, .8, 0, .45)), ("ncr", (.2, .4, 1, .6)), ("et", (1, .1, .1, .6))]:
        ov[rois[k][zt]] = c
    zh = int(centre[0])
    hl = rois.get("healthy_1728")
    axs[-1].imshow(sitk.GetArrayFromImage(imgs[seqs[0]])[zt], cmap="gray"); axs[-1].imshow(ov)
    if hl is not None and hl[zt].any(): axs[-1].contour(hl[zt], colors="lime", linewidths=1)
    axs[-1].set_title("ED yellow, NCR blue, ET red, healthy_1728 green"); axs[-1].axis("off")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(out, f"qc_{case}.png"), dpi=70, bbox_inches="tight", pad_inches=.15); plt.close(fig)


def process(case, path, a):
    if a.jobs > 1: sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)   # one thread per worker
    import radiomics
    from radiomics import featureextractor
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    dst = os.path.join(a.out, "cases", f"{case}.csv")
    if os.path.exists(dst): return case, "skip"
    try:
        t0 = time.time()
        imgs, lab, ref = load_case(case, path, a.format, a.seqs)
        sp = ref.GetSpacing()[::-1]
        first = imgs[a.seqs[0]]
        _, brain = brain_normalise(first)
        rois, info = build_rois(lab, brain, sp, a.format, a.healthy_margin_mm)
        if not rois:
            pd.DataFrame([dict(case=case, status="no_tumor")]).to_csv(dst, index=False); return case, "no_tumor"
        if a.qc_cases and case in a.qc_cases:
            try:
                qc_png(case, {s: brain_normalise(imgs[s])[0] for s in a.seqs}, rois, info["centre_healthy"], os.path.join(a.out, "qc"))
            except Exception:
                pass
        ex = featureextractor.RadiomicsFeatureExtractor(a.params)
        rows = []
        for s in a.seqs:
            im, _ = brain_normalise(imgs[s])
            for name, m in rois.items():
                nv = int(m.sum())
                row = dict(case=case, seq=s, roi=name, n_voxels=nv, n_components=int(ndimage.label(m)[1]) if nv else 0,
                           whole_tumor_voxels=info["whole_voxels"], status="ok")
                if nv < a.min_voxels:
                    row["status"] = "too_small"; rows.append(row); continue
                mk = sitk.GetImageFromArray(m.astype(np.uint8)); mk.CopyInformation(ref)
                try:
                    f = ex.execute(im, mk, label=1)
                    row.update({k: (np.real(v).item() if hasattr(v, "item") else v) for k, v in f.items() if not k.startswith("diagnostics")})
                except Exception as e:
                    row["status"] = f"error: {e}"
                rows.append(row)
        pd.DataFrame(rows).to_csv(dst, index=False, float_format="%.7g")
        return case, f"ok {time.time()-t0:.0f}s"
    except Exception as e:
        pd.DataFrame([dict(case=case, status=f"error: {e}")]).to_csv(dst, index=False)
        return case, "error " + traceback.format_exc(limit=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--format", choices=list(LABELS), default="msd")
    ap.add_argument("--out", default="results_brats"); ap.add_argument("--params", default=os.path.join(os.path.dirname(__file__), "params_brats.yaml"))
    ap.add_argument("--seqs", default="flair,t1ce,t1", help="comma list among flair,t1,t1ce,t2")
    ap.add_argument("--healthy-margin-mm", type=float, default=10.0)
    ap.add_argument("--min-voxels", type=int, default=50)
    ap.add_argument("--n-qc", type=int, default=6, help="number of cases with a QC overlay image")
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(); a.seqs = a.seqs.split(",")
    os.makedirs(os.path.join(a.out, "cases"), exist_ok=True)
    cases = list_cases(a.data, a.format)
    if a.max_cases: cases = cases[:a.max_cases]
    os.makedirs(os.path.join(a.out, "qc"), exist_ok=True)
    a.qc_cases = set(c for c, _ in cases[:: max(1, len(cases) // a.n_qc)][:a.n_qc]) if a.n_qc else set()
    print(f"{len(cases)} cases, sequences {a.seqs}, jobs {a.jobs}", flush=True)
    if a.jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(process)(c, p, a) for c, p in cases)
    else:
        res = []
        for i, (c, p) in enumerate(cases):
            res.append(process(c, p, a)); print(f"[{i+1}/{len(cases)}] {res[-1][0]}: {res[-1][1]}", flush=True)
    parts = [pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(a.out, "cases", "*.csv")))]
    tab = pd.concat(parts, ignore_index=True)
    dst = os.path.join(a.out, "brats_features.csv.gz")
    tab.to_csv(dst, index=False, float_format="%.7g", compression="gzip")
    mb = os.path.getsize(dst) / 2**20
    ok = tab[tab.status == "ok"]
    print(f"\nWritten {dst} ({mb:.0f} MB): {tab.case.nunique()} cases, {len(ok)} ROI extractions ok")
    if mb > 380:   # split by sequence so that every file can be attached (< 400 MB)
        for s, g in tab.groupby("seq"):
            g.to_csv(os.path.join(a.out, f"brats_features_{s}.csv.gz"), index=False, float_format="%.7g", compression="gzip")
        print("File > 380 MB: also written one file per sequence (brats_features_<seq>.csv.gz). Send those instead.")
    print(ok.groupby(["seq", "roi"]).size().unstack(0).to_string())
    bad = tab[tab.status != "ok"]
    if len(bad): print("\nNot ok:\n", bad.groupby("status").size().to_string())
    import zipfile
    with zipfile.ZipFile(os.path.join(a.out, "qc_images.zip"), "w") as z:
        for f in glob.glob(os.path.join(a.out, "qc", "*.png")): z.write(f, os.path.basename(f))
    with open(os.path.join(a.out, "run_settings.txt"), "w") as fh:
        fh.write(str(vars(a)))


if __name__ == "__main__":
    main()
