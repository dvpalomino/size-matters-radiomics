"""Synthetic pilot: size dependence of radiomic features in stationary textures.

Stationary Gaussian random-field textures (the same 'tissue' everywhere) are generated; nested,
compact ROIs of exactly N voxels are placed at the same centre; all features are extracted with
the same PyRadiomics configuration as the BraTS study (params_brats.yaml). Because the texture is
statistically uniform, any systematic change of a feature with N can only be caused by size.

Usage: python synthetic_pilot.py <out_dir> [n_rep]
"""
import os, sys, time, logging, warnings
import numpy as np, pandas as pd, SimpleITK as sitk, yaml
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
from radiomics import featureextractor

logging.getLogger("radiomics").setLevel(logging.ERROR); warnings.filterwarnings("ignore")
OUT = sys.argv[1] if len(sys.argv) > 1 else "pilot_out"; os.makedirs(OUT, exist_ok=True)
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
SIZES = [108, 432, 1728, 6912]
TEXTURES = {"fine": 1.5, "coarse": 3.0}       # correlation length of the random field (voxels)
BLOCK = 66                                      # 66^3 block: largest ROI (radius ~12) + 20-voxel padding
_EX = {}

def extractor():
    if "x" not in _EX:
        p = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_brats.yaml")))
        p["setting"].pop("minimumROISize", None)
        _EX["x"] = featureextractor.RadiomicsFeatureExtractor(p)
    return _EX["x"]

_off = np.stack(np.meshgrid(*[np.arange(-20, 21)] * 3, indexing="ij"), -1).reshape(-1, 3)

def nested_masks(center_frac):
    """Compact nested ROIs: the N voxels closest to a sub-voxel centre (ties broken stably)."""
    rel = _off - center_frac
    order = np.argsort(np.linalg.norm(rel, axis=1), kind="stable")
    c = BLOCK // 2
    out = {}
    for n in SIZES:
        m = np.zeros((BLOCK,) * 3, np.uint8); v = _off[order[:n]] + c
        m[v[:, 0], v[:, 1], v[:, 2]] = 1; out[n] = m
    return out

def job(args):
    tex, rep = args
    rng = np.random.default_rng(1000 * rep + (0 if tex == "fine" else 1))
    v = gaussian_filter(rng.standard_normal((BLOCK,) * 3), TEXTURES[tex])
    v = ((v - v.mean()) / v.std() * 100).astype(np.float32)        # z-score x 100, as in the study
    img = sitk.GetImageFromArray(v); img.SetSpacing((1.0, 1.0, 1.0))
    rows = []
    for n, m in nested_masks(rng.uniform(-.5, .5, 3)).items():
        msk = sitk.GetImageFromArray(m); msk.CopyInformation(img)
        r = extractor().execute(img, msk)
        row = {k: float(x) for k, x in r.items() if not k.startswith("diagnostics") and "_shape_" not in k}
        row.update(texture=tex, rep=rep, N=n); rows.append(row)
    return rows

if __name__ == "__main__":
    jobs = [(t, r) for r in range(N_REP) for t in TEXTURES]
    t0 = time.time(); rows = []
    with Pool(int(os.environ.get("JOBS", 2))) as pool:
        for i, rr in enumerate(pool.imap_unordered(job, jobs)):
            rows += rr
            if (i + 1) % 10 == 0: print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/synthetic_features.csv.gz", index=False)
    print("OK", len(rows), f"{time.time()-t0:.0f}s")
