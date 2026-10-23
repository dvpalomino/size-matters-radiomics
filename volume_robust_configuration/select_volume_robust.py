"""Keep only the 66 volume-robust features (plus non-feature columns) from a PyRadiomics output table.
Usage: python select_volume_robust.py <features.csv> <out.csv>"""
import os, sys
import pandas as pd

src, dst = sys.argv[1:3]
keep = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "volume_robust_features.txt")) if l.strip()]
d = pd.read_csv(src, low_memory=False)
feat_cols = [c for c in d.columns if c.startswith(("original_", "log-sigma", "wavelet"))]
other = [c for c in d.columns if c not in feat_cols and not c.startswith("diagnostics")]
missing = [k for k in keep if k not in d.columns]
if missing: print(f"Warning: {len(missing)} volume-robust features not found, e.g. {missing[:3]}")
d[other + [k for k in keep if k in d.columns]].to_csv(dst, index=False)
print(f"Written {dst}: {sum(k in d.columns for k in keep)} volume-robust features")
