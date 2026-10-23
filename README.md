# Supplementary code and results

**Size matters in radiomics: standardized features are reproducible but not yet comparable across region-of-interest sizes**

## Data
Medical Segmentation Decathlon, Task01_BrainTumour (484 training cases, CC BY-SA 4.0):
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar (MD5 240a19d752f0d9e9101544901065d872)

## 1. Feature extraction (`extraction/`)
```
pip install -r extraction/requirements.txt      # PyRadiomics 3.1.0 (install from the GitHub tag v3.1.0 if the PyPI sdist fails)
python extraction/01_extract.py --data /path/Task01_BrainTumour --format msd --out results_brats --jobs 8
```
For every case and for FLAIR, T1gd and T1 the script builds the regions of interest and extracts 1 130 features per region (`params_brats.yaml`):
- ET, NCR, ED: the expert labels;
- seven nested healthy-tissue regions of exactly 108–6 912 voxels in the contralateral hemisphere.

Images are z-scored within the brain mask (× 100) before extraction. Output: `results_brats/brats_features.csv.gz` and QC overlays.

## 2. Analysis (`analysis/`)
Run from a working folder; outputs go to `out/`.
```
python analysis/brats_analysis.py results_brats/brats_features.csv.gz out
python analysis/brats_figures.py out results_brats/brats_features.csv.gz
python analysis/design_fig2.py      # Figure 1; expects data/qc/qc_BRATS_001.png (QC overlay from step 1),
                                    # a copy of the feature table at data/brats_features.csv.gz and the outputs in out/
```
`brats_analysis.py` computes:
- cross-sectional dependence on volume: Spearman ρ, bootstrap CIs, FDR and scaling exponents;
- the healthy-tissue size-only test;
- volume-adjusted paired comparisons (ET vs ED, NCR vs ET);
- the simulated feature selection by tertile of volume difference.

The full run takes about 25 minutes on two cores.

## 3. Volume-robust configuration (`volume_robust_configuration/`)
- `params_volume_robust.yaml`: PyRadiomics parameter file.
- `volume_robust_features.txt`: the 66 features (Supplementary Table S5).
- `select_volume_robust.py`: keeps only those features in a PyRadiomics output table.

## 4. Per-feature results (`results/`)
| File | Content |
|---|---|
| `brats_dependence.csv` | ρ with log volume, CI, q, exponent, category, per sequence and region |
| `brats_sizeonly.csv` | healthy-tissue size-only AUC (between and within patient), scaling slope |
| `brats_paired.csv` | ET-vs-ED and NCR-vs-ET comparisons, volume-adjusted classes |
| `rho_consistency_matrix.csv` | agreement of volume dependence across region–sequence combinations (Supplementary Fig. S3) |
| `brats_summary.json` | summary statistics quoted in the text |

## License
Code: MIT. Files in `results/` are derived from the Medical Segmentation Decathlon data and are distributed under CC BY-SA 4.0.
