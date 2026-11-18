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
For every case and for FLAIR, T1gd and T1 the script builds the regions of interest and extracts 1,130 features per region (`params_brats.yaml`):
- ET, NCR, ED: the expert labels;
- seven nested healthy-tissue regions of exactly 108–6,912 voxels in the contralateral hemisphere.

Images are z-scored within the brain mask (× 100) before extraction. Output: `results_brats/brats_features.csv.gz` and QC overlays.

## 2. Synthetic pilot (`synthetic_pilot/`)
```
python synthetic_pilot/synthetic_pilot.py pilot_out 100          # about 15 minutes on two cores
python synthetic_pilot/pilot_analysis.py pilot_out results/brats_sizeonly.csv \
       volume_robust_configuration/volume_robust_features.txt out/fig
```
Stationary Gaussian random-field textures (fine and coarse grain) sampled with nested ROIs of 108–6,912 voxels and extracted with
`params_brats.yaml` (Supplementary Section S4, Table S3, Fig. S2). `pilot_sizeonly.csv` and `pilot_summary.json` hold the results.

## 3. Analysis (`analysis/`)
Run from a working folder; outputs go to `out/`.
```
python analysis/brats_analysis.py results_brats/brats_features.csv.gz out
python analysis/brats_figures.py out results_brats/brats_features.csv.gz
python analysis/design_fig2.py      # Figure 1; expects data/qc/qc_BRATS_001.png (QC overlay from step 1),
                                    # a copy of the feature table at data/brats_features.csv.gz and the outputs in out/
python analysis/graphical_abstract.py
```
`brats_analysis.py` computes:
- cross-sectional dependence on volume: Spearman ρ, bootstrap CIs, FDR and scaling exponents;
- the healthy-tissue size-only test (size-only ROC AUC);
- volume-adjusted paired comparisons (ED vs ET, NCR vs ET);
- the simulated feature selection by tertile of volume difference.

The full run takes about 25 minutes on two cores. In the result files the ED-versus-ET comparison is labelled `et_vs_ed` / `et_ed`.

## 4. Volume-robust configuration (`volume_robust_configuration/`)
- `params_volume_robust.yaml`: PyRadiomics parameter file.
- `volume_robust_features.txt`: the 66 features (Supplementary Table S2).
- `select_volume_robust.py`: keeps only those features in a PyRadiomics output table.

## 5. Per-feature results (`results/`)
| File | Content |
|---|---|
| `brats_dependence.csv` | ρ with log volume, CI, q, exponent, category, per sequence and region |
| `brats_sizeonly.csv` | healthy-tissue size-only ROC AUC (between and within patient), size slope |
| `brats_paired.csv` | ED-vs-ET and NCR-vs-ET comparisons, volume-adjusted classes |
| `rho_consistency_matrix.csv` | agreement of volume dependence across region–sequence combinations (Supplementary Fig. S5) |
| `brats_summary.json` | summary statistics quoted in the text |
| `pilot_sizeonly.csv`, `pilot_summary.json` | synthetic pilot results (Supplementary Section S4) |

## License
Code: MIT. Files in `results/` are derived from the Medical Segmentation Decathlon data and are distributed under CC BY-SA 4.0.
