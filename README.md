# Cross-Subject Motor Imagery EEG Classification

**A Systematic Study on Cross-Subject Motor Imagery EEG Classification: From Data Alignment to Ensemble Learning**

*Yunuo He¹*

¹ Johns Hopkins University; 

📄 Accepted at **ICBET 2026**

---

## Overview

This repository contains the code for our systematic ablation study on cross-subject motor imagery (MI) EEG classification. Starting from a baseline EEGNet, we progressively integrate six optimization strategies and evaluate their individual and combined contributions under a leave-one-subject-out (LOSO) protocol on the BCI Competition IV Dataset 2a.

### Key Results

| Configuration | Accuracy (%) | Δ |
|---|---|---|
| EEGNet baseline | 40.9 ± 11.4 | — |
| + Euclidean Alignment | 44.6 ± 13.2 | +3.7 |
| + Data Augmentation (4×) | 48.2 ± 16.6 | +3.6 |
| + Center Loss + MMD | 52.8 ± 13.4 | +4.6 |
| + Temporal Attention | 54.5 ± 15.5 | +1.7 |
| + Ensemble (5 seeds) | **57.4 ± 16.7** | +2.9 |

**+14.3% absolute improvement** over the vanilla EEGNet baseline.

---

## Repository Structure

```
BCI/
├── README.md
├── requirements.txt
├── EEGNet_cross_subject.py          # Cross-subject EEGNet (early version)
├── EEGNet_within_subject.py         # Within-subject EEGNet
├── LMDA_Net_cross_subject.py        # LMDA-Net cross-subject
├── LMDA_Net_within_subject.py       # LMDA-Net within-subject
├── Test.py                          # Testing utilities
├── true_labels/                     # Ground truth labels for evaluation
├── ablation_study/                  # 🔬 Paper ablation scripts
│   ├── EEGNet_baseline.py           # Step 0: Vanilla EEGNet (40.9%)
│   ├── EEGNet_EA.py                 # Step 1: + Euclidean Alignment (44.6%)
│   ├── EEGNet_aug.py                # Step 2: + Data Augmentation (48.2%)
│   ├── EEGNet_EA_aug8x_v2_center_mmd.py          # Step 3: + Center Loss + MMD (52.8%)
│   ├── EEGNet_EA_aug8x_v2_center_mmd_temporal_attn.py  # Step 4: + Temporal Attention (54.5%)
│   ├── EEGNet_temporal_attn_aug8x.py              # Step 5: Full pipeline + Ensemble (55.2%)
│   └── LMDA_baseline.py             # LMDA-Net baseline for comparison (41.4%)
└── paper/
    ├── gen_figures.py               # Script to generate paper figures
    └── figures/                     # Paper figures (PDF + PNG)
```

---

## Dataset

We use the [BCI Competition IV Dataset 2a](https://www.bbci.de/competition/iv/#dataset2a):
- **9 subjects**, 22 EEG channels, 250 Hz sampling rate
- **4 MI classes**: left hand, right hand, feet, tongue
- **288 trials** per subject (144 training + 144 evaluation)

Download the `.gdf` files and place them in a `BCICIV_2a_gdf/` directory.

---

## Quick Start

### Installation

```bash
git clone https://github.com/Jacob-H12/BCI.git
cd BCI
pip install -r requirements.txt
```

### Run the Full Pipeline (Best Model)

```bash
cd ablation_study
python EEGNet_temporal_attn_aug8x.py
```

This runs 5-seed × 9-fold LOSO evaluation with all optimizations enabled.

### Reproduce Ablation Study

Run each script in `ablation_study/` sequentially to reproduce Table I in the paper:

```bash
cd ablation_study
python EEGNet_baseline.py                              # Step 0: 40.9%
python EEGNet_EA.py                                     # Step 1: 44.6%
python EEGNet_aug.py                                    # Step 2: 48.2%
python EEGNet_EA_aug8x_v2_center_mmd.py                # Step 3: 52.8%
python EEGNet_EA_aug8x_v2_center_mmd_temporal_attn.py  # Step 4: 54.5%
python EEGNet_temporal_attn_aug8x.py                   # Step 5: 57.4%
```

---

## Methods

- **Euclidean Alignment (EA)**: Covariance-level domain adaptation across subjects
- **Data Augmentation**: Gaussian noise, temporal shifting, channel dropout (4× per trial)
- **Center Loss**: Intra-class feature compactness regularization (λ=0.03)
- **MMD Loss**: Cross-domain feature alignment with Gaussian kernel (λ=0.1)
- **Temporal Self-Attention**: Multi-head attention (H=2) over temporal features
- **Ensemble**: 5-seed soft voting aggregation

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{he2026crosssubject,
  title={A Systematic Study on Cross-Subject Motor Imagery EEG Classification: From Data Alignment to Ensemble Learning},
  author={He, Yunuo and Wu, Jixian},
  booktitle={Proceedings of the International Conference on Biomedical Engineering and Technology (ICBET)},
  year={2026}
}
```

---

## License

This project is for academic research purposes.
