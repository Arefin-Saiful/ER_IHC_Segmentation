````markdown
# ER_IHC_Segmentation

ER_IHC_Segmentation provides scripts, manifests, reports, metrics, and manuscript-ready figures for an ER-IHC breast cancer image segmentation study using AMC-Ordinal, a minority-sensitive and ordinal-aware deep learning framework. Raw data are excluded due to privacy restrictions.

## Project Overview

This repository supports a computational pathology study on pixel-level segmentation of estrogen receptor immunohistochemistry (ER-IHC) images. The task is to segment ER-IHC image regions into five classes:

- Background
- C1
- C2
- C3
- C4

The C1–C4 classes represent ordinal ER-IHC expression categories. This makes the segmentation task challenging because the labels are not only class-imbalanced but also ordered according to expression intensity.

The proposed method, **AMC-Ordinal**, combines a ResUNet-DS segmentation backbone with attention, deep supervision, adaptive minority curriculum learning, focal-Tversky loss, and foreground ordinal Earth Mover’s Distance loss.

## Proposed Method

The proposed AMC-Ordinal framework includes:

- ResUNet-DS backbone
- scSE attention blocks
- Deep supervision
- Adaptive minority curriculum sampling
- Weighted cross-entropy and Dice loss
- Focal-Tversky loss
- Foreground ordinal EMD loss

The method is designed to improve segmentation sensitivity for under-represented ER-IHC classes, especially minority ordinal regions such as C2, C3, and C4.

## Compared Models

The study compares the proposed AMC-Ordinal model with the following segmentation models:

- U-Net
- ResUNet-DS
- DeepLabV3-ResNet50
- Proposed AMC-Ordinal

## Repository Contents

```text
ER_IHC_Segmentation/
│
├── scripts/                 # Training, evaluation, statistical testing, and visualization scripts
├── reports/                 # JSON summaries, logs, and evaluation reports
├── manuscript_assets/        # Manuscript-ready figures and tables
├── data_manifest/            # Dataset manifest and fold information
├── figures/                 # Result figures and qualitative visualizations
├── logs/                    # Training and execution logs
├── requirements.txt          # Required Python packages
├── data_access_note.txt      # Data privacy and access statement
└── README.md
````

## Dataset Availability

The raw ER-IHC image and ground-truth mask dataset is not included in this repository because it is private and subject to institutional, ethical, and data-sharing restrictions.

To support reproducibility, this repository includes shareable artifacts such as:

* Dataset manifest
* Image-mask pairing information
* Fold assignments
* Class-distribution summaries
* Preprocessing reports
* Training scripts
* Evaluation scripts
* Statistical analysis outputs
* Result tables
* Manuscript-ready figures

Access to the raw dataset may be considered upon reasonable request and subject to institutional approval and data-sharing restrictions.

## Experimental Workflow

The overall workflow includes:

1. Dataset preparation and image-mask pairing
2. RGB mask to class-label conversion
3. Quality-control checking
4. Five-fold cross-validation setup
5. Model training using baseline and proposed models
6. Full-patch validation
7. Quantitative evaluation using segmentation and ordinal metrics
8. Statistical comparison across folds
9. Qualitative prediction visualization
10. Probability, uncertainty, and ordinal error map generation

## Evaluation Metrics

The models are evaluated using:

* Per-class Dice score
* Per-class IoU
* Mean foreground Dice
* Minority Dice
* Mean foreground IoU
* Macro precision
* Macro recall
* Weighted kappa
* Ordinal MAE
* Adjacent error rate
* Distant error rate

## Interpretability Outputs

The repository includes scripts and outputs for interpretability analysis, including:

* Class-specific probability maps
* C2 probability maps
* C4 probability maps
* Softmax entropy uncertainty maps
* Ordinal error maps
* Qualitative prediction montages

These outputs help explain model behavior, especially the improvement in minority C2 segmentation and the trade-off observed in some C4 regions.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ER_IHC_Segmentation.git
cd ER_IHC_Segmentation
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run dataset preparation:

```bash
python scripts/01_prepare_dataset_manifest.py
```

Run fold creation and exploratory analysis:

```bash
python scripts/02_eda_and_create_folds.py
```

Train baseline U-Net:

```bash
python scripts/03_train_unet_baseline.py
```

Train ResUNet-DS baseline:

```bash
python scripts/04_train_resunetds_baseline.py
```

Train proposed AMC-Ordinal model:

```bash
python scripts/05_train_proposed_amc_ordinal.py
```

Run statistical testing:

```bash
python scripts/07_statistical_testing_final_models.py
```

Generate qualitative figures:

```bash
python scripts/12_generate_qualitative_prediction_figures.py
```

Generate interpretability maps:

```bash
python scripts/14_generate_xai_probability_uncertainty_maps.py
```

Note: The scripts require access to the private ER-IHC dataset in the expected local directory structure. The raw dataset is not provided in this repository.

## Manuscript Figures

This repository includes manuscript-ready figures for:

* Dataset and annotation overview
* Overall methodology workflow
* Proposed AMC-Ordinal architecture
* Model performance comparison
* Per-class Dice comparison
* Qualitative segmentation examples
* Probability, uncertainty, and ordinal error maps

## Reproducibility Note

Because the raw dataset is private, full end-to-end reproduction requires authorized access to the ER-IHC image-mask pairs. However, the repository provides the full computational workflow, experimental scripts, fold-level outputs, statistical summaries, and manuscript-ready artifacts to support transparent review and verification.

## Data Privacy Statement

No private raw image data or patient-identifiable information is included in this repository. Only derived, shareable research artifacts are provided.

## Citation

If you use this repository or refer to this work, please cite the associated manuscript once available.

```bibtex
@article{ER_IHC_AMC_Ordinal,
  title={Minority-Sensitive and Ordinal-Aware Deep Learning for ER-IHC Image Segmentation with Uncertainty-Guided Error Interpretation},
  author={Author names to be added},
  journal={Computerized Medical Imaging and Graphics},
  year={2026}
}
```

## License

This repository is released for academic and research use. Please check the license file for details.

## Contact

For questions about the code, reproducibility package, or data access restrictions, please contact the corresponding author listed in the manuscript.

```
```
