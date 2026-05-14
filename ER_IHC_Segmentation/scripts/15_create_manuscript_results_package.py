from pathlib import Path
from datetime import datetime
import json
import pandas as pd

PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

OUT_DIR = PROJECT_ROOT / "manuscript_assets"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
FIG_RESULT_DIR = OUT_DIR / "figures_results"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COMPACT_TABLE = TABLE_DIR / "deeplabv3_all_models_compact_table.csv"
STATS_TABLE = TABLE_DIR / "all_final_models_key_statistical_tests.csv"
XAI_TABLE = TABLE_DIR / "xai_case_summary.csv"
QUAL_TABLE = TABLE_DIR / "qualitative_selected_cases.csv"

OUT_MD = OUT_DIR / "final_manuscript_results_package.md"
OUT_JSON = REPORT_DIR / "15_final_manuscript_results_package_summary.json"


def read_csv_safe(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def markdown_table(df):
    return df.to_markdown(index=False)


def main():
    compact = read_csv_safe(COMPACT_TABLE)
    stats = read_csv_safe(STATS_TABLE)
    xai = read_csv_safe(XAI_TABLE)
    qual = read_csv_safe(QUAL_TABLE)

    key_metrics = compact[compact["Metric"].isin([
        "Mean Dice (C1-C4)",
        "Minority Dice (C2-C4)",
        "Mean IoU (C1-C4)",
        "Weighted Kappa",
        "Ordinal MAE",
        "Dice C1",
        "Dice C2",
        "Dice C3",
        "Dice C4",
    ])].copy()

    proposed_vs_resunet = stats[
        (stats["comparison"] == "Proposed AMC-Ordinal vs ResUNet-DS")
        & (stats["metric_display"].isin([
            "Mean Dice (C1-C4)",
            "Minority Dice (C2-C4)",
            "Mean IoU (C1-C4)",
            "Dice C2",
            "Dice C4",
            "Weighted Kappa",
            "Ordinal MAE",
        ]))
    ].copy()

    proposed_vs_deeplab = stats[
        (stats["comparison"] == "Proposed AMC-Ordinal vs DeepLabV3-ResNet50")
        & (stats["metric_display"].isin([
            "Mean Dice (C1-C4)",
            "Minority Dice (C2-C4)",
            "Mean IoU (C1-C4)",
            "Dice C2",
            "Dice C4",
        ]))
    ].copy()

    xai_keep = xai[[
        "case_label",
        "fold",
        "image_id",
        "resunet_mean_dice_present_fg",
        "proposed_mean_dice_present_fg",
        "resunet_dice_c2",
        "proposed_dice_c2",
        "resunet_dice_c4",
        "proposed_dice_c4",
        "proposed_mean_prob_gt_c2",
        "proposed_mean_prob_gt_c4",
        "proposed_mean_uncertainty_fg",
        "proposed_mean_uncertainty_error_pixels",
        "proposed_error_rate_fg",
    ]].copy()

    content = f"""# Final Manuscript Results Package

Generated: {datetime.now().isoformat(timespec="seconds")}

## 1. Main Results Table

{markdown_table(key_metrics)}

## 2. Key Statistical Tests: Proposed AMC-Ordinal vs ResUNet-DS

{markdown_table(proposed_vs_resunet[[
    "metric_display",
    "model_a_mean",
    "model_b_mean",
    "mean_difference",
    "bootstrap_95ci_low",
    "bootstrap_95ci_high",
    "wilcoxon_p",
    "paired_ttest_p",
    "paired_cohens_d",
    "direction_score",
]])}

## 3. Key Statistical Tests: Proposed AMC-Ordinal vs DeepLabV3-ResNet50

{markdown_table(proposed_vs_deeplab[[
    "metric_display",
    "model_a_mean",
    "model_b_mean",
    "mean_difference",
    "bootstrap_95ci_low",
    "bootstrap_95ci_high",
    "wilcoxon_p",
    "paired_ttest_p",
    "paired_cohens_d",
    "direction_score",
]])}

## 4. XAI Case Summary

{markdown_table(xai_keep)}

## 5. Final Results Paragraph

Across five-fold validation, the proposed AMC-Ordinal model achieved the highest mean foreground Dice, minority Dice, mean IoU, macro precision, Dice C1, Dice C2, and Dice C3 among the evaluated models. Compared with ResUNet-DS, the proposed model improved mean Dice, minority Dice, and mean IoU by modest margins, while showing its clearest benefit for the under-represented C2 class. In particular, Dice C2 improved consistently across all five folds, indicating that adaptive minority-aware learning improved sensitivity to weakly represented ordinal expression regions. However, the proposed model did not dominate every metric. ResUNet-DS retained slightly better C4 Dice, weighted kappa, and ordinal MAE, suggesting a trade-off between improved rare-class sensitivity and preservation of high-intensity C4 regions.

## 6. Statistical Interpretation Paragraph

The paired statistical analysis showed that the global improvement of the proposed AMC-Ordinal model over ResUNet-DS was modest and not statistically significant for mean Dice, minority Dice, or mean IoU. The strongest statistical signal was observed for Dice C2, where the proposed model improved over ResUNet-DS in all five folds and achieved a positive paired effect. This supports a cautious interpretation: the proposed method provides targeted benefit for the under-represented C2 class rather than broad statistically dominant performance across all segmentation metrics.

## 7. Qualitative Results Paragraph

Qualitative analysis supported the quantitative findings. Representative cases showed that the proposed model better recovered C2 regions that were missed by ResUNet-DS, while C4 trade-off and challenging cases revealed that the proposed model could under-segment high-intensity C4 regions. The external DeepLabV3-ResNet50 baseline provided a useful non-U-Net comparison, but it did not surpass ResUNet-DS or the proposed model in overall foreground segmentation performance.

## 8. XAI / Interpretability Paragraph

The interpretability analysis further clarified the behavior of the proposed model. In the C2-sensitive case, the proposed AMC-Ordinal model achieved substantially higher C2 recovery and assigned high C2 probability to true C2 pixels, supporting the observed improvement in Dice C2. In contrast, the C4 trade-off case showed low C4 probability over true C4 regions, explaining the reduced C4 Dice in selected cases. The challenging case showed higher uncertainty over error pixels than over foreground pixels overall, indicating that the uncertainty map can help localize ambiguous or failure-prone tissue regions.

## 9. Discussion Point

The findings suggest that adaptive minority-aware training and ordinal supervision can improve sensitivity to under-represented ER-IHC expression regions, particularly C2. However, the C4 trade-off indicates that minority-sensitive sampling may shift model attention toward weak or moderate expression classes at the expense of high-intensity class preservation. Future work should therefore investigate multi-objective checkpoint selection, class-adaptive sampling schedules, and uncertainty-guided refinement to jointly improve rare-class detection and ordinal consistency.

## 10. Limitation Statement

This study has several limitations. First, the dataset contains only 220 image-mask pairs, which may limit the statistical power of five-fold comparisons. Second, although the proposed AMC-Ordinal model improved C2 segmentation, the overall improvement over ResUNet-DS was modest and not statistically significant for all global metrics. Third, the proposed method showed a C4 trade-off in selected folds and qualitative cases. Finally, external validation on independent ER-IHC cohorts is required before clinical generalization can be claimed.

## 11. Suggested Figure Captions

### Figure: Main metric comparison
Comparison of foreground segmentation performance across U-Net, ResUNet-DS, Proposed AMC-Ordinal, and DeepLabV3-ResNet50 models using five-fold validation. Bars show mean performance and error bars indicate standard deviation across folds.

### Figure: Per-class Dice comparison
Per-class Dice comparison across ordinal ER-IHC expression classes C1-C4. The proposed AMC-Ordinal model shows its strongest benefit for the under-represented C2 class, while ResUNet-DS retains slightly stronger C4 performance.

### Figure: Qualitative prediction montage
Representative qualitative segmentation examples comparing ResUNet-DS, Proposed AMC-Ordinal, and DeepLabV3-ResNet50. Cases include high-agreement predictions, C2-sensitive improvement, C4 trade-off, external-baseline gap, and challenging segmentation behavior.

### Figure: XAI probability, uncertainty, and ordinal error maps
Interpretability analysis of representative ER-IHC segmentation cases. Each row shows the original image, ground truth, ResUNet-DS prediction, proposed AMC-Ordinal prediction, proposed C2 probability map, proposed C4 probability map, uncertainty map, and ordinal error map. The C2-sensitive case demonstrates improved recovery of under-represented C2 regions, whereas the C4 trade-off and challenging cases reveal residual high-intensity under-segmentation and increased uncertainty in difficult regions.

## 12. Important Manuscript Asset Paths

Main compact table:
`{COMPACT_TABLE}`

Statistical test table:
`{STATS_TABLE}`

Qualitative selected cases:
`{QUAL_TABLE}`

XAI summary table:
`{XAI_TABLE}`

Main metric figure:
`{FIG_RESULT_DIR / "fig_deeplabv3_01_main_metric_comparison.pdf"}`

Per-class Dice figure:
`{FIG_RESULT_DIR / "fig_deeplabv3_02_per_class_dice_comparison.pdf"}`

Qualitative montage:
`{FIG_DIR / "fig_qualitative_01_model_prediction_montage.pdf"}`

XAI montage:
`{FIG_DIR / "fig_xai_01_probability_uncertainty_montage.pdf"}`
"""

    OUT_MD.write_text(content)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_markdown": str(OUT_MD),
        "main_compact_table": str(COMPACT_TABLE),
        "statistics_table": str(STATS_TABLE),
        "xai_table": str(XAI_TABLE),
        "qualitative_table": str(QUAL_TABLE),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=4)

    print("Saved manuscript results package:", OUT_MD)
    print("Saved report:", OUT_JSON)
    print()
    print("Key metrics:")
    print(key_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
