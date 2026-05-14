# Final Manuscript Results Package

Generated: 2026-05-14T04:38:30

## 1. Main Results Table

| Metric                | U-Net           | ResUNet-DS      | Proposed AMC-Ordinal   | DeepLabV3-ResNet50   |
|:----------------------|:----------------|:----------------|:-----------------------|:---------------------|
| Mean Dice (C1-C4)     | 0.7231 ± 0.0120 | 0.7366 ± 0.0083 | 0.7406 ± 0.0151        | 0.7281 ± 0.0065      |
| Minority Dice (C2-C4) | 0.6819 ± 0.0156 | 0.6980 ± 0.0113 | 0.7016 ± 0.0197        | 0.6889 ± 0.0067      |
| Mean IoU (C1-C4)      | 0.5760 ± 0.0137 | 0.5920 ± 0.0094 | 0.5964 ± 0.0180        | 0.5813 ± 0.0079      |
| Weighted Kappa        | 0.9732 ± 0.0023 | 0.9735 ± 0.0023 | 0.9731 ± 0.0009        | 0.9731 ± 0.0037      |
| Ordinal MAE           | 0.0498 ± 0.0045 | 0.0482 ± 0.0029 | 0.0490 ± 0.0023        | 0.0495 ± 0.0061      |
| Dice C1               | 0.8464 ± 0.0071 | 0.8526 ± 0.0022 | 0.8578 ± 0.0045        | 0.8456 ± 0.0066      |
| Dice C2               | 0.5685 ± 0.0258 | 0.5903 ± 0.0285 | 0.6066 ± 0.0274        | 0.5846 ± 0.0314      |
| Dice C3               | 0.7161 ± 0.0148 | 0.7351 ± 0.0179 | 0.7373 ± 0.0255        | 0.7198 ± 0.0132      |
| Dice C4               | 0.7612 ± 0.0195 | 0.7685 ± 0.0195 | 0.7609 ± 0.0210        | 0.7622 ± 0.0320      |

## 2. Key Statistical Tests: Proposed AMC-Ordinal vs ResUNet-DS

| metric_display        |   model_a_mean |   model_b_mean |   mean_difference |   bootstrap_95ci_low |   bootstrap_95ci_high |   wilcoxon_p |   paired_ttest_p |   paired_cohens_d |   direction_score |
|:----------------------|---------------:|---------------:|------------------:|---------------------:|----------------------:|-------------:|-----------------:|------------------:|------------------:|
| Mean Dice (C1-C4)     |      0.74064   |      0.736629  |       0.00401109  |         -0.00305205  |           0.0108306   |       0.4375 |        0.372965  |          0.44819  |               0.2 |
| Minority Dice (C2-C4) |      0.701596  |      0.697964  |       0.003632    |         -0.0061688   |           0.0123156   |       0.625  |        0.525986  |          0.310282 |               0.2 |
| Mean IoU (C1-C4)      |      0.596407  |      0.592017  |       0.00439012  |         -0.00557093  |           0.0135264   |       0.4375 |        0.466174  |          0.359808 |               0.2 |
| Weighted Kappa        |      0.973138  |      0.973495  |      -0.00035651  |         -0.00201556  |           0.000728325 |       1      |        0.689306  |         -0.19233  |               0.2 |
| Ordinal MAE           |      0.0490053 |      0.0482056 |       0.000799724 |         -0.000926545 |           0.00334981  |       0.8125 |        0.558507  |          0.285064 |              -0.2 |
| Dice C2               |      0.606561  |      0.590321  |       0.0162395   |          0.00719786  |           0.0280688   |       0.0625 |        0.0469062 |          1.26972  |               1   |
| Dice C4               |      0.760893  |      0.768476  |      -0.00758368  |         -0.0339621   |           0.0123905   |       0.8125 |        0.593084  |         -0.259334 |              -0.2 |

## 3. Key Statistical Tests: Proposed AMC-Ordinal vs DeepLabV3-ResNet50

| metric_display        |   model_a_mean |   model_b_mean |   mean_difference |   bootstrap_95ci_low |   bootstrap_95ci_high |   wilcoxon_p |   paired_ttest_p |   paired_cohens_d |   direction_score |
|:----------------------|---------------:|---------------:|------------------:|---------------------:|----------------------:|-------------:|-----------------:|------------------:|------------------:|
| Mean Dice (C1-C4)     |       0.74064  |       0.728057 |        0.0125832  |          0.000701634 |             0.0235483 |       0.1875 |         0.124827 |         0.866189  |               0.6 |
| Minority Dice (C2-C4) |       0.701596 |       0.688866 |        0.01273    |         -0.00352323  |             0.0274246 |       0.1875 |         0.220469 |         0.648792  |               0.6 |
| Mean IoU (C1-C4)      |       0.596407 |       0.581299 |        0.015108   |         -0.000855431 |             0.0284667 |       0.1875 |         0.14088  |         0.819351  |               0.6 |
| Dice C2               |       0.606561 |       0.584603 |        0.0219575  |         -0.00306337  |             0.0453015 |       0.3125 |         0.187936 |         0.709255  |               0.2 |
| Dice C4               |       0.760893 |       0.762172 |       -0.00127918 |         -0.0369235   |             0.0284155 |       1      |         0.947845 |        -0.0311308 |               0.2 |

## 4. XAI Case Summary

| case_label        |   fold |   image_id |   resunet_mean_dice_present_fg |   proposed_mean_dice_present_fg |   resunet_dice_c2 |   proposed_dice_c2 |   resunet_dice_c4 |   proposed_dice_c4 |   proposed_mean_prob_gt_c2 |   proposed_mean_prob_gt_c4 |   proposed_mean_uncertainty_fg |   proposed_mean_uncertainty_error_pixels |   proposed_error_rate_fg |
|:------------------|-------:|-----------:|-------------------------------:|--------------------------------:|------------------:|-------------------:|------------------:|-------------------:|---------------------------:|---------------------------:|-------------------------------:|-----------------------------------------:|-------------------------:|
| C2-sensitive gain |      1 |        102 |                       0.443829 |                        0.748434 |                 0 |           0.589013 |        nan        |         nan        |                   0.947976 |                 nan        |                       0.136388 |                                 0.323356 |                0.0787946 |
| C4 trade-off      |      4 |        125 |                       0.492976 |                        0.438933 |                 0 |           0        |          0.566038 |           0.195105 |                 nan        |                   0.223318 |                       0.230901 |                                 0.371972 |                0.207346  |
| Challenging case  |      3 |         62 |                       0.384037 |                        0.330676 |                 0 |           0        |          0.750267 |           0.595632 |                 nan        |                   0.482074 |                       0.260186 |                                 0.395334 |                0.245115  |

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
`/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/manuscript_assets/tables/deeplabv3_all_models_compact_table.csv`

Statistical test table:
`/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/manuscript_assets/tables/all_final_models_key_statistical_tests.csv`

Qualitative selected cases:
`/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/manuscript_assets/tables/qualitative_selected_cases.csv`

XAI summary table:
`/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/manuscript_assets/tables/xai_case_summary.csv`

Main metric figure:
`/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/manuscript_assets/figures_results/fig_deeplabv3_01_main_metric_comparison.pdf`

Per-class Dice figure:
`/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/manuscript_assets/figures_results/fig_deeplabv3_02_per_class_dice_comparison.pdf`

Qualitative montage:
`/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/manuscript_assets/figures/fig_qualitative_01_model_prediction_montage.pdf`

XAI montage:
`/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/manuscript_assets/figures/fig_xai_01_probability_uncertainty_montage.pdf`
