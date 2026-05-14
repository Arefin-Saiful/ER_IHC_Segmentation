import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, ttest_rel
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

FOLDWISE_PATH = PROJECT_ROOT / "outputs/tables/results/final_all_models_foldwise_metrics.csv"

OUT_DIR = PROJECT_ROOT / "outputs/tables/statistics"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"
MANUSCRIPT_TABLE_DIR = PROJECT_ROOT / "manuscript_assets/tables"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

COMPARISONS = [
    ("Proposed", "ResUNet-DS"),
    ("Proposed", "U-Net"),
    ("ResUNet-DS", "U-Net"),
]

METRICS = [
    "mean_dice_no_bg",
    "minority_dice_c2_c3_c4",
    "mean_iou_no_bg",
    "macro_precision_no_bg",
    "macro_recall_no_bg",
    "weighted_kappa_fg",
    "ordinal_mae_fg",
    "adjacent_error_rate_fg",
    "distant_error_rate_fg",
    "dice_c1",
    "dice_c2",
    "dice_c3",
    "dice_c4",
    "iou_c1",
    "iou_c2",
    "iou_c3",
    "iou_c4",
]

DISPLAY_NAMES = {
    "mean_dice_no_bg": "Mean Dice (C1-C4)",
    "minority_dice_c2_c3_c4": "Minority Dice (C2-C4)",
    "mean_iou_no_bg": "Mean IoU (C1-C4)",
    "macro_precision_no_bg": "Macro Precision (C1-C4)",
    "macro_recall_no_bg": "Macro Recall (C1-C4)",
    "weighted_kappa_fg": "Weighted Kappa",
    "ordinal_mae_fg": "Ordinal MAE",
    "adjacent_error_rate_fg": "Adjacent Error Rate",
    "distant_error_rate_fg": "Distant Error Rate",
    "dice_c1": "Dice C1",
    "dice_c2": "Dice C2",
    "dice_c3": "Dice C3",
    "dice_c4": "Dice C4",
    "iou_c1": "IoU C1",
    "iou_c2": "IoU C2",
    "iou_c3": "IoU C3",
    "iou_c4": "IoU C4",
}


def bootstrap_ci(values, n_boot=10000, seed=42, ci=95):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (100 - ci) / 2
    low = np.percentile(boot_means, alpha)
    high = np.percentile(boot_means, 100 - alpha)

    return float(low), float(high)


def cohens_d_paired(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) < 2:
        return np.nan

    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan

    return float(np.mean(diff) / sd)


def cliffs_direction(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return np.nan

    return float(np.mean(diff > 0) - np.mean(diff < 0))


def safe_wilcoxon(a, b):
    if not SCIPY_AVAILABLE:
        return np.nan

    diff = np.asarray(a) - np.asarray(b)

    if np.allclose(diff, 0):
        return 1.0

    try:
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def safe_ttest(a, b):
    if not SCIPY_AVAILABLE:
        return np.nan

    try:
        return float(ttest_rel(a, b).pvalue)
    except Exception:
        return np.nan


def main():
    print("=" * 90)
    print("Final model statistical testing")
    print("=" * 90)

    if not FOLDWISE_PATH.exists():
        raise FileNotFoundError(f"Missing foldwise metrics: {FOLDWISE_PATH}")

    df = pd.read_csv(FOLDWISE_PATH)

    rows = []

    for model_a, model_b in COMPARISONS:
        a_df = df[df["model"] == model_a].set_index("fold")
        b_df = df[df["model"] == model_b].set_index("fold")

        common_folds = sorted(set(a_df.index).intersection(set(b_df.index)))

        for metric in METRICS:
            if metric not in a_df.columns or metric not in b_df.columns:
                continue

            a = a_df.loc[common_folds, metric].astype(float).values
            b = b_df.loc[common_folds, metric].astype(float).values
            diff = a - b

            ci_low, ci_high = bootstrap_ci(diff, n_boot=10000, seed=42)

            row = {
                "comparison": f"{model_a} vs {model_b}",
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "metric_display": DISPLAY_NAMES.get(metric, metric),
                "n_folds": len(common_folds),
                "model_a_mean": float(np.nanmean(a)),
                "model_a_std": float(np.nanstd(a, ddof=1)),
                "model_b_mean": float(np.nanmean(b)),
                "model_b_std": float(np.nanstd(b, ddof=1)),
                "mean_difference": float(np.nanmean(diff)),
                "std_difference": float(np.nanstd(diff, ddof=1)),
                "median_difference": float(np.nanmedian(diff)),
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "wilcoxon_p": safe_wilcoxon(a, b),
                "paired_ttest_p": safe_ttest(a, b),
                "paired_cohens_d": cohens_d_paired(diff),
                "direction_score": cliffs_direction(diff),
                "fold_differences": "; ".join([f"fold{fold}:{d:.5f}" for fold, d in zip(common_folds, diff)]),
            }

            rows.append(row)

    stats = pd.DataFrame(rows)

    stats_path = OUT_DIR / "final_model_statistical_tests.csv"
    manuscript_stats_path = MANUSCRIPT_TABLE_DIR / "final_model_statistical_tests.csv"

    stats.to_csv(stats_path, index=False)
    stats.to_csv(manuscript_stats_path, index=False)

    key = stats[
        (stats["comparison"] == "Proposed vs ResUNet-DS")
        & (stats["metric"].isin([
            "mean_dice_no_bg",
            "minority_dice_c2_c3_c4",
            "mean_iou_no_bg",
            "dice_c2",
            "dice_c4",
            "weighted_kappa_fg",
            "ordinal_mae_fg",
        ]))
    ].copy()

    key_path = OUT_DIR / "final_model_key_statistical_tests.csv"
    manuscript_key_path = MANUSCRIPT_TABLE_DIR / "final_model_key_statistical_tests.csv"

    key.to_csv(key_path, index=False)
    key.to_csv(manuscript_key_path, index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "foldwise_path": str(FOLDWISE_PATH),
        "stats_path": str(stats_path),
        "key_stats_path": str(key_path),
        "manuscript_stats_path": str(manuscript_stats_path),
        "manuscript_key_path": str(manuscript_key_path),
        "scipy_available": SCIPY_AVAILABLE,
    }

    report_path = REPORT_DIR / "07_final_model_statistical_testing_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Key statistical tests: Proposed vs ResUNet-DS")
    print(key[[
        "metric_display",
        "model_a_mean",
        "model_b_mean",
        "mean_difference",
        "bootstrap_95ci_low",
        "bootstrap_95ci_high",
        "wilcoxon_p",
        "paired_cohens_d",
        "direction_score",
        "fold_differences",
    ]].to_string(index=False))

    print()
    print("=" * 90)
    print("Statistical testing completed")
    print("=" * 90)
    print(f"Saved full statistics: {stats_path}")
    print(f"Saved key statistics: {key_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
