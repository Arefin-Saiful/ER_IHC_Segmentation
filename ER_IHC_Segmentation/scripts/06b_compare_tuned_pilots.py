import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

RESUNET_ROOT = PROJECT_ROOT / "outputs/metrics/resunetds_baseline"
PROP_ROOT = PROJECT_ROOT / "outputs/metrics/proposed_amc_ordinal"

OUT_DIR = PROJECT_ROOT / "outputs/tables/results"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "ResUNet-DS": {
        "root": RESUNET_ROOT,
        "pattern": "resunetds_baseline_fullval_fold{fold}_aug-full_base32_crop320/best_metrics.json",
    },
    "Original proposed": {
        "root": PROP_ROOT,
        "pattern": "proposed_amc_ordinal_fullval_fold{fold}_aug-full_base32_crop320_ord0.1_ft0.5/best_metrics.json",
    },
    "T1 ord0.05_ft0.25_amc0.65": {
        "root": PROP_ROOT,
        "pattern": "proposed_amc_ordinal_fullval_fold{fold}_aug-full_base32_crop320_ord0.05_ft0.25_amcmax0.65_gain0.05_target0.7/best_metrics.json",
    },
    "T2 ord0.05_ft0.00_amc0.65": {
        "root": PROP_ROOT,
        "pattern": "proposed_amc_ordinal_fullval_fold{fold}_aug-full_base32_crop320_ord0.05_ft0.0_amcmax0.65_gain0.05_target0.7/best_metrics.json",
    },
}

FOLDS = [0, 4]

KEY_METRICS = [
    "selection_score",
    "mean_dice_no_bg",
    "minority_dice_c2_c3_c4",
    "mean_iou_no_bg",
    "dice_c1",
    "dice_c2",
    "dice_c3",
    "dice_c4",
    "weighted_kappa_fg",
    "ordinal_mae_fg",
    "adjacent_error_rate_fg",
    "distant_error_rate_fg",
    "val_loss",
]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_ordinal_from_confusion(metrics_path):
    cm_path = metrics_path.parent / "best_confusion_matrix.csv"
    if not cm_path.exists():
        return {}

    cm = pd.read_csv(cm_path, index_col=0).values.astype(float)
    fg = cm[1:5, 1:5]
    total = fg.sum()

    if total <= 0:
        return {}

    true_idx = np.arange(4).reshape(-1, 1)
    pred_idx = np.arange(4).reshape(1, -1)
    dist = np.abs(true_idx - pred_idx)

    observed = fg / total
    row_marginal = observed.sum(axis=1, keepdims=True)
    col_marginal = observed.sum(axis=0, keepdims=True)
    expected = row_marginal @ col_marginal

    weights = (dist / 3.0) ** 2
    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum()

    kappa = np.nan if denominator <= 1e-12 else 1.0 - numerator / denominator

    return {
        "weighted_kappa_fg": float(kappa),
        "ordinal_mae_fg": float((dist * fg).sum() / total),
        "adjacent_error_rate_fg": float(((dist == 1) * fg).sum() / total),
        "distant_error_rate_fg": float(((dist >= 2) * fg).sum() / total),
    }

rows = []

for run_name, cfg in RUNS.items():
    for fold in FOLDS:
        path = cfg["root"] / cfg["pattern"].format(fold=fold)

        if not path.exists():
            print(f"Missing: {run_name} fold {fold}: {path}")
            continue

        metrics = load_json(path)
        metrics.update(compute_ordinal_from_confusion(path))

        row = {
            "run": run_name,
            "fold": fold,
            "metrics_path": str(path),
        }

        for metric in KEY_METRICS:
            row[metric] = metrics.get(metric, np.nan)

        rows.append(row)

df = pd.DataFrame(rows)

foldwise_path = OUT_DIR / "pilot_tuning_fold0_fold4_comparison.csv"
df.to_csv(foldwise_path, index=False)

summary_rows = []
for run_name in df["run"].unique():
    sub = df[df["run"] == run_name]

    row = {
        "run": run_name,
        "num_folds": len(sub),
    }

    for metric in KEY_METRICS:
        values = sub[metric].astype(float).values
        row[f"{metric}_mean"] = float(np.nanmean(values))
        row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
        row[f"{metric}_mean_sd"] = f"{np.nanmean(values):.4f} ± {np.nanstd(values, ddof=1):.4f}" if len(values) > 1 else f"{np.nanmean(values):.4f}"

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary_path = OUT_DIR / "pilot_tuning_fold0_fold4_summary.csv"
summary.to_csv(summary_path, index=False)

resunet = summary[summary["run"] == "ResUNet-DS"].iloc[0]

delta_rows = []
for _, row in summary.iterrows():
    if row["run"] == "ResUNet-DS":
        continue

    delta_rows.append({
        "run": row["run"],
        "delta_mean_dice_vs_resunet": row["mean_dice_no_bg_mean"] - resunet["mean_dice_no_bg_mean"],
        "delta_minority_dice_vs_resunet": row["minority_dice_c2_c3_c4_mean"] - resunet["minority_dice_c2_c3_c4_mean"],
        "delta_mean_iou_vs_resunet": row["mean_iou_no_bg_mean"] - resunet["mean_iou_no_bg_mean"],
        "delta_dice_c2_vs_resunet": row["dice_c2_mean"] - resunet["dice_c2_mean"],
        "delta_dice_c4_vs_resunet": row["dice_c4_mean"] - resunet["dice_c4_mean"],
        "delta_ordinal_mae_vs_resunet": row["ordinal_mae_fg_mean"] - resunet["ordinal_mae_fg_mean"],
        "delta_kappa_vs_resunet": row["weighted_kappa_fg_mean"] - resunet["weighted_kappa_fg_mean"],
    })

delta = pd.DataFrame(delta_rows)

delta_path = OUT_DIR / "pilot_tuning_fold0_fold4_delta_vs_resunet.csv"
delta.to_csv(delta_path, index=False)

report = {
    "foldwise_path": str(foldwise_path),
    "summary_path": str(summary_path),
    "delta_path": str(delta_path),
}

with open(REPORT_DIR / "06b_pilot_tuning_comparison_summary.json", "w") as f:
    json.dump(report, f, indent=4)

print("=" * 90)
print("Pilot fold 0 and fold 4 comparison")
print("=" * 90)

print("\nFoldwise:")
print(df[[
    "run",
    "fold",
    "selection_score",
    "mean_dice_no_bg",
    "minority_dice_c2_c3_c4",
    "mean_iou_no_bg",
    "dice_c2",
    "dice_c4",
    "weighted_kappa_fg",
    "ordinal_mae_fg",
]].to_string(index=False))

print("\nSummary:")
print(summary[[
    "run",
    "mean_dice_no_bg_mean_sd",
    "minority_dice_c2_c3_c4_mean_sd",
    "mean_iou_no_bg_mean_sd",
    "dice_c2_mean_sd",
    "dice_c4_mean_sd",
    "weighted_kappa_fg_mean_sd",
    "ordinal_mae_fg_mean_sd",
]].to_string(index=False))

print("\nDelta vs ResUNet-DS:")
print(delta.to_string(index=False))

print("\nSaved:")
print(foldwise_path)
print(summary_path)
print(delta_path)
