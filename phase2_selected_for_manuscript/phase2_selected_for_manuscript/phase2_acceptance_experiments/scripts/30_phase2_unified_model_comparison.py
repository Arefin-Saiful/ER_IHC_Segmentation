import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")
PHASE2_ROOT = PROJECT_ROOT / "phase2_acceptance_experiments"

OUT_TABLE_DIR = PHASE2_ROOT / "outputs/tables"
OUT_REPORT_DIR = PHASE2_ROOT / "outputs/reports"
MANUSCRIPT_TABLE_DIR = PHASE2_ROOT / "manuscript_assets/tables"

for d in [OUT_TABLE_DIR, OUT_REPORT_DIR, MANUSCRIPT_TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


MODEL_SPECS = [
    {
        "model": "U-Net 120",
        "source": "phase2",
        "pattern": PHASE2_ROOT / "outputs/metrics/unet_120/phase2_unet120_fullval_fold*_aug-full_base32_crop320/best_metrics.json",
    },
    {
        "model": "Attention U-Net 120",
        "source": "phase2",
        "pattern": PHASE2_ROOT / "outputs/metrics/attention_unet_120/phase2_attention_unet120_fullval_fold*_aug-full_base32_crop320/best_metrics.json",
    },
    {
        "model": "TransUNet-style 120",
        "source": "phase2",
        "pattern": PHASE2_ROOT / "outputs/metrics/transunet_style_120/phase2_transunet_style120_fullval_fold*_aug-full_base32_crop320_tl2_h8/best_metrics.json",
    },
    {
        "model": "Final AMC-Focal ResUNet-DS",
        "source": "main",
        "pattern": PROJECT_ROOT / "outputs/metrics/ablation_components/ablation_amc_focal_tversky_fullval_fold*_aug-full_base32_crop320_amc1_ord0.0_ft0.5/best_metrics.json",
    },
]


NNUNET_RESULTS = (
    PHASE2_ROOT
    / "nnunet_workspace/nnUNet_results/Dataset501_ERIHCSeg/nnUNetTrainer_120epochs__nnUNetPlans__2d"
)

SPLIT_DIR = PROJECT_ROOT / "data/splits"

CLASS_NAMES = {
    0: "C0",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}


def extract_fold_from_path(path):
    text = str(path)
    for fold in range(5):
        if f"fold{fold}" in text or f"fold_{fold}" in text:
            return fold
    return None


def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def load_json_metrics_models():
    rows = []

    for spec in MODEL_SPECS:
        paths = sorted(Path().glob(str(spec["pattern"]))) if not spec["pattern"].is_absolute() else sorted(spec["pattern"].parent.parent.glob(spec["pattern"].parent.name + "/best_metrics.json"))

        # More reliable glob for absolute patterns
        paths = sorted(spec["pattern"].parent.parent.glob(spec["pattern"].parent.name + "/best_metrics.json"))

        if len(paths) == 0:
            print(f"WARNING: no files found for {spec['model']}")
            print(f"Pattern: {spec['pattern']}")
            continue

        for p in paths:
            fold = extract_fold_from_path(p)
            with open(p, "r") as f:
                js = json.load(f)

            row = {
                "model": spec["model"],
                "fold": fold,
                "metrics_path": str(p),
                "best_epoch": safe_float(js.get("epoch", js.get("best_epoch"))),
                "selection_score": safe_float(js.get("selection_score")),
                "mean_dice_no_bg": safe_float(js.get("mean_dice_no_bg")),
                "minority_dice_c2_c3_c4": safe_float(js.get("minority_dice_c2_c3_c4")),
                "mean_iou_no_bg": safe_float(js.get("mean_iou_no_bg")),
                "dice_c1": safe_float(js.get("dice_c1")),
                "dice_c2": safe_float(js.get("dice_c2")),
                "dice_c3": safe_float(js.get("dice_c3")),
                "dice_c4": safe_float(js.get("dice_c4")),
                "iou_c1": safe_float(js.get("iou_c1")),
                "iou_c2": safe_float(js.get("iou_c2")),
                "iou_c3": safe_float(js.get("iou_c3")),
                "iou_c4": safe_float(js.get("iou_c4")),
                "weighted_kappa_fg": safe_float(js.get("weighted_kappa_fg")),
                "ordinal_mae_fg": safe_float(js.get("ordinal_mae_fg")),
            }

            rows.append(row)

    return rows


def quadratic_weighted_kappa_from_confusion(conf):
    conf = conf.astype(np.float64)
    n = conf.shape[0]

    total = conf.sum()
    if total == 0:
        return np.nan

    row_marginals = conf.sum(axis=1)
    col_marginals = conf.sum(axis=0)

    expected = np.outer(row_marginals, col_marginals) / total

    weights = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    observed_score = (weights * conf).sum()
    expected_score = (weights * expected).sum()

    if expected_score == 0:
        return np.nan

    return 1.0 - observed_score / expected_score


def dice_iou_from_confusion(conf):
    metrics = {}

    for c in range(conf.shape[0]):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp

        denom_dice = (2 * tp + fp + fn)
        denom_iou = (tp + fp + fn)

        dice = np.nan if denom_dice == 0 else (2 * tp / denom_dice)
        iou = np.nan if denom_iou == 0 else (tp / denom_iou)

        metrics[f"dice_c{c}"] = float(dice)
        metrics[f"iou_c{c}"] = float(iou)

    return metrics


def load_nnunet_fold_metrics():
    rows = []

    for fold in range(5):
        fold_dir = NNUNET_RESULTS / f"fold_{fold}"
        val_dir = fold_dir / "validation"
        summary_path = val_dir / "summary.json"
        split_path = SPLIT_DIR / f"fold_{fold}_val.csv"

        if not summary_path.exists():
            print(f"WARNING: missing nnU-Net summary for fold {fold}: {summary_path}")
            continue

        if not split_path.exists():
            print(f"WARNING: missing split file for fold {fold}: {split_path}")
            continue

        val_df = pd.read_csv(split_path)

        conf = np.zeros((5, 5), dtype=np.int64)
        fg_gt_values = []
        fg_pred_values = []

        missing_preds = []

        for _, r in val_df.iterrows():
            image_id = int(r["image_id"])
            case = f"ERIHC_{image_id:04d}"
            pred_path = val_dir / f"{case}.png"
            gt_path = Path(r["mask_label_path"])

            if not pred_path.exists():
                missing_preds.append(str(pred_path))
                continue

            pred = np.array(Image.open(pred_path).convert("L")).astype(np.int64)
            gt = np.array(Image.open(gt_path).convert("L")).astype(np.int64)

            if pred.shape != gt.shape:
                raise ValueError(f"Shape mismatch for {case}: pred={pred.shape}, gt={gt.shape}")

            pred = np.clip(pred, 0, 4)
            gt = np.clip(gt, 0, 4)

            flat_gt = gt.reshape(-1)
            flat_pred = pred.reshape(-1)

            for g, p in zip(flat_gt, flat_pred):
                conf[g, p] += 1

            fg = flat_gt > 0
            fg_gt_values.append(flat_gt[fg])
            fg_pred_values.append(flat_pred[fg])

        if missing_preds:
            print(f"WARNING: fold {fold} missing predictions: {len(missing_preds)}")
            print(missing_preds[:5])

        metrics = dice_iou_from_confusion(conf)

        mean_dice_no_bg = float(np.nanmean([metrics[f"dice_c{i}"] for i in [1, 2, 3, 4]]))
        minority_dice = float(np.nanmean([metrics[f"dice_c{i}"] for i in [2, 3, 4]]))
        mean_iou_no_bg = float(np.nanmean([metrics[f"iou_c{i}"] for i in [1, 2, 3, 4]]))

        if fg_gt_values:
            fg_gt_values = np.concatenate(fg_gt_values)
            fg_pred_values = np.concatenate(fg_pred_values)
            fg_conf = np.zeros((4, 4), dtype=np.int64)

            gt_shift = fg_gt_values - 1
            pred_shift = np.clip(fg_pred_values, 1, 4) - 1

            for g, p in zip(gt_shift, pred_shift):
                fg_conf[g, p] += 1

            weighted_kappa_fg = quadratic_weighted_kappa_from_confusion(fg_conf)
            ordinal_mae_fg = float(np.mean(np.abs(fg_gt_values - fg_pred_values)) / 4.0)
        else:
            weighted_kappa_fg = np.nan
            ordinal_mae_fg = np.nan

        selection_score = (
            0.60 * minority_dice
            + 0.30 * mean_dice_no_bg
            + 0.10 * weighted_kappa_fg
        )

        row = {
            "model": "nnU-Net 120",
            "fold": fold,
            "metrics_path": str(summary_path),
            "best_epoch": 120,
            "selection_score": selection_score,
            "mean_dice_no_bg": mean_dice_no_bg,
            "minority_dice_c2_c3_c4": minority_dice,
            "mean_iou_no_bg": mean_iou_no_bg,
            "dice_c1": metrics["dice_c1"],
            "dice_c2": metrics["dice_c2"],
            "dice_c3": metrics["dice_c3"],
            "dice_c4": metrics["dice_c4"],
            "iou_c1": metrics["iou_c1"],
            "iou_c2": metrics["iou_c2"],
            "iou_c3": metrics["iou_c3"],
            "iou_c4": metrics["iou_c4"],
            "weighted_kappa_fg": weighted_kappa_fg,
            "ordinal_mae_fg": ordinal_mae_fg,
        }

        rows.append(row)

    return rows


def mean_sd_string(values, digits=4):
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return ""

    if len(arr) == 1:
        return f"{arr[0]:.{digits}f}"

    return f"{arr.mean():.{digits}f} ± {arr.std(ddof=1):.{digits}f}"


def make_summary(fold_df):
    metric_cols = [
        "selection_score",
        "mean_dice_no_bg",
        "minority_dice_c2_c3_c4",
        "mean_iou_no_bg",
        "dice_c1",
        "dice_c2",
        "dice_c3",
        "dice_c4",
        "iou_c1",
        "iou_c2",
        "iou_c3",
        "iou_c4",
        "weighted_kappa_fg",
        "ordinal_mae_fg",
    ]

    rows = []

    for model, g in fold_df.groupby("model", sort=False):
        row = {
            "model": model,
            "n_folds": int(g["fold"].nunique()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(np.nanmean(g[col].values))
            row[f"{col}_sd"] = float(np.nanstd(g[col].values, ddof=1)) if len(g) > 1 else np.nan
            row[f"{col}_mean_sd"] = mean_sd_string(g[col].values)

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("selection_score_mean", ascending=False).reset_index(drop=True)

    return summary


def make_delta_vs_amcfocal(summary_df):
    ref = summary_df[summary_df["model"] == "Final AMC-Focal ResUNet-DS"]

    if ref.empty:
        print("WARNING: Final AMC-Focal ResUNet-DS not found. Cannot compute deltas.")
        return pd.DataFrame()

    ref = ref.iloc[0]

    cols = [
        "selection_score_mean",
        "mean_dice_no_bg_mean",
        "minority_dice_c2_c3_c4_mean",
        "mean_iou_no_bg_mean",
        "dice_c2_mean",
        "dice_c4_mean",
        "weighted_kappa_fg_mean",
        "ordinal_mae_fg_mean",
    ]

    rows = []

    for _, row in summary_df.iterrows():
        out = {"model": row["model"]}

        for col in cols:
            if col == "ordinal_mae_fg_mean":
                out[f"delta_{col}_vs_amcfocal"] = row[col] - ref[col]
            else:
                out[f"delta_{col}_vs_amcfocal"] = row[col] - ref[col]

        rows.append(out)

    return pd.DataFrame(rows)


def main():
    print("=" * 90)
    print("Phase 2F: Unified model comparison")
    print("=" * 90)

    rows = []
    rows.extend(load_json_metrics_models())
    rows.extend(load_nnunet_fold_metrics())

    fold_df = pd.DataFrame(rows)

    if fold_df.empty:
        raise RuntimeError("No metrics found.")

    fold_df = fold_df.sort_values(["model", "fold"]).reset_index(drop=True)

    summary_df = make_summary(fold_df)
    delta_df = make_delta_vs_amcfocal(summary_df)

    foldwise_path = OUT_TABLE_DIR / "30_phase2_unified_foldwise_metrics.csv"
    summary_path = OUT_TABLE_DIR / "30_phase2_unified_summary_mean_sd.csv"
    delta_path = OUT_TABLE_DIR / "30_phase2_unified_delta_vs_amcfocal.csv"

    manuscript_summary_path = MANUSCRIPT_TABLE_DIR / "30_phase2_unified_summary_mean_sd.csv"
    manuscript_foldwise_path = MANUSCRIPT_TABLE_DIR / "30_phase2_unified_foldwise_metrics.csv"
    manuscript_delta_path = MANUSCRIPT_TABLE_DIR / "30_phase2_unified_delta_vs_amcfocal.csv"

    fold_df.to_csv(foldwise_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    fold_df.to_csv(manuscript_foldwise_path, index=False)
    summary_df.to_csv(manuscript_summary_path, index=False)
    delta_df.to_csv(manuscript_delta_path, index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 2F",
        "models": summary_df["model"].tolist(),
        "n_foldwise_rows": int(len(fold_df)),
        "foldwise_metrics": str(foldwise_path),
        "summary_table": str(summary_path),
        "delta_vs_amcfocal": str(delta_path),
        "manuscript_summary_table": str(manuscript_summary_path),
    }

    report_path = OUT_REPORT_DIR / "30_phase2_unified_model_comparison_summary.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Foldwise metrics:")
    print(fold_df[[
        "model", "fold", "selection_score", "mean_dice_no_bg",
        "minority_dice_c2_c3_c4", "dice_c2", "dice_c4",
        "weighted_kappa_fg", "ordinal_mae_fg"
    ]].to_string(index=False))

    print()
    print("Summary:")
    print(summary_df[[
        "model",
        "n_folds",
        "selection_score_mean_sd",
        "mean_dice_no_bg_mean_sd",
        "minority_dice_c2_c3_c4_mean_sd",
        "mean_iou_no_bg_mean_sd",
        "dice_c2_mean_sd",
        "dice_c4_mean_sd",
        "weighted_kappa_fg_mean_sd",
        "ordinal_mae_fg_mean_sd",
    ]].to_string(index=False))

    print()
    print("Delta vs Final AMC-Focal:")
    print(delta_df.to_string(index=False))

    print()
    print("Saved:")
    print(foldwise_path)
    print(summary_path)
    print(delta_path)
    print(report_path)


if __name__ == "__main__":
    main()
