import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

TABLE_DIR = PROJECT_ROOT / "outputs/tables/results"
FIG_DIR = PROJECT_ROOT / "outputs/figures/results/final_model_comparison"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

MANUSCRIPT_TABLE_DIR = PROJECT_ROOT / "manuscript_assets/tables"
MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "manuscript_assets/figures_results"

MODEL_CONFIGS = {
    "U-Net": {
        "metrics_root": PROJECT_ROOT / "outputs/metrics/unet_baseline",
        "pattern": "unet_baseline_fullval_fold*_aug-full_base32_crop320/best_metrics.json",
    },
    "ResUNet-DS": {
        "metrics_root": PROJECT_ROOT / "outputs/metrics/resunetds_baseline",
        "pattern": "resunetds_baseline_fullval_fold*_aug-full_base32_crop320/best_metrics.json",
    },
    "Proposed": {
        "metrics_root": PROJECT_ROOT / "outputs/metrics/proposed_amc_ordinal",
        "pattern": "proposed_amc_ordinal_fullval_fold*_aug-full_base32_crop320_ord0.1_ft0.5/best_metrics.json",
    },
}

CLASS_NAMES = {
    0: "Background",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}

DISPLAY_NAMES = {
    "mean_dice_no_bg": "Mean Dice (C1-C4)",
    "minority_dice_c2_c3_c4": "Minority Dice (C2-C4)",
    "mean_iou_no_bg": "Mean IoU (C1-C4)",
    "macro_precision_no_bg": "Macro Precision (C1-C4)",
    "macro_recall_no_bg": "Macro Recall (C1-C4)",
    "dice_c1": "Dice C1",
    "dice_c2": "Dice C2",
    "dice_c3": "Dice C3",
    "dice_c4": "Dice C4",
    "iou_c1": "IoU C1",
    "iou_c2": "IoU C2",
    "iou_c3": "IoU C3",
    "iou_c4": "IoU C4",
    "ordinal_mae_fg": "Ordinal MAE",
    "adjacent_error_rate_fg": "Adjacent Error Rate",
    "distant_error_rate_fg": "Distant Error Rate",
    "exact_rate_fg": "Exact Ordinal Agreement",
    "weighted_kappa_fg": "Weighted Kappa",
    "val_loss": "Validation Loss",
}

MAIN_METRICS = [
    "mean_dice_no_bg",
    "minority_dice_c2_c3_c4",
    "mean_iou_no_bg",
    "macro_precision_no_bg",
    "macro_recall_no_bg",
    "weighted_kappa_fg",
    "ordinal_mae_fg",
    "adjacent_error_rate_fg",
    "distant_error_rate_fg",
]

PER_CLASS_METRICS = [
    "dice_c1", "dice_c2", "dice_c3", "dice_c4",
    "iou_c1", "iou_c2", "iou_c3", "iou_c4",
]

MODEL_COLORS = {
    "U-Net": "#2F4858",
    "ResUNet-DS": "#B25D4C",
    "Proposed": "#2E7D5B",
}

CLASS_COLORS = {
    "C1": "#4C9ED9",
    "C2": "#3BAA5B",
    "C3": "#D7A21B",
    "C4": "#A8322D",
}

TEXT_COLOR = "#1F2933"
GRID_COLOR = "#D7DCE2"
PANEL_BG = "#FBFAF7"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#2F3437",
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "figure.facecolor": "white",
        "axes.facecolor": PANEL_BG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.55, color=GRID_COLOR)


def save_figure(fig, name):
    for out_dir in [FIG_DIR, MANUSCRIPT_FIG_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"{name}.png"
        pdf_path = out_dir / f"{name}.pdf"
        fig.savefig(png_path, bbox_inches="tight", dpi=600)
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    plt.close(fig)


def extract_fold(path):
    match = re.search(r"fold(\d+)", str(path))
    if match is None:
        return None
    return int(match.group(1))


def compute_ordinal_metrics_from_confusion(confusion):
    cm = confusion[1:5, 1:5].astype(np.float64)
    total = cm.sum()

    if total <= 0:
        return {
            "ordinal_mae_fg": np.nan,
            "adjacent_error_rate_fg": np.nan,
            "distant_error_rate_fg": np.nan,
            "exact_rate_fg": np.nan,
            "weighted_kappa_fg": np.nan,
        }

    true_idx = np.arange(4).reshape(-1, 1)
    pred_idx = np.arange(4).reshape(1, -1)
    dist = np.abs(true_idx - pred_idx)

    mae = float((dist * cm).sum() / total)
    adjacent = float(((dist == 1) * cm).sum() / total)
    distant = float(((dist >= 2) * cm).sum() / total)
    exact = float(np.trace(cm) / total)

    observed = cm / total
    row_marginal = observed.sum(axis=1, keepdims=True)
    col_marginal = observed.sum(axis=0, keepdims=True)
    expected = row_marginal @ col_marginal

    weights = (dist / 3.0) ** 2
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())

    if denominator <= 1e-12:
        kappa = np.nan
    else:
        kappa = 1.0 - numerator / denominator

    return {
        "ordinal_mae_fg": mae,
        "adjacent_error_rate_fg": adjacent,
        "distant_error_rate_fg": distant,
        "exact_rate_fg": exact,
        "weighted_kappa_fg": float(kappa),
    }


def load_confusion_metrics(metrics_path):
    confusion_path = metrics_path.parent / "best_confusion_matrix.csv"

    if not confusion_path.exists():
        return {}

    cm_df = pd.read_csv(confusion_path, index_col=0)
    confusion = cm_df.values.astype(np.float64)

    return compute_ordinal_metrics_from_confusion(confusion)


def load_model_metrics(model_name, root, pattern):
    paths = sorted(root.glob(pattern))
    rows = []

    for path in paths:
        fold = extract_fold(path)
        if fold is None:
            continue

        with open(path, "r") as f:
            metrics = json.load(f)

        row = {
            "model": model_name,
            "fold": fold,
            "metrics_path": str(path),
        }

        row.update(metrics)

        # Add ordinal metrics from confusion matrix for all models.
        # This keeps U-Net, ResUNet-DS, and Proposed comparable.
        ordinal_from_cm = load_confusion_metrics(path)
        row.update(ordinal_from_cm)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    expected = set(range(5))
    found = set(df["fold"].tolist()) if len(df) else set()
    missing = sorted(expected - found)

    if missing:
        raise RuntimeError(f"{model_name}: missing folds {missing}")

    return df


def load_all_metrics():
    dfs = []
    for model_name, cfg in MODEL_CONFIGS.items():
        df = load_model_metrics(model_name, cfg["metrics_root"], cfg["pattern"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summarize(df):
    metrics = MAIN_METRICS + PER_CLASS_METRICS + ["val_loss"]
    rows = []

    for model_name in ["U-Net", "ResUNet-DS", "Proposed"]:
        sub = df[df["model"] == model_name]

        for metric in metrics:
            if metric not in sub.columns:
                continue

            values = sub[metric].astype(float).values

            rows.append({
                "model": model_name,
                "metric": metric,
                "metric_display": DISPLAY_NAMES.get(metric, metric),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)),
                "median": float(np.nanmedian(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "mean_sd": f"{np.nanmean(values):.4f} ± {np.nanstd(values, ddof=1):.4f}",
            })

    return pd.DataFrame(rows)


def make_compact_table(summary):
    keep = [
        "mean_dice_no_bg",
        "minority_dice_c2_c3_c4",
        "mean_iou_no_bg",
        "macro_precision_no_bg",
        "macro_recall_no_bg",
        "weighted_kappa_fg",
        "ordinal_mae_fg",
        "adjacent_error_rate_fg",
        "distant_error_rate_fg",
        "dice_c1", "dice_c2", "dice_c3", "dice_c4",
        "iou_c1", "iou_c2", "iou_c3", "iou_c4",
    ]

    rows = []
    for metric in keep:
        row = {"Metric": DISPLAY_NAMES.get(metric, metric)}

        means = {}
        for model_name in ["U-Net", "ResUNet-DS", "Proposed"]:
            sub = summary[(summary["model"] == model_name) & (summary["metric"] == metric)]

            if len(sub) == 0:
                row[model_name] = ""
                means[model_name] = np.nan
            else:
                row[model_name] = sub["mean_sd"].iloc[0]
                means[model_name] = float(sub["mean"].iloc[0])

        row["Proposed - ResUNet-DS"] = means["Proposed"] - means["ResUNet-DS"]
        row["Proposed - U-Net"] = means["Proposed"] - means["U-Net"]

        rows.append(row)

    return pd.DataFrame(rows)


def paired_delta_table(df):
    rows = []

    metrics = [
        "mean_dice_no_bg",
        "minority_dice_c2_c3_c4",
        "mean_iou_no_bg",
        "weighted_kappa_fg",
        "ordinal_mae_fg",
        "adjacent_error_rate_fg",
        "distant_error_rate_fg",
        "dice_c1", "dice_c2", "dice_c3", "dice_c4",
    ]

    unet = df[df["model"] == "U-Net"].set_index("fold")
    resunet = df[df["model"] == "ResUNet-DS"].set_index("fold")
    proposed = df[df["model"] == "Proposed"].set_index("fold")

    for fold in range(5):
        row = {"fold": fold}

        for metric in metrics:
            row[f"unet_{metric}"] = float(unet.loc[fold, metric])
            row[f"resunetds_{metric}"] = float(resunet.loc[fold, metric])
            row[f"proposed_{metric}"] = float(proposed.loc[fold, metric])
            row[f"delta_proposed_vs_resunetds_{metric}"] = float(proposed.loc[fold, metric] - resunet.loc[fold, metric])
            row[f"delta_proposed_vs_unet_{metric}"] = float(proposed.loc[fold, metric] - unet.loc[fold, metric])

        rows.append(row)

    return pd.DataFrame(rows)


def figure_main_metrics(df):
    metrics = [
        ("mean_dice_no_bg", "Mean Dice"),
        ("minority_dice_c2_c3_c4", "Minority Dice"),
        ("mean_iou_no_bg", "Mean IoU"),
        ("weighted_kappa_fg", "Weighted Kappa"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.9), facecolor="white")

    x = np.arange(len(metrics))
    width = 0.24

    for idx, model_name in enumerate(["U-Net", "ResUNet-DS", "Proposed"]):
        sub = df[df["model"] == model_name]
        means = []
        stds = []

        for metric, _ in metrics:
            values = sub[metric].astype(float).values
            means.append(np.nanmean(values))
            stds.append(np.nanstd(values, ddof=1))

        offset = (idx - 1) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=MODEL_COLORS[model_name],
            edgecolor="#202428",
            linewidth=0.65,
            label=model_name,
            alpha=0.94,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    save_figure(fig, "fig_final_01_main_model_comparison")


def figure_per_class_dice(df):
    classes = [1, 2, 3, 4]
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(8.2, 4.9), facecolor="white")

    for model_name in ["U-Net", "ResUNet-DS", "Proposed"]:
        sub = df[df["model"] == model_name]

        means = np.array([sub[f"dice_c{c}"].astype(float).mean() for c in classes])
        stds = np.array([sub[f"dice_c{c}"].astype(float).std(ddof=1) for c in classes])

        ax.plot(
            x,
            means,
            marker="o",
            markersize=6.2,
            linewidth=2.2,
            color=MODEL_COLORS[model_name],
            label=model_name,
        )

        ax.fill_between(
            x,
            means - stds,
            means + stds,
            color=MODEL_COLORS[model_name],
            alpha=0.12,
            linewidth=0,
        )

    for idx, c in enumerate(classes):
        ax.scatter(
            [idx],
            [0.04],
            s=105,
            color=CLASS_COLORS[f"C{c}"],
            edgecolor="#202428",
            linewidth=0.6,
            clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in classes])
    ax.set_xlabel("Ordinal foreground class")
    ax.set_ylabel("Dice score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    clean_axis(ax)

    save_figure(fig, "fig_final_02_per_class_dice_profile")


def figure_ordinal_error(df):
    metrics = [
        ("ordinal_mae_fg", "Ordinal MAE"),
        ("adjacent_error_rate_fg", "Adjacent error"),
        ("distant_error_rate_fg", "Distant error"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")

    x = np.arange(len(metrics))
    width = 0.24

    for idx, model_name in enumerate(["U-Net", "ResUNet-DS", "Proposed"]):
        sub = df[df["model"] == model_name]
        means = []
        stds = []

        for metric, _ in metrics:
            values = sub[metric].astype(float).values
            means.append(np.nanmean(values))
            stds.append(np.nanstd(values, ddof=1))

        offset = (idx - 1) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=MODEL_COLORS[model_name],
            edgecolor="#202428",
            linewidth=0.65,
            label=model_name,
            alpha=0.94,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Error rate / ordinal distance")
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)

    save_figure(fig, "fig_final_03_ordinal_error_comparison")


def figure_foldwise_delta(delta):
    metrics = [
        ("delta_proposed_vs_resunetds_mean_dice_no_bg", "Mean Dice"),
        ("delta_proposed_vs_resunetds_minority_dice_c2_c3_c4", "Minority Dice"),
        ("delta_proposed_vs_resunetds_mean_iou_no_bg", "Mean IoU"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor="white")

    x = np.arange(5)
    width = 0.24

    colors = ["#2E7D5B", "#4C9ED9", "#D7A21B"]

    for idx, (metric, label) in enumerate(metrics):
        values = delta[metric].values
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=colors[idx],
            edgecolor="#202428",
            linewidth=0.55,
            label=label,
        )

    ax.axhline(0, color="#202428", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x])
    ax.set_xlabel("Validation fold")
    ax.set_ylabel("Proposed minus ResUNet-DS")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    save_figure(fig, "fig_final_04_foldwise_delta_vs_resunetds")


def main():
    print("=" * 90)
    print("Aggregating final comparison: U-Net vs ResUNet-DS vs Proposed")
    print("=" * 90)

    setup_style()

    for d in [TABLE_DIR, FIG_DIR, REPORT_DIR, MANUSCRIPT_TABLE_DIR, MANUSCRIPT_FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_all_metrics()
    summary = summarize(df)
    compact = make_compact_table(summary)
    delta = paired_delta_table(df)

    foldwise_path = TABLE_DIR / "final_all_models_foldwise_metrics.csv"
    summary_path = TABLE_DIR / "final_all_models_summary_mean_sd.csv"
    compact_path = TABLE_DIR / "final_all_models_compact_manuscript_table.csv"
    delta_path = TABLE_DIR / "final_all_models_foldwise_delta.csv"

    manuscript_summary_path = MANUSCRIPT_TABLE_DIR / "final_all_models_summary_mean_sd.csv"
    manuscript_compact_path = MANUSCRIPT_TABLE_DIR / "final_all_models_compact_manuscript_table.csv"
    manuscript_delta_path = MANUSCRIPT_TABLE_DIR / "final_all_models_foldwise_delta.csv"

    df.to_csv(foldwise_path, index=False)
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    delta.to_csv(delta_path, index=False)

    summary.to_csv(manuscript_summary_path, index=False)
    compact.to_csv(manuscript_compact_path, index=False)
    delta.to_csv(manuscript_delta_path, index=False)

    figure_main_metrics(df)
    figure_per_class_dice(df)
    figure_ordinal_error(df)
    figure_foldwise_delta(delta)

    key_metrics = compact[
        compact["Metric"].isin([
            "Mean Dice (C1-C4)",
            "Minority Dice (C2-C4)",
            "Mean IoU (C1-C4)",
            "Weighted Kappa",
            "Ordinal MAE",
            "Adjacent Error Rate",
            "Distant Error Rate",
            "Dice C1",
            "Dice C2",
            "Dice C3",
            "Dice C4",
        ])
    ].copy()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": ["U-Net", "ResUNet-DS", "Proposed"],
        "num_folds_per_model": 5,
        "foldwise_path": str(foldwise_path),
        "summary_path": str(summary_path),
        "compact_path": str(compact_path),
        "delta_path": str(delta_path),
        "manuscript_summary_path": str(manuscript_summary_path),
        "manuscript_compact_path": str(manuscript_compact_path),
        "manuscript_delta_path": str(manuscript_delta_path),
        "figure_dir": str(FIG_DIR),
        "manuscript_figure_dir": str(MANUSCRIPT_FIG_DIR),
        "key_metrics": key_metrics.to_dict(orient="records"),
    }

    report_path = REPORT_DIR / "05b_final_all_models_comparison_summary.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Compact final comparison:")
    print(compact.to_string(index=False))

    print()
    print("Key result metrics:")
    print(key_metrics.to_string(index=False))

    print()
    print("=" * 90)
    print("Final aggregation completed")
    print("=" * 90)
    print(f"Compact table: {compact_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
