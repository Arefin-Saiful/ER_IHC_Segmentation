import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

METRIC_ROOT = PROJECT_ROOT / "outputs/metrics/unet_baseline"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/results"
FIG_DIR = PROJECT_ROOT / "outputs/figures/results/unet_baseline_summary"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

MANUSCRIPT_TABLE_DIR = PROJECT_ROOT / "manuscript_assets/tables"
MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "manuscript_assets/figures_results"

MODEL_NAME = "U-Net baseline"

CLASS_NAMES = {
    0: "Background",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}

CLASS_COLORS = {
    "C1": "#4C9ED9",
    "C2": "#3BAA5B",
    "C3": "#D7A21B",
    "C4": "#A8322D",
}

MAIN_COLOR = "#2F4858"
SECONDARY_COLOR = "#B25D4C"
GRID_COLOR = "#D7DCE2"
TEXT_COLOR = "#1F2933"
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


def load_fold_metrics():
    paths = sorted(METRIC_ROOT.glob("unet_baseline_fullval_fold*_aug-full_base32_crop320/best_metrics.json"))

    rows = []

    for path in paths:
        match = re.search(r"fold(\d+)", str(path))
        if not match:
            continue

        fold = int(match.group(1))

        with open(path, "r") as f:
            metrics = json.load(f)

        row = {
            "model": MODEL_NAME,
            "fold": fold,
            "metrics_path": str(path),
        }

        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    expected_folds = set(range(5))
    found_folds = set(df["fold"].tolist()) if len(df) > 0 else set()
    missing = sorted(expected_folds - found_folds)

    if missing:
        raise RuntimeError(f"Missing full-validation U-Net folds: {missing}")

    return df


def create_summary(df):
    metric_order = [
        "selection_score",
        "val_loss",
        "mean_dice_no_bg",
        "minority_dice_c2_c3_c4",
        "mean_iou_no_bg",
        "macro_precision_no_bg",
        "macro_recall_no_bg",
        "dice_c1",
        "dice_c2",
        "dice_c3",
        "dice_c4",
        "iou_c1",
        "iou_c2",
        "iou_c3",
        "iou_c4",
        "precision_c1",
        "precision_c2",
        "precision_c3",
        "precision_c4",
        "recall_c1",
        "recall_c2",
        "recall_c3",
        "recall_c4",
    ]

    rows = []

    for metric in metric_order:
        if metric not in df.columns:
            continue

        values = df[metric].astype(float).values

        rows.append({
            "metric": metric,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean_sd": f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}",
        })

    return pd.DataFrame(rows)


def create_compact_manuscript_table(summary):
    name_map = {
        "mean_dice_no_bg": "Mean Dice (C1-C4)",
        "minority_dice_c2_c3_c4": "Minority Dice (C2-C4)",
        "mean_iou_no_bg": "Mean IoU (C1-C4)",
        "macro_precision_no_bg": "Macro Precision (C1-C4)",
        "macro_recall_no_bg": "Macro Recall (C1-C4)",
        "selection_score": "Selection Score",
        "val_loss": "Validation Loss",
        "dice_c1": "Dice C1",
        "dice_c2": "Dice C2",
        "dice_c3": "Dice C3",
        "dice_c4": "Dice C4",
        "iou_c1": "IoU C1",
        "iou_c2": "IoU C2",
        "iou_c3": "IoU C3",
        "iou_c4": "IoU C4",
    }

    keep = list(name_map.keys())

    compact = summary[summary["metric"].isin(keep)].copy()
    compact["Metric"] = compact["metric"].map(name_map)
    compact["U-Net baseline"] = compact["mean_sd"]

    compact = compact[["Metric", "U-Net baseline", "mean", "std", "median", "min", "max"]]

    return compact


def figure_foldwise_performance(df):
    metrics = [
        ("mean_dice_no_bg", "Mean Dice"),
        ("minority_dice_c2_c3_c4", "Minority Dice"),
        ("mean_iou_no_bg", "Mean IoU"),
        ("macro_precision_no_bg", "Macro Precision"),
        ("macro_recall_no_bg", "Macro Recall"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")

    y_positions = np.arange(len(metrics))
    rng = np.random.default_rng(42)

    for i, (metric, label) in enumerate(metrics):
        values = df[metric].astype(float).values
        mean = values.mean()
        std = values.std(ddof=1)

        jitter = rng.normal(0, 0.045, size=len(values))

        ax.scatter(
            values,
            np.full(len(values), i) + jitter,
            s=42,
            color="#FFFFFF",
            edgecolor=MAIN_COLOR,
            linewidth=1.2,
            zorder=3
        )

        ax.plot(
            [mean - std, mean + std],
            [i, i],
            color=SECONDARY_COLOR,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=2
        )

        ax.scatter(
            mean,
            i,
            s=82,
            color=SECONDARY_COLOR,
            edgecolor="#202428",
            linewidth=0.7,
            zorder=4
        )

        ax.text(
            min(mean + std + 0.015, 1.02),
            i,
            f"{mean:.3f}±{std:.3f}",
            va="center",
            ha="left",
            fontsize=8.7,
            color=TEXT_COLOR
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([x[1] for x in metrics])
    ax.set_xlabel("Validation performance across folds")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    clean_axis(ax)

    save_figure(fig, "fig_unet_01_foldwise_performance_profile")


def figure_per_class_profile(df):
    classes = [1, 2, 3, 4]
    x = np.arange(len(classes))

    dice_means = np.array([df[f"dice_c{c}"].astype(float).mean() for c in classes])
    dice_stds = np.array([df[f"dice_c{c}"].astype(float).std(ddof=1) for c in classes])

    iou_means = np.array([df[f"iou_c{c}"].astype(float).mean() for c in classes])
    iou_stds = np.array([df[f"iou_c{c}"].astype(float).std(ddof=1) for c in classes])

    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor="white")

    ax.plot(
        x,
        dice_means,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=MAIN_COLOR,
        label="Dice"
    )
    ax.fill_between(
        x,
        dice_means - dice_stds,
        dice_means + dice_stds,
        color=MAIN_COLOR,
        alpha=0.16,
        linewidth=0
    )

    ax.plot(
        x,
        iou_means,
        marker="s",
        markersize=6,
        linewidth=2.2,
        color=SECONDARY_COLOR,
        label="IoU"
    )
    ax.fill_between(
        x,
        iou_means - iou_stds,
        iou_means + iou_stds,
        color=SECONDARY_COLOR,
        alpha=0.16,
        linewidth=0
    )

    for idx, c in enumerate(classes):
        ax.scatter(
            [idx],
            [dice_means[idx]],
            s=130,
            color=CLASS_COLORS[f"C{c}"],
            edgecolor="#202428",
            linewidth=0.65,
            zorder=5
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in classes])
    ax.set_xlabel("Ordinal foreground class")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    clean_axis(ax)

    save_figure(fig, "fig_unet_02_per_class_dice_iou_profile")


def figure_fold_metric_matrix(df):
    metrics = [
        "mean_dice_no_bg",
        "minority_dice_c2_c3_c4",
        "mean_iou_no_bg",
        "dice_c1",
        "dice_c2",
        "dice_c3",
        "dice_c4",
    ]

    labels = [
        "Mean Dice",
        "Minority Dice",
        "Mean IoU",
        "Dice C1",
        "Dice C2",
        "Dice C3",
        "Dice C4",
    ]

    matrix = df[metrics].astype(float).values

    fig, ax = plt.subplots(figsize=(8.4, 4.5), facecolor="white")

    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([f"Fold {int(f)}" for f in df["fold"]])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Validation fold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111"
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric value")

    save_figure(fig, "fig_unet_03_fold_metric_matrix")


def main():
    print("=" * 90)
    print("Aggregating U-Net full-validation baseline results")
    print("=" * 90)

    setup_style()

    for d in [TABLE_DIR, FIG_DIR, REPORT_DIR, MANUSCRIPT_TABLE_DIR, MANUSCRIPT_FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_fold_metrics()
    summary = create_summary(df)
    compact = create_compact_manuscript_table(summary)

    foldwise_path = TABLE_DIR / "unet_baseline_fullval_foldwise_metrics.csv"
    summary_path = TABLE_DIR / "unet_baseline_fullval_summary_mean_sd.csv"
    compact_path = TABLE_DIR / "unet_baseline_fullval_manuscript_compact_table.csv"

    manuscript_summary_path = MANUSCRIPT_TABLE_DIR / "unet_baseline_fullval_summary_mean_sd.csv"
    manuscript_compact_path = MANUSCRIPT_TABLE_DIR / "unet_baseline_fullval_manuscript_compact_table.csv"

    df.to_csv(foldwise_path, index=False)
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)

    summary.to_csv(manuscript_summary_path, index=False)
    compact.to_csv(manuscript_compact_path, index=False)

    figure_foldwise_performance(df)
    figure_per_class_profile(df)
    figure_fold_metric_matrix(df)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "num_folds": int(len(df)),
        "foldwise_path": str(foldwise_path),
        "summary_path": str(summary_path),
        "compact_table_path": str(compact_path),
        "manuscript_summary_path": str(manuscript_summary_path),
        "manuscript_compact_path": str(manuscript_compact_path),
        "figure_dir": str(FIG_DIR),
        "manuscript_figure_dir": str(MANUSCRIPT_FIG_DIR),
        "main_results": {
            "mean_dice_no_bg": compact[compact["Metric"] == "Mean Dice (C1-C4)"]["U-Net baseline"].iloc[0],
            "minority_dice_c2_c3_c4": compact[compact["Metric"] == "Minority Dice (C2-C4)"]["U-Net baseline"].iloc[0],
            "mean_iou_no_bg": compact[compact["Metric"] == "Mean IoU (C1-C4)"]["U-Net baseline"].iloc[0],
        }
    }

    report_path = REPORT_DIR / "03b_unet_baseline_fullval_aggregate_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Foldwise metrics:")
    print(df[[
        "fold",
        "best_score" if "best_score" in df.columns else "selection_score",
        "mean_dice_no_bg",
        "minority_dice_c2_c3_c4",
        "mean_iou_no_bg",
        "dice_c1",
        "dice_c2",
        "dice_c3",
        "dice_c4"
    ]].to_string(index=False))

    print()
    print("Compact manuscript table:")
    print(compact[["Metric", "U-Net baseline"]].to_string(index=False))

    print()
    print("=" * 90)
    print("Aggregation completed")
    print("=" * 90)
    print(f"Foldwise CSV: {foldwise_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Compact table: {compact_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
