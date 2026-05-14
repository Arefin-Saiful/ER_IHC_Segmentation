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

UNET_METRIC_ROOT = PROJECT_ROOT / "outputs/metrics/unet_baseline"
RESUNET_METRIC_ROOT = PROJECT_ROOT / "outputs/metrics/resunetds_baseline"

TABLE_DIR = PROJECT_ROOT / "outputs/tables/results"
FIG_DIR = PROJECT_ROOT / "outputs/figures/results/baseline_comparison"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

MANUSCRIPT_TABLE_DIR = PROJECT_ROOT / "manuscript_assets/tables"
MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "manuscript_assets/figures_results"

MODEL_PATTERNS = {
    "U-Net": {
        "root": UNET_METRIC_ROOT,
        "pattern": "unet_baseline_fullval_fold*_aug-full_base32_crop320/best_metrics.json",
    },
    "ResUNet-DS": {
        "root": RESUNET_METRIC_ROOT,
        "pattern": "resunetds_baseline_fullval_fold*_aug-full_base32_crop320/best_metrics.json",
    },
}

METRICS_MAIN = [
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

DISPLAY_NAMES = {
    "selection_score": "Selection Score",
    "val_loss": "Validation Loss",
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
    "precision_c1": "Precision C1",
    "precision_c2": "Precision C2",
    "precision_c3": "Precision C3",
    "precision_c4": "Precision C4",
    "recall_c1": "Recall C1",
    "recall_c2": "Recall C2",
    "recall_c3": "Recall C3",
    "recall_c4": "Recall C4",
}

TEXT_COLOR = "#1F2933"
GRID_COLOR = "#D7DCE2"
PANEL_BG = "#FBFAF7"

MODEL_COLORS = {
    "U-Net": "#2F4858",
    "ResUNet-DS": "#B25D4C",
}

CLASS_COLORS = {
    "C1": "#4C9ED9",
    "C2": "#3BAA5B",
    "C3": "#D7A21B",
    "C4": "#A8322D",
}


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
    if not match:
        return None
    return int(match.group(1))


def load_metrics_for_model(model_name, root, pattern):
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
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    expected = set(range(5))
    found = set(df["fold"].tolist()) if len(df) > 0 else set()
    missing = sorted(expected - found)

    if missing:
        raise RuntimeError(f"{model_name}: missing folds {missing}")

    return df


def load_all_metrics():
    all_dfs = []

    for model_name, cfg in MODEL_PATTERNS.items():
        df = load_metrics_for_model(model_name, cfg["root"], cfg["pattern"])
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def summarize_model(df):
    rows = []

    for model_name in sorted(df["model"].unique()):
        sub = df[df["model"] == model_name]

        for metric in METRICS_MAIN:
            if metric not in sub.columns:
                continue

            values = sub[metric].astype(float).values

            rows.append({
                "model": model_name,
                "metric": metric,
                "metric_display": DISPLAY_NAMES.get(metric, metric),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean_sd": f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}",
            })

    return pd.DataFrame(rows)


def make_compact_comparison(summary):
    keep = [
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
    ]

    rows = []

    for metric in keep:
        row = {
            "Metric": DISPLAY_NAMES.get(metric, metric)
        }

        for model_name in ["U-Net", "ResUNet-DS"]:
            sub = summary[(summary["model"] == model_name) & (summary["metric"] == metric)]

            if len(sub) == 0:
                row[model_name] = ""
                row[f"{model_name}_mean"] = np.nan
                row[f"{model_name}_std"] = np.nan
            else:
                row[model_name] = sub["mean_sd"].iloc[0]
                row[f"{model_name}_mean"] = sub["mean"].iloc[0]
                row[f"{model_name}_std"] = sub["std"].iloc[0]

        if not np.isnan(row.get("U-Net_mean", np.nan)) and not np.isnan(row.get("ResUNet-DS_mean", np.nan)):
            row["Absolute improvement"] = row["ResUNet-DS_mean"] - row["U-Net_mean"]
        else:
            row["Absolute improvement"] = np.nan

        rows.append(row)

    compact = pd.DataFrame(rows)

    return compact


def make_foldwise_delta_table(df):
    metrics = [
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
    ]

    unet = df[df["model"] == "U-Net"].set_index("fold")
    resunet = df[df["model"] == "ResUNet-DS"].set_index("fold")

    rows = []

    for fold in range(5):
        row = {"fold": fold}

        for metric in metrics:
            row[f"unet_{metric}"] = float(unet.loc[fold, metric])
            row[f"resunetds_{metric}"] = float(resunet.loc[fold, metric])
            row[f"delta_{metric}"] = float(resunet.loc[fold, metric] - unet.loc[fold, metric])

        rows.append(row)

    return pd.DataFrame(rows)


def figure_main_metric_comparison(df):
    metrics = [
        ("mean_dice_no_bg", "Mean Dice"),
        ("minority_dice_c2_c3_c4", "Minority Dice"),
        ("mean_iou_no_bg", "Mean IoU"),
        ("macro_precision_no_bg", "Macro Precision"),
        ("macro_recall_no_bg", "Macro Recall"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor="white")

    x = np.arange(len(metrics))
    width = 0.34

    for idx, model_name in enumerate(["U-Net", "ResUNet-DS"]):
        sub = df[df["model"] == model_name]

        means = []
        stds = []

        for metric, _ in metrics:
            values = sub[metric].astype(float).values
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1))

        offset = -width / 2 if idx == 0 else width / 2

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
            alpha=0.92
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    save_figure(fig, "fig_baseline_01_main_metric_comparison")


def figure_per_class_dice_comparison(df):
    classes = [1, 2, 3, 4]
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(7.8, 4.8), facecolor="white")

    for model_name in ["U-Net", "ResUNet-DS"]:
        sub = df[df["model"] == model_name]

        means = np.array([sub[f"dice_c{c}"].astype(float).mean() for c in classes])
        stds = np.array([sub[f"dice_c{c}"].astype(float).std(ddof=1) for c in classes])

        ax.plot(
            x,
            means,
            marker="o",
            markersize=6.5,
            linewidth=2.2,
            color=MODEL_COLORS[model_name],
            label=model_name
        )

        ax.fill_between(
            x,
            means - stds,
            means + stds,
            color=MODEL_COLORS[model_name],
            alpha=0.14,
            linewidth=0
        )

    for idx, c in enumerate(classes):
        ax.scatter(
            [idx],
            [0.04],
            s=110,
            color=CLASS_COLORS[f"C{c}"],
            edgecolor="#202428",
            linewidth=0.6,
            clip_on=False
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in classes])
    ax.set_xlabel("Ordinal foreground class")
    ax.set_ylabel("Dice score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    clean_axis(ax)

    save_figure(fig, "fig_baseline_02_per_class_dice_comparison")


def figure_foldwise_delta(df_delta):
    metric = "delta_minority_dice_c2_c3_c4"

    values = df_delta[metric].values
    folds = df_delta["fold"].values

    fig, ax = plt.subplots(figsize=(7.4, 4.5), facecolor="white")

    colors = ["#3BAA5B" if v >= 0 else "#A8322D" for v in values]

    ax.bar(
        folds,
        values,
        color=colors,
        edgecolor="#202428",
        linewidth=0.65,
        width=0.58
    )

    ax.axhline(0, color="#202428", linewidth=0.9)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {int(f)}" for f in folds])
    ax.set_ylabel("ResUNet-DS minus U-Net minority Dice")
    ax.set_xlabel("Validation fold")
    clean_axis(ax)

    for fold, value in zip(folds, values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.005 if value >= 0 else -0.005
        ax.text(
            fold,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=8.5,
            color=TEXT_COLOR
        )

    save_figure(fig, "fig_baseline_03_foldwise_minority_dice_delta")


def main():
    print("=" * 90)
    print("Aggregating ResUNet-DS baseline and comparing with U-Net")
    print("=" * 90)

    setup_style()

    for d in [TABLE_DIR, FIG_DIR, REPORT_DIR, MANUSCRIPT_TABLE_DIR, MANUSCRIPT_FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_all_metrics()
    summary = summarize_model(df)
    compact = make_compact_comparison(summary)
    delta = make_foldwise_delta_table(df)

    foldwise_path = TABLE_DIR / "baseline_unet_vs_resunetds_foldwise_metrics.csv"
    summary_path = TABLE_DIR / "baseline_unet_vs_resunetds_summary_mean_sd.csv"
    compact_path = TABLE_DIR / "baseline_unet_vs_resunetds_compact_manuscript_table.csv"
    delta_path = TABLE_DIR / "baseline_unet_vs_resunetds_foldwise_delta.csv"

    manuscript_compact_path = MANUSCRIPT_TABLE_DIR / "baseline_unet_vs_resunetds_compact_manuscript_table.csv"
    manuscript_summary_path = MANUSCRIPT_TABLE_DIR / "baseline_unet_vs_resunetds_summary_mean_sd.csv"
    manuscript_delta_path = MANUSCRIPT_TABLE_DIR / "baseline_unet_vs_resunetds_foldwise_delta.csv"

    df.to_csv(foldwise_path, index=False)
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    delta.to_csv(delta_path, index=False)

    compact.to_csv(manuscript_compact_path, index=False)
    summary.to_csv(manuscript_summary_path, index=False)
    delta.to_csv(manuscript_delta_path, index=False)

    figure_main_metric_comparison(df)
    figure_per_class_dice_comparison(df)
    figure_foldwise_delta(delta)

    key_metrics = compact[
        compact["Metric"].isin([
            "Mean Dice (C1-C4)",
            "Minority Dice (C2-C4)",
            "Mean IoU (C1-C4)",
            "Dice C1",
            "Dice C2",
            "Dice C3",
            "Dice C4",
        ])
    ].copy()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": ["U-Net", "ResUNet-DS"],
        "num_folds_per_model": 5,
        "foldwise_metrics_path": str(foldwise_path),
        "summary_path": str(summary_path),
        "compact_table_path": str(compact_path),
        "delta_path": str(delta_path),
        "manuscript_compact_path": str(manuscript_compact_path),
        "figure_dir": str(FIG_DIR),
        "manuscript_figure_dir": str(MANUSCRIPT_FIG_DIR),
        "key_metrics": key_metrics[["Metric", "U-Net", "ResUNet-DS", "Absolute improvement"]].to_dict(orient="records"),
    }

    report_path = REPORT_DIR / "04b_resunetds_baseline_comparison_summary.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Compact baseline comparison:")
    print(compact[["Metric", "U-Net", "ResUNet-DS", "Absolute improvement"]].to_string(index=False))

    print()
    print("Foldwise minority Dice delta:")
    print(delta[["fold", "unet_minority_dice_c2_c3_c4", "resunetds_minority_dice_c2_c3_c4", "delta_minority_dice_c2_c3_c4"]].to_string(index=False))

    print()
    print("=" * 90)
    print("Baseline comparison completed")
    print("=" * 90)
    print(f"Compact table: {compact_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
