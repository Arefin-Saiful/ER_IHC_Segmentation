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
FIG_DIR = PROJECT_ROOT / "outputs/figures/results/deeplabv3_comparison"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

MANUSCRIPT_TABLE_DIR = PROJECT_ROOT / "manuscript_assets/tables"
MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "manuscript_assets/figures_results"

MODEL_CONFIGS = {
    "U-Net": {
        "root": PROJECT_ROOT / "outputs/metrics/unet_baseline",
        "pattern": "unet_baseline_fullval_fold*_aug-full_base32_crop320/best_metrics.json",
    },
    "ResUNet-DS": {
        "root": PROJECT_ROOT / "outputs/metrics/resunetds_baseline",
        "pattern": "resunetds_baseline_fullval_fold*_aug-full_base32_crop320/best_metrics.json",
    },
    "Proposed AMC-Ordinal": {
        "root": PROJECT_ROOT / "outputs/metrics/proposed_amc_ordinal",
        "pattern": "proposed_amc_ordinal_fullval_fold*_aug-full_base32_crop320_ord0.1_ft0.5/best_metrics.json",
    },
    "DeepLabV3-ResNet50": {
        "root": PROJECT_ROOT / "outputs/metrics/deeplabv3_resnet50",
        "pattern": "deeplabv3_resnet50_fullval_fold*_aug-full_crop320_pretrained/best_metrics.json",
    },
}

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

MODEL_COLORS = {
    "U-Net": "#2F4858",
    "ResUNet-DS": "#B25D4C",
    "Proposed AMC-Ordinal": "#2E7D5B",
    "DeepLabV3-ResNet50": "#6A4C93",
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
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())

    kappa = np.nan if denominator <= 1e-12 else 1.0 - numerator / denominator

    return {
        "weighted_kappa_fg": float(kappa),
        "ordinal_mae_fg": float((dist * fg).sum() / total),
        "adjacent_error_rate_fg": float(((dist == 1) * fg).sum() / total),
        "distant_error_rate_fg": float(((dist >= 2) * fg).sum() / total),
    }


def load_model_metrics(model_name, root, pattern):
    paths = sorted(root.glob(pattern))

    if len(paths) == 0:
        raise RuntimeError(f"No metrics found for {model_name}: {root}/{pattern}")

    rows = []

    for path in paths:
        fold = extract_fold(path)

        if fold is None:
            continue

        with open(path, "r") as f:
            metrics = json.load(f)

        metrics.update(compute_ordinal_from_confusion(path))

        row = {
            "model": model_name,
            "fold": fold,
            "metrics_path": str(path),
        }

        row.update(metrics)
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
        print(f"Loading {model_name}")
        df = load_model_metrics(model_name, cfg["root"], cfg["pattern"])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def summarize(df):
    rows = []

    for model_name in MODEL_CONFIGS.keys():
        sub = df[df["model"] == model_name]

        for metric in METRICS:
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
        row = {"Metric": DISPLAY_NAMES.get(metric, metric)}

        for model_name in MODEL_CONFIGS.keys():
            sub = summary[(summary["model"] == model_name) & (summary["metric"] == metric)]
            row[model_name] = sub["mean_sd"].iloc[0] if len(sub) else ""

        rows.append(row)

    return pd.DataFrame(rows)


def make_delta_table(summary):
    baseline = "ResUNet-DS"
    rows = []

    for model_name in MODEL_CONFIGS.keys():
        if model_name == baseline:
            continue

        for metric in METRICS:
            a = summary[(summary["model"] == model_name) & (summary["metric"] == metric)]
            b = summary[(summary["model"] == baseline) & (summary["metric"] == metric)]

            if len(a) == 0 or len(b) == 0:
                continue

            rows.append({
                "model": model_name,
                "baseline": baseline,
                "metric": metric,
                "metric_display": DISPLAY_NAMES.get(metric, metric),
                "model_mean": float(a["mean"].iloc[0]),
                "baseline_mean": float(b["mean"].iloc[0]),
                "delta_vs_resunetds": float(a["mean"].iloc[0] - b["mean"].iloc[0]),
            })

    return pd.DataFrame(rows)


def figure_main_metric_comparison(df):
    metrics = [
        ("mean_dice_no_bg", "Mean Dice"),
        ("minority_dice_c2_c3_c4", "Minority Dice"),
        ("mean_iou_no_bg", "Mean IoU"),
        ("weighted_kappa_fg", "Weighted Kappa"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor="white")

    model_names = list(MODEL_CONFIGS.keys())
    x = np.arange(len(metrics))
    width = 0.18

    for idx, model_name in enumerate(model_names):
        sub = df[df["model"] == model_name]

        means = []
        stds = []

        for metric, _ in metrics:
            values = sub[metric].astype(float).values
            means.append(np.nanmean(values))
            stds.append(np.nanstd(values, ddof=1))

        offset = (idx - 1.5) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2.5,
            color=MODEL_COLORS[model_name],
            edgecolor="#202428",
            linewidth=0.55,
            label=model_name,
            alpha=0.94
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    save_figure(fig, "fig_deeplabv3_01_main_metric_comparison")


def figure_per_class_dice(df):
    classes = [1, 2, 3, 4]
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor="white")

    for model_name in MODEL_CONFIGS.keys():
        sub = df[df["model"] == model_name]

        means = np.array([sub[f"dice_c{c}"].astype(float).mean() for c in classes])
        stds = np.array([sub[f"dice_c{c}"].astype(float).std(ddof=1) for c in classes])

        ax.plot(
            x,
            means,
            marker="o",
            markersize=6,
            linewidth=2.0,
            color=MODEL_COLORS[model_name],
            label=model_name
        )

        ax.fill_between(
            x,
            means - stds,
            means + stds,
            color=MODEL_COLORS[model_name],
            alpha=0.10,
            linewidth=0
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in classes])
    ax.set_xlabel("Ordinal foreground class")
    ax.set_ylabel("Dice score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    clean_axis(ax)

    save_figure(fig, "fig_deeplabv3_02_per_class_dice_comparison")


def main():
    print("=" * 90)
    print("Aggregating DeepLabV3-ResNet50 with all final models")
    print("=" * 90)

    setup_style()

    for d in [TABLE_DIR, FIG_DIR, REPORT_DIR, MANUSCRIPT_TABLE_DIR, MANUSCRIPT_FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_all_metrics()
    summary = summarize(df)
    compact = make_compact_table(summary)
    delta = make_delta_table(summary)

    foldwise_path = TABLE_DIR / "deeplabv3_all_models_foldwise_metrics.csv"
    summary_path = TABLE_DIR / "deeplabv3_all_models_summary_mean_sd.csv"
    compact_path = TABLE_DIR / "deeplabv3_all_models_compact_table.csv"
    delta_path = TABLE_DIR / "deeplabv3_all_models_delta_vs_resunetds.csv"

    manuscript_summary_path = MANUSCRIPT_TABLE_DIR / "deeplabv3_all_models_summary_mean_sd.csv"
    manuscript_compact_path = MANUSCRIPT_TABLE_DIR / "deeplabv3_all_models_compact_table.csv"
    manuscript_delta_path = MANUSCRIPT_TABLE_DIR / "deeplabv3_all_models_delta_vs_resunetds.csv"

    df.to_csv(foldwise_path, index=False)
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    delta.to_csv(delta_path, index=False)

    summary.to_csv(manuscript_summary_path, index=False)
    compact.to_csv(manuscript_compact_path, index=False)
    delta.to_csv(manuscript_delta_path, index=False)

    figure_main_metric_comparison(df)
    figure_per_class_dice(df)

    key_metrics = compact[
        compact["Metric"].isin([
            "Mean Dice (C1-C4)",
            "Minority Dice (C2-C4)",
            "Mean IoU (C1-C4)",
            "Weighted Kappa",
            "Ordinal MAE",
            "Dice C1",
            "Dice C2",
            "Dice C3",
            "Dice C4",
        ])
    ].copy()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": list(MODEL_CONFIGS.keys()),
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

    report_path = REPORT_DIR / "10b_deeplabv3_all_models_comparison_summary.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Compact comparison table:")
    print(compact.to_string(index=False))

    print()
    print("Delta vs ResUNet-DS:")
    print(delta[delta["metric"].isin([
        "mean_dice_no_bg",
        "minority_dice_c2_c3_c4",
        "mean_iou_no_bg",
        "dice_c2",
        "dice_c4",
        "weighted_kappa_fg",
        "ordinal_mae_fg",
    ])].to_string(index=False))

    print()
    print("=" * 90)
    print("DeepLabV3 comparison aggregation completed")
    print("=" * 90)
    print(f"Compact table: {compact_path}")
    print(f"Delta table: {delta_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
