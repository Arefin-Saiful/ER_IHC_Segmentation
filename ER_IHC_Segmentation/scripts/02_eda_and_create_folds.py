import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import StratifiedKFold, KFold


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

MANIFEST_PATH = PROJECT_ROOT / "data/manifests/manifest_er_ihc.csv"

SPLIT_DIR = PROJECT_ROOT / "data/splits"

EDA_TABLE_DIR = PROJECT_ROOT / "outputs/tables/eda"
EDA_FIG_DIR = PROJECT_ROOT / "outputs/figures/eda"
FOLD_FIG_DIR = PROJECT_ROOT / "outputs/figures/folds"
QC_FIG_DIR = PROJECT_ROOT / "outputs/figures/qc"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "manuscript_assets/figures"
MANUSCRIPT_TABLE_DIR = PROJECT_ROOT / "manuscript_assets/tables"

CLASS_NAMES = {
    0: "Background",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 159, 255),
    2: (0, 180, 0),
    3: (230, 180, 0),
    4: (220, 0, 0),
}


def setup_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, out_base):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")

    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def save_to_both_locations(fig_builder, filename):
    fig1 = fig_builder()
    save_figure(fig1, filename)

    manuscript_name = MANUSCRIPT_FIG_DIR / Path(filename).name
    fig2 = fig_builder()
    save_figure(fig2, manuscript_name)


def get_label_cmap():
    colors = []
    for cls in range(5):
        rgb = np.array(CLASS_COLORS[cls]) / 255.0
        colors.append((*rgb, 1.0))
    return ListedColormap(colors)


def overlay_mask_on_image(image, mask, alpha=0.45):
    image = np.asarray(image).astype(np.float32)
    overlay = image.copy()
    color_mask = np.zeros_like(image)

    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        color_mask[mask == cls] = np.array(color)

    foreground = mask > 0
    overlay[foreground] = (1.0 - alpha) * image[foreground] + alpha * color_mask[foreground]

    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_class_pixel_distribution(df):
    total_pixels = df["total_pixels"].sum()

    rows = []
    for cls in range(5):
        total_cls_pixels = df[f"pix_c{cls}"].sum()
        rows.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "total_pixels": int(total_cls_pixels),
            "percentage": float(total_cls_pixels / total_pixels * 100.0),
        })

    table = pd.DataFrame(rows)
    table_path = EDA_TABLE_DIR / "eda_class_pixel_distribution.csv"
    manuscript_table_path = MANUSCRIPT_TABLE_DIR / "eda_class_pixel_distribution.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(manuscript_table_path, index=False)

    def build_fig():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        y = np.arange(len(table))
        values = table["percentage"].values

        ax.barh(y, values, edgecolor="black", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(table["class_name"].values)
        ax.set_xlabel("Pixel proportion (%)")
        ax.set_ylabel("Class")

        offset = max(values) * 0.01
        for i, value in enumerate(values):
            ax.text(value + offset, i, f"{value:.2f}%", va="center", fontsize=9)

        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
        return fig

    save_to_both_locations(build_fig, EDA_FIG_DIR / "fig_01_class_pixel_distribution")

    return table


def create_image_level_class_presence(df):
    rows = []
    total_images = len(df)

    for cls in range(5):
        present = int((df[f"pix_c{cls}"] > 0).sum())
        rows.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "images_present": present,
            "percentage": present / total_images * 100.0,
        })

    table = pd.DataFrame(rows)
    table_path = EDA_TABLE_DIR / "eda_image_level_class_presence.csv"
    manuscript_table_path = MANUSCRIPT_TABLE_DIR / "eda_image_level_class_presence.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(manuscript_table_path, index=False)

    def build_fig():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = np.arange(len(table))
        values = table["images_present"].values

        ax.bar(x, values, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(table["class_name"].values)
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of images")

        for i, value in enumerate(values):
            ax.text(i, value + 2, f"{value}", ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, max(values) * 1.12)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        return fig

    save_to_both_locations(build_fig, EDA_FIG_DIR / "fig_02_image_level_class_presence")

    return table


def create_ratio_histogram(df, column, xlabel, filename):
    values = df[column].values * 100.0

    stats = pd.DataFrame([{
        "column": column,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "q75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }])

    stats.to_csv(EDA_TABLE_DIR / f"{filename}_summary.csv", index=False)
    stats.to_csv(MANUSCRIPT_TABLE_DIR / f"{filename}_summary.csv", index=False)

    def build_fig():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.hist(values, bins=20, edgecolor="black", linewidth=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of images")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        return fig

    save_to_both_locations(build_fig, EDA_FIG_DIR / filename)


def create_per_image_class_ratio_boxplot(df):
    ratio_cols = [f"ratio_c{cls}" for cls in range(1, 5)]
    values = [df[col].values * 100.0 for col in ratio_cols]
    labels = [CLASS_NAMES[cls] for cls in range(1, 5)]

    summary_rows = []
    for cls, col in zip(range(1, 5), ratio_cols):
        arr = df[col].values * 100.0
        summary_rows.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "mean_percent": float(np.mean(arr)),
            "std_percent": float(np.std(arr)),
            "median_percent": float(np.median(arr)),
            "max_percent": float(np.max(arr)),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(EDA_TABLE_DIR / "eda_per_image_foreground_class_ratio_summary.csv", index=False)
    summary.to_csv(MANUSCRIPT_TABLE_DIR / "eda_per_image_foreground_class_ratio_summary.csv", index=False)

    def build_fig():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.boxplot(values, labels=labels, showfliers=True)
        ax.set_xlabel("Foreground class")
        ax.set_ylabel("Image-level pixel proportion (%)")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        return fig

    save_to_both_locations(build_fig, EDA_FIG_DIR / "fig_05_per_image_class_ratio_boxplot")


def create_sample_overlay_panel(df):
    selected_ids = []

    def add_id(image_id):
        image_id = int(image_id)
        if image_id not in selected_ids:
            selected_ids.append(image_id)

    fg_sorted = df.sort_values("foreground_ratio")
    add_id(fg_sorted.iloc[len(fg_sorted) // 2]["image_id"])
    add_id(fg_sorted.iloc[-1]["image_id"])

    for cls in [1, 2, 3, 4]:
        col = f"pix_c{cls}"
        sub = df[df[col] > 0].sort_values(col, ascending=False)
        if len(sub) > 0:
            add_id(sub.iloc[0]["image_id"])

    selected_ids = selected_ids[:6]

    def build_fig():
        n = len(selected_ids)
        fig, axes = plt.subplots(n, 3, figsize=(8.5, 2.55 * n))

        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        cmap = get_label_cmap()

        for row_idx, image_id in enumerate(selected_ids):
            row = df[df["image_id"] == image_id].iloc[0]

            image = Image.open(row["image_path"]).convert("RGB")
            mask = np.array(Image.open(row["mask_label_path"]))

            overlay = overlay_mask_on_image(image, mask)

            axes[row_idx, 0].imshow(image)
            axes[row_idx, 1].imshow(mask, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            axes[row_idx, 2].imshow(overlay)

            axes[row_idx, 0].set_ylabel(
                f"ID {image_id}",
                rotation=0,
                labelpad=26,
                va="center",
                fontsize=9
            )

            for col_idx in range(3):
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])
                for spine in axes[row_idx, col_idx].spines.values():
                    spine.set_linewidth(0.6)

        plt.tight_layout()
        return fig

    save_to_both_locations(build_fig, QC_FIG_DIR / "fig_06_dataset_input_mask_overlay_examples")

    selected_table = pd.DataFrame({"selected_image_id": selected_ids})
    selected_table.to_csv(EDA_TABLE_DIR / "selected_ids_for_overlay_examples.csv", index=False)


def make_quantile_bins(series, q):
    values = series.values

    try:
        bins = pd.qcut(values, q=q, labels=False, duplicates="drop")
        bins = pd.Series(bins).fillna(0).astype(int).values
    except Exception:
        bins = pd.cut(values, bins=q, labels=False, include_lowest=True)
        bins = pd.Series(bins).fillna(0).astype(int).values

    return bins


def create_stratification_labels(df, n_splits=5):
    work = df.copy()

    work["fg_bin_5"] = make_quantile_bins(work["foreground_ratio"], 5)
    work["fg_bin_3"] = make_quantile_bins(work["foreground_ratio"], 3)
    work["minority_bin_3"] = make_quantile_bins(work["minority_ratio_c2_c3_c4"], 3)

    work["has_c2"] = (work["pix_c2"] > 0).astype(int)
    work["has_c3"] = (work["pix_c3"] > 0).astype(int)
    work["has_c4"] = (work["pix_c4"] > 0).astype(int)

    foreground_pixels = work[[f"pix_c{cls}" for cls in range(1, 5)]].values
    dominant_foreground = np.argmax(foreground_pixels, axis=1) + 1
    work["dominant_foreground"] = dominant_foreground

    strongest_present = []
    for _, row in work.iterrows():
        if row["pix_c4"] > 0:
            strongest_present.append(4)
        elif row["pix_c3"] > 0:
            strongest_present.append(3)
        elif row["pix_c2"] > 0:
            strongest_present.append(2)
        elif row["pix_c1"] > 0:
            strongest_present.append(1)
        else:
            strongest_present.append(0)

    work["strongest_present"] = strongest_present

    candidates = []

    candidates.append((
        "strongest_present_plus_foreground_bin_5",
        work["strongest_present"].astype(str) + "_fg" + work["fg_bin_5"].astype(str)
    ))

    candidates.append((
        "strongest_present_plus_foreground_bin_3",
        work["strongest_present"].astype(str) + "_fg" + work["fg_bin_3"].astype(str)
    ))

    candidates.append((
        "c4_c3_presence_plus_foreground_bin_3",
        "c4" + work["has_c4"].astype(str) +
        "_c3" + work["has_c3"].astype(str) +
        "_fg" + work["fg_bin_3"].astype(str)
    ))

    candidates.append((
        "strongest_present_only",
        work["strongest_present"].astype(str)
    ))

    candidates.append((
        "foreground_bin_5_only",
        work["fg_bin_5"].astype(str)
    ))

    for strategy_name, labels in candidates:
        counts = pd.Series(labels).value_counts()
        if counts.min() >= n_splits:
            print(f"Selected stratification strategy: {strategy_name}")
            print("Stratification group counts:")
            print(counts.sort_index().to_string())
            return labels.values, strategy_name

    print("Warning: no safe stratification label found. Falling back to KFold.")
    return None, "kfold_fallback"


def create_folds(df):
    df = df.copy()
    df["fold"] = -1

    n_splits = 5
    seed = 42

    labels, strategy = create_stratification_labels(df, n_splits=n_splits)

    if labels is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = splitter.split(df)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = splitter.split(df, labels)

    for fold_idx, (_, val_indices) in enumerate(split_iterator):
        df.loc[val_indices, "fold"] = fold_idx

    fold_csv_path = SPLIT_DIR / "folds_5_image_level.csv"
    df.to_csv(fold_csv_path, index=False)

    for fold_idx in range(n_splits):
        train_df = df[df["fold"] != fold_idx].copy()
        val_df = df[df["fold"] == fold_idx].copy()

        train_df.to_csv(SPLIT_DIR / f"fold_{fold_idx}_train.csv", index=False)
        val_df.to_csv(SPLIT_DIR / f"fold_{fold_idx}_val.csv", index=False)

    summary_rows = []
    for fold_idx in range(n_splits):
        fold_df = df[df["fold"] == fold_idx]
        total_pixels = fold_df["total_pixels"].sum()

        row = {
            "fold": fold_idx,
            "num_images": int(len(fold_df)),
            "foreground_ratio": float(fold_df["foreground_pixels"].sum() / total_pixels),
            "minority_ratio_c2_c3_c4": float(fold_df["minority_pixels_c2_c3_c4"].sum() / total_pixels),
        }

        for cls in range(5):
            row[f"ratio_c{cls}"] = float(fold_df[f"pix_c{cls}"].sum() / total_pixels)
            row[f"images_with_c{cls}"] = int((fold_df[f"pix_c{cls}"] > 0).sum())

        summary_rows.append(row)

    fold_summary = pd.DataFrame(summary_rows)

    fold_summary_path = EDA_TABLE_DIR / "fold_distribution_summary.csv"
    manuscript_fold_summary_path = MANUSCRIPT_TABLE_DIR / "fold_distribution_summary.csv"

    fold_summary.to_csv(fold_summary_path, index=False)
    fold_summary.to_csv(manuscript_fold_summary_path, index=False)

    return df, fold_summary, strategy, fold_csv_path, fold_summary_path


def create_fold_heatmap(fold_summary):
    ratio_cols = [f"ratio_c{cls}" for cls in range(5)]
    matrix = fold_summary[ratio_cols].values * 100.0

    def build_fig():
        fig, ax = plt.subplots(figsize=(7.5, 4.2))

        image = ax.imshow(matrix, aspect="auto")

        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([CLASS_NAMES[cls] for cls in range(5)])
        ax.set_yticks(np.arange(len(fold_summary)))
        ax.set_yticklabels([f"Fold {int(fold)}" for fold in fold_summary["fold"]])

        ax.set_xlabel("Class")
        ax.set_ylabel("Validation fold")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Pixel proportion (%)")

        return fig

    save_to_both_locations(build_fig, FOLD_FIG_DIR / "fig_07_fold_class_distribution_heatmap")


def create_fold_foreground_bar(fold_summary):
    plot_df = fold_summary.copy()
    plot_df["foreground_percent"] = plot_df["foreground_ratio"] * 100.0
    plot_df["minority_percent"] = plot_df["minority_ratio_c2_c3_c4"] * 100.0

    def build_fig():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        x = np.arange(len(plot_df))
        width = 0.35

        ax.bar(
            x - width / 2,
            plot_df["foreground_percent"].values,
            width,
            label="Foreground"
        )

        ax.bar(
            x + width / 2,
            plot_df["minority_percent"].values,
            width,
            label="C2-C4"
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {int(fold)}" for fold in plot_df["fold"]])
        ax.set_xlabel("Validation fold")
        ax.set_ylabel("Pixel proportion (%)")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

        return fig

    save_to_both_locations(build_fig, FOLD_FIG_DIR / "fig_08_fold_foreground_minority_distribution")


def main():
    print("=" * 90)
    print("Phase 2: Q1-quality EDA figures and 5-fold split creation")
    print("=" * 90)

    setup_plot_style()

    for directory in [
        SPLIT_DIR,
        EDA_TABLE_DIR,
        EDA_FIG_DIR,
        FOLD_FIG_DIR,
        QC_FIG_DIR,
        REPORT_DIR,
        MANUSCRIPT_FIG_DIR,
        MANUSCRIPT_TABLE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    print(f"Loaded manifest: {MANIFEST_PATH}")
    print(f"Number of images: {len(df)}")

    class_distribution = create_class_pixel_distribution(df)
    image_presence = create_image_level_class_presence(df)

    create_ratio_histogram(
        df,
        column="foreground_ratio",
        xlabel="Foreground pixel proportion per image (%)",
        filename="fig_03_foreground_ratio_distribution"
    )

    create_ratio_histogram(
        df,
        column="minority_ratio_c2_c3_c4",
        xlabel="Minority-class pixel proportion per image (%)",
        filename="fig_04_minority_ratio_distribution"
    )

    create_per_image_class_ratio_boxplot(df)

    create_sample_overlay_panel(df)

    fold_df, fold_summary, strategy, fold_csv_path, fold_summary_path = create_folds(df)

    create_fold_heatmap(fold_summary)
    create_fold_foreground_bar(fold_summary)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "manifest_path": str(MANIFEST_PATH),
        "num_images": int(len(df)),
        "stratification_strategy": strategy,
        "fold_csv_path": str(fold_csv_path),
        "fold_summary_path": str(fold_summary_path),
        "eda_figure_dir": str(EDA_FIG_DIR),
        "fold_figure_dir": str(FOLD_FIG_DIR),
        "qc_figure_dir": str(QC_FIG_DIR),
        "manuscript_figure_dir": str(MANUSCRIPT_FIG_DIR),
        "manuscript_table_dir": str(MANUSCRIPT_TABLE_DIR),
    }

    summary_path = REPORT_DIR / "02_eda_and_folds_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 90)
    print("Phase 2 completed successfully")
    print("=" * 90)

    print()
    print("Class pixel distribution:")
    print(class_distribution.to_string(index=False))

    print()
    print("Image-level class presence:")
    print(image_presence.to_string(index=False))

    print()
    print("Fold summary:")
    print(fold_summary.to_string(index=False))

    print()
    print(f"Saved fold CSV: {fold_csv_path}")
    print(f"Saved fold summary: {fold_summary_path}")
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
