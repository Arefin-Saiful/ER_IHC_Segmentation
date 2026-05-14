import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

MANIFEST_PATH = PROJECT_ROOT / "data/manifests/manifest_er_ihc.csv"
FOLD_PATH = PROJECT_ROOT / "data/splits/folds_5_image_level.csv"

OUT_DIR = PROJECT_ROOT / "outputs/figures/eda_premium"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript_assets/figures_premium"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/eda_premium"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

CLASS_NAMES = {
    0: "Background",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}

CLASS_DESCRIPTIVE_NAMES = {
    0: "Background",
    1: "C1 weak",
    2: "C2 low-moderate",
    3: "C3 moderate",
    4: "C4 strong",
}

CLASS_COLORS = {
    0: "#ECE7DF",
    1: "#4C9ED9",
    2: "#3BAA5B",
    3: "#D7A21B",
    4: "#A8322D",
}

CLASS_RGB = {
    0: (236, 231, 223),
    1: (76, 158, 217),
    2: (59, 170, 91),
    3: (215, 162, 27),
    4: (168, 50, 45),
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    for base_dir in [OUT_DIR, MANUSCRIPT_DIR]:
        png_path = base_dir / f"{name}.png"
        pdf_path = base_dir / f"{name}.pdf"
        tiff_path = base_dir / f"{name}.tiff"

        fig.savefig(png_path, bbox_inches="tight", dpi=600)
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(tiff_path, bbox_inches="tight", dpi=600)

        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
        print(f"Saved: {tiff_path}")

    plt.close(fig)


def add_panel_label(ax, label):
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
        va="top",
        ha="left"
    )


def load_data():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    if not FOLD_PATH.exists():
        raise FileNotFoundError(f"Missing fold file: {FOLD_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    fold_df = pd.read_csv(FOLD_PATH)

    return df, fold_df


def strongest_class(row):
    for cls in [4, 3, 2, 1]:
        if row[f"pix_c{cls}"] > 0:
            return cls
    return 0


def build_class_summary(df):
    total_pixels = df["total_pixels"].sum()

    rows = []
    for cls in range(5):
        pixels = int(df[f"pix_c{cls}"].sum())
        present = int((df[f"pix_c{cls}"] > 0).sum())
        rows.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "class_description": CLASS_DESCRIPTIVE_NAMES[cls],
            "pixels": pixels,
            "pixel_percent": pixels / total_pixels * 100.0,
            "images_present": present,
            "image_present_percent": present / len(df) * 100.0,
        })

    return pd.DataFrame(rows)


def figure_ordinal_imbalance_overview(df):
    summary = build_class_summary(df)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "premium_ordinal_imbalance_overview_source.csv", index=False)

    minority_percent = summary.loc[summary["class_id"].isin([2, 3, 4]), "pixel_percent"].sum()
    foreground_percent = summary.loc[summary["class_id"].isin([1, 2, 3, 4]), "pixel_percent"].sum()
    c4_image_percent = float(summary.loc[summary["class_id"] == 4, "image_present_percent"].iloc[0])

    fig = plt.figure(figsize=(11.5, 6.4), facecolor="white")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.2, 1.0, 1.0],
        height_ratios=[1.1, 0.85],
        wspace=0.35,
        hspace=0.45
    )

    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2])

    add_panel_label(ax_a, "A")
    y = np.arange(len(summary))
    values = summary["pixel_percent"].values

    for i, row in summary.iterrows():
        cls = int(row["class_id"])
        ax_a.hlines(
            y=i,
            xmin=0.08,
            xmax=row["pixel_percent"],
            color=CLASS_COLORS[cls],
            linewidth=3.5,
            alpha=0.8
        )
        ax_a.scatter(
            row["pixel_percent"],
            i,
            s=260,
            color=CLASS_COLORS[cls],
            edgecolor="#222222",
            linewidth=0.7,
            zorder=3
        )
        ax_a.text(
            row["pixel_percent"] * 1.08,
            i,
            f"{row['pixel_percent']:.2f}%",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT_COLOR
        )

    ax_a.set_xscale("log")
    ax_a.set_xlim(0.08, 100)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(summary["class_description"].values)
    ax_a.set_xlabel("Dataset pixel proportion, log scale (%)")
    ax_a.set_ylabel("Ordinal class")
    ax_a.invert_yaxis()
    clean_axis(ax_a)

    add_panel_label(ax_b, "B")
    ax_b.axis("off")
    card_values = [
        ("Images", f"{len(df)}"),
        ("Foreground pixels", f"{foreground_percent:.1f}%"),
        ("C2-C4 pixels", f"{minority_percent:.1f}%"),
        ("C4-present images", f"{c4_image_percent:.1f}%"),
    ]

    y0 = 0.88
    for idx, (label, value) in enumerate(card_values):
        y_pos = y0 - idx * 0.22
        ax_b.text(
            0.02,
            y_pos,
            value,
            transform=ax_b.transAxes,
            fontsize=18,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="left",
            va="center"
        )
        ax_b.text(
            0.02,
            y_pos - 0.075,
            label,
            transform=ax_b.transAxes,
            fontsize=9.5,
            color="#59636E",
            ha="left",
            va="center"
        )
        ax_b.plot(
            [0.02, 0.82],
            [y_pos - 0.125, y_pos - 0.125],
            transform=ax_b.transAxes,
            color="#E2E6EA",
            linewidth=0.8
        )

    add_panel_label(ax_c, "C")
    start = 0
    for _, row in summary.iterrows():
        cls = int(row["class_id"])
        width = row["pixel_percent"]
        ax_c.barh(
            0,
            width,
            left=start,
            color=CLASS_COLORS[cls],
            edgecolor="white",
            linewidth=1.0,
            height=0.38
        )
        if width >= 2.0:
            ax_c.text(
                start + width / 2,
                0,
                CLASS_NAMES[cls],
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if cls in [1, 4] else TEXT_COLOR,
                fontweight="bold"
            )
        start += width

    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(-0.55, 0.55)
    ax_c.set_yticks([])
    ax_c.set_xlabel("Pixel composition across the full dataset (%)")
    ax_c.spines["left"].set_visible(False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.45, color=GRID_COLOR)

    add_panel_label(ax_d, "D")
    presence = summary["image_present_percent"].values
    classes = summary["class_id"].values
    ypos = np.arange(len(classes))

    ax_d.scatter(
        presence,
        ypos,
        s=summary["images_present"].values * 2.8,
        c=[CLASS_COLORS[int(c)] for c in classes],
        edgecolors="#222222",
        linewidths=0.7,
        alpha=0.95
    )

    for i, row in summary.iterrows():
        ax_d.text(
            min(row["image_present_percent"] + 3, 101),
            i,
            f"{int(row['images_present'])}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=TEXT_COLOR
        )

    ax_d.set_xlim(0, 108)
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels(summary["class_name"].values)
    ax_d.set_xlabel("Images containing class (%)")
    ax_d.set_ylabel("Class")
    ax_d.invert_yaxis()
    clean_axis(ax_d)

    save_figure(fig, "fig_premium_01_ordinal_imbalance_overview")


def figure_image_phenotype_map(df):
    work = df.copy()
    work["foreground_percent"] = work["foreground_ratio"] * 100.0
    work["minority_percent"] = work["minority_ratio_c2_c3_c4"] * 100.0
    work["c4_percent"] = work["ratio_c4"] * 100.0
    work["strongest_class"] = work.apply(strongest_class, axis=1)

    work.to_csv(TABLE_DIR / "premium_image_phenotype_map_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 6.2), facecolor="white")

    for cls in [1, 2, 3, 4]:
        sub = work[work["strongest_class"] == cls]
        if len(sub) == 0:
            continue

        sizes = 30 + sub["c4_percent"].values * 18
        ax.scatter(
            sub["foreground_percent"],
            sub["minority_percent"],
            s=sizes,
            color=CLASS_COLORS[cls],
            edgecolor="#222222",
            linewidth=0.45,
            alpha=0.75,
            label=CLASS_DESCRIPTIVE_NAMES[cls]
        )

    ax.axhline(
        work["minority_percent"].median(),
        color="#525A60",
        linestyle="--",
        linewidth=0.9,
        alpha=0.65
    )
    ax.axvline(
        work["foreground_percent"].median(),
        color="#525A60",
        linestyle="--",
        linewidth=0.9,
        alpha=0.65
    )

    high_cases = work.sort_values(["c4_percent", "minority_percent"], ascending=False).head(5)
    for _, row in high_cases.iterrows():
        ax.text(
            row["foreground_percent"] + 0.35,
            row["minority_percent"] + 0.35,
            f"{int(row['image_id'])}",
            fontsize=7.5,
            color=TEXT_COLOR
        )

    ax.set_xlabel("Foreground pixel proportion per image (%)")
    ax.set_ylabel("Minority-class pixel proportion per image, C2-C4 (%)")
    clean_axis(ax)

    legend = ax.legend(
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="Strongest class present"
    )
    legend.get_frame().set_edgecolor("#E1E4E8")
    legend.get_frame().set_linewidth(0.8)

    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#B7BDC5",
            markeredgecolor="#222222",
            markersize=s,
            label=label
        )
        for s, label in [(5, "low C4"), (8, "medium C4"), (11, "high C4")]
    ]

    legend2 = ax.legend(
        handles=size_handles,
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        title="Bubble size"
    )
    legend2.get_frame().set_edgecolor("#E1E4E8")
    legend2.get_frame().set_linewidth(0.8)
    ax.add_artist(legend)

    save_figure(fig, "fig_premium_02_image_phenotype_map")


def figure_class_ratio_distribution(df):
    classes = [1, 2, 3, 4]
    values = [df[f"ratio_c{cls}"].values * 100.0 for cls in classes]

    source_rows = []
    for cls, arr in zip(classes, values):
        for value in arr:
            source_rows.append({
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "image_level_percent": value
            })
    pd.DataFrame(source_rows).to_csv(TABLE_DIR / "premium_class_ratio_distribution_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.4), facecolor="white")

    parts = ax.violinplot(
        values,
        positions=np.arange(len(classes)) + 1,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.75
    )

    for body, cls in zip(parts["bodies"], classes):
        body.set_facecolor(CLASS_COLORS[cls])
        body.set_edgecolor("#222222")
        body.set_alpha(0.35)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(42)

    for i, (cls, arr) in enumerate(zip(classes, values), start=1):
        jitter = rng.normal(0, 0.045, size=len(arr))
        ax.scatter(
            np.full(len(arr), i) + jitter,
            arr,
            s=15,
            color=CLASS_COLORS[cls],
            edgecolor="#222222",
            linewidth=0.25,
            alpha=0.55,
            zorder=3
        )

        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        p10, p90 = np.percentile(arr, [10, 90])

        ax.plot([i, i], [p10, p90], color="#202428", linewidth=1.2, zorder=4)
        ax.plot([i - 0.18, i + 0.18], [q1, q1], color="#202428", linewidth=1.0, zorder=4)
        ax.plot([i - 0.18, i + 0.18], [q3, q3], color="#202428", linewidth=1.0, zorder=4)
        ax.plot([i - 0.25, i + 0.25], [median, median], color="#202428", linewidth=2.0, zorder=5)

    ax.set_xticks(np.arange(len(classes)) + 1)
    ax.set_xticklabels([CLASS_DESCRIPTIVE_NAMES[cls] for cls in classes])
    ax.set_ylabel("Image-level pixel proportion (%)")
    ax.set_xlabel("Foreground ordinal class")
    clean_axis(ax)

    save_figure(fig, "fig_premium_03_class_ratio_distribution")


def figure_fold_balance_stacked(fold_df):
    fold_summary = []
    for fold in sorted(fold_df["fold"].unique()):
        sub = fold_df[fold_df["fold"] == fold]
        total = sub["total_pixels"].sum()

        row = {
            "fold": int(fold),
            "num_images": int(len(sub)),
        }

        for cls in range(5):
            row[f"ratio_c{cls}"] = sub[f"pix_c{cls}"].sum() / total * 100.0
            row[f"images_c{cls}"] = int((sub[f"pix_c{cls}"] > 0).sum())

        row["minority_c2_c4"] = row["ratio_c2"] + row["ratio_c3"] + row["ratio_c4"]
        row["foreground"] = 100.0 - row["ratio_c0"]
        fold_summary.append(row)

    summary = pd.DataFrame(fold_summary)
    summary.to_csv(TABLE_DIR / "premium_fold_balance_source.csv", index=False)

    fig = plt.figure(figsize=(10.5, 5.2), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1.0], wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    add_panel_label(ax_a, "A")

    y_positions = np.arange(len(summary))
    left = np.zeros(len(summary))

    for cls in range(5):
        vals = summary[f"ratio_c{cls}"].values
        ax_a.barh(
            y_positions,
            vals,
            left=left,
            color=CLASS_COLORS[cls],
            edgecolor="white",
            linewidth=0.9,
            height=0.62,
            label=CLASS_NAMES[cls]
        )
        left += vals

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([f"Fold {int(x)}" for x in summary["fold"]])
    ax_a.set_xlabel("Validation fold pixel composition (%)")
    ax_a.set_xlim(0, 100)
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.45, color=GRID_COLOR)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=CLASS_COLORS[cls], edgecolor="white", label=CLASS_NAMES[cls])
        for cls in range(5)
    ]
    ax_a.legend(
        handles=legend_handles,
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28)
    )

    add_panel_label(ax_b, "B")
    ax_b.axis("off")

    ax_b.text(
        0.02,
        0.94,
        "Fold balance check",
        transform=ax_b.transAxes,
        fontsize=11,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="left",
        va="top"
    )

    y0 = 0.78
    line_gap = 0.13

    metric_rows = [
        ("Images/fold", f"{int(summary['num_images'].min())}-{int(summary['num_images'].max())}"),
        ("Foreground range", f"{summary['foreground'].min():.1f}-{summary['foreground'].max():.1f}%"),
        ("C2-C4 range", f"{summary['minority_c2_c4'].min():.1f}-{summary['minority_c2_c4'].max():.1f}%"),
        ("C4 images/fold", f"{summary['images_c4'].min()}-{summary['images_c4'].max()}"),
    ]

    for idx, (label, value) in enumerate(metric_rows):
        y = y0 - idx * line_gap
        ax_b.text(
            0.02,
            y,
            value,
            transform=ax_b.transAxes,
            fontsize=15,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="left",
            va="center"
        )
        ax_b.text(
            0.45,
            y,
            label,
            transform=ax_b.transAxes,
            fontsize=9.3,
            color="#59636E",
            ha="left",
            va="center"
        )

    save_figure(fig, "fig_premium_04_fold_balance_stacked")


def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for cls in range(5):
        out[mask == cls] = CLASS_RGB[cls]

    return out


def overlay_mask(image, mask, alpha=0.42):
    image = np.asarray(image).astype(np.float32)
    color_mask = colorize_mask(mask).astype(np.float32)
    out = image.copy()

    fg = mask > 0
    out[fg] = (1 - alpha) * image[fg] + alpha * color_mask[fg]

    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contours(image, mask):
    image_np = np.asarray(image).copy()

    if not HAS_CV2:
        return overlay_mask(image_np, mask, alpha=0.42)

    contour_img = overlay_mask(image_np, mask, alpha=0.26)

    for cls in [1, 2, 3, 4]:
        binary = (mask == cls).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rgb = CLASS_RGB[cls]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        contour_img = cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR)
        cv2.drawContours(contour_img, contours, -1, bgr, 1, lineType=cv2.LINE_AA)
        contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)

    return contour_img


def select_representative_ids(df):
    selected = []

    def add(image_id):
        image_id = int(image_id)
        if image_id not in selected:
            selected.append(image_id)

    df = df.copy()
    df["strongest_class"] = df.apply(strongest_class, axis=1)
    df["minority_percent"] = df["minority_ratio_c2_c3_c4"] * 100.0
    df["foreground_percent"] = df["foreground_ratio"] * 100.0

    weak = df[(df["strongest_class"] == 1)].sort_values("foreground_percent", ascending=False)
    low_mod = df[(df["strongest_class"] == 2)].sort_values("minority_percent", ascending=False)
    moderate = df[(df["strongest_class"] == 3)].sort_values("minority_percent", ascending=False)
    strong = df[(df["strongest_class"] == 4)].sort_values("ratio_c4", ascending=False)
    mixed = df.sort_values(["minority_percent", "foreground_percent"], ascending=False)

    for subset in [weak, low_mod, moderate, strong, mixed]:
        if len(subset) > 0:
            add(subset.iloc[0]["image_id"])

    return selected[:5]


def figure_dataset_morphology_examples(df):
    selected_ids = select_representative_ids(df)
    pd.DataFrame({"image_id": selected_ids}).to_csv(TABLE_DIR / "premium_dataset_morphology_selected_ids.csv", index=False)

    n = len(selected_ids)
    fig = plt.figure(figsize=(10.5, 2.45 * n + 0.6), facecolor="white")
    gs = GridSpec(n + 1, 4, figure=fig, height_ratios=[*([1] * n), 0.20], width_ratios=[1, 1, 1, 0.28], hspace=0.08, wspace=0.06)

    for r, image_id in enumerate(selected_ids):
        row = df[df["image_id"] == image_id].iloc[0]

        image = Image.open(row["image_path"]).convert("RGB")
        mask = np.array(Image.open(row["mask_label_path"]))

        mask_rgb = colorize_mask(mask)
        contour = draw_contours(image, mask)

        panels = [np.asarray(image), mask_rgb, contour]

        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(panels[c])
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_edgecolor("#E1E4E8")

            if c == 0:
                ax.set_ylabel(
                    f"ID {image_id}",
                    rotation=0,
                    labelpad=24,
                    va="center",
                    fontsize=8.5,
                    color=TEXT_COLOR
                )

        ax_info = fig.add_subplot(gs[r, 3])
        ax_info.axis("off")

        stats_text = [
            f"FG {row['foreground_ratio'] * 100:.1f}%",
            f"C2-C4 {row['minority_ratio_c2_c3_c4'] * 100:.1f}%",
            f"C4 {row['ratio_c4'] * 100:.1f}%"
        ]

        for k, text in enumerate(stats_text):
            ax_info.text(
                0.02,
                0.80 - k * 0.25,
                text,
                transform=ax_info.transAxes,
                fontsize=8.2,
                color=TEXT_COLOR,
                ha="left",
                va="center"
            )

    legend_ax = fig.add_subplot(gs[-1, :])
    legend_ax.axis("off")

    handles = [
        Patch(facecolor=CLASS_COLORS[cls], edgecolor="#222222", label=CLASS_DESCRIPTIVE_NAMES[cls])
        for cls in range(1, 5)
    ]

    legend_ax.legend(
        handles=handles,
        ncol=4,
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.5, 0.5)
    )

    save_figure(fig, "fig_premium_05_dataset_morphology_examples")


def main():
    print("=" * 90)
    print("Phase 2B: Premium Q1-style EDA figure generation")
    print("=" * 90)

    setup_style()

    for directory in [OUT_DIR, MANUSCRIPT_DIR, TABLE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    df, fold_df = load_data()

    print(f"Loaded manifest: {MANIFEST_PATH}")
    print(f"Loaded fold file: {FOLD_PATH}")
    print(f"Images: {len(df)}")

    figure_ordinal_imbalance_overview(df)
    figure_image_phenotype_map(df)
    figure_class_ratio_distribution(df)
    figure_fold_balance_stacked(fold_df)
    figure_dataset_morphology_examples(df)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(OUT_DIR),
        "manuscript_dir": str(MANUSCRIPT_DIR),
        "figures": [
            "fig_premium_01_ordinal_imbalance_overview",
            "fig_premium_02_image_phenotype_map",
            "fig_premium_03_class_ratio_distribution",
            "fig_premium_04_fold_balance_stacked",
            "fig_premium_05_dataset_morphology_examples",
        ],
        "note": "No internal figure titles were added. Captions should be written in the manuscript."
    }

    summary_path = REPORT_DIR / "02b_premium_eda_figures_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 90)
    print("Premium EDA figure generation completed")
    print("=" * 90)
    print(f"Saved summary: {summary_path}")
    print(f"Output figures: {OUT_DIR}")
    print(f"Manuscript figures: {MANUSCRIPT_DIR}")


if __name__ == "__main__":
    main()
