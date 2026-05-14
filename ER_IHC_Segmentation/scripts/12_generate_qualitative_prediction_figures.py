import json
import importlib.util
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

SPLIT_DIR = PROJECT_ROOT / "data/splits"

OUT_TABLE_DIR = PROJECT_ROOT / "outputs/tables/qualitative"
OUT_FIG_DIR = PROJECT_ROOT / "outputs/figures/qualitative"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "manuscript_assets/figures"
MANUSCRIPT_TABLE_DIR = PROJECT_ROOT / "manuscript_assets/tables"

for d in [OUT_TABLE_DIR, OUT_FIG_DIR, REPORT_DIR, MANUSCRIPT_FIG_DIR, MANUSCRIPT_TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5

MODEL_INFO = {
    "ResUNet-DS": {
        "script": PROJECT_ROOT / "scripts/04_train_resunetds_baseline.py",
        "checkpoint_pattern": PROJECT_ROOT / "outputs/checkpoints/resunetds_baseline/resunetds_baseline_fullval_fold{fold}_aug-full_base32_crop320/best_model.pt",
        "type": "resunet",
    },
    "Proposed AMC-Ordinal": {
        "script": PROJECT_ROOT / "scripts/05_train_proposed_amc_ordinal.py",
        "checkpoint_pattern": PROJECT_ROOT / "outputs/checkpoints/proposed_amc_ordinal/proposed_amc_ordinal_fullval_fold{fold}_aug-full_base32_crop320_ord0.1_ft0.5/best_model.pt",
        "type": "resunet",
    },
    "DeepLabV3-ResNet50": {
        "script": PROJECT_ROOT / "scripts/10_train_deeplabv3_resnet50_baseline.py",
        "checkpoint_pattern": PROJECT_ROOT / "outputs/checkpoints/deeplabv3_resnet50/deeplabv3_resnet50_fullval_fold{fold}_aug-full_crop320_pretrained/best_model.pt",
        "type": "deeplab",
    },
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (48, 145, 196),
    2: (57, 168, 91),
    3: (218, 164, 37),
    4: (177, 58, 48),
}

CASE_ORDER = [
    "overall_best",
    "c2_gain",
    "c4_tradeoff",
    "deeplab_gap",
    "challenging_case",
]

CASE_LABELS = {
    "overall_best": "High-agreement case",
    "c2_gain": "C2-sensitive gain",
    "c4_tradeoff": "C4 trade-off case",
    "deeplab_gap": "External-baseline gap",
    "challenging_case": "Challenging case",
}


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def normalize_image(image):
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))

    return image.astype(np.float32)


def build_model(model_name, fold, device):
    info = MODEL_INFO[model_name]
    module = import_module(info["script"], f"module_{model_name.replace(' ', '_').replace('-', '_')}_{fold}")

    ckpt_path = Path(str(info["checkpoint_pattern"]).format(fold=fold))

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_name} fold {fold}: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if info["type"] == "resunet":
        model = module.ResUNetDS(in_channels=3, num_classes=NUM_CLASSES, base_ch=32)
    elif info["type"] == "deeplab":
        model = module.build_deeplabv3_resnet50(num_classes=NUM_CLASSES, pretrained_backbone=False)
    else:
        raise ValueError(f"Unknown model type: {info['type']}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_mask(model, image, device, model_type):
    x = normalize_image(image)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        if model_type == "deeplab":
            logits = model(x)["out"]
        else:
            logits, _ = model(x)

    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    return pred


def dice_for_class(pred, true, cls):
    p = pred == cls
    t = true == cls

    denom = p.sum() + t.sum()

    if denom == 0:
        return np.nan

    return float(2.0 * np.logical_and(p, t).sum() / denom)


def per_image_metrics(pred, true):
    out = {}

    dice_values = []
    minority_values = []

    for cls in range(1, NUM_CLASSES):
        d = dice_for_class(pred, true, cls)
        out[f"dice_c{cls}"] = d

        if not np.isnan(d):
            dice_values.append(d)

        if cls in [2, 3, 4] and not np.isnan(d):
            minority_values.append(d)

    out["mean_dice_present_fg"] = float(np.nanmean(dice_values)) if len(dice_values) else np.nan
    out["minority_dice_present"] = float(np.nanmean(minority_values)) if len(minority_values) else np.nan

    return out


def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        out[mask == cls] = color

    return out


def overlay_mask(image, mask, alpha=0.45):
    color = colorize_mask(mask)
    fg = mask > 0

    overlay = image.copy().astype(np.float32)
    overlay[fg] = (1 - alpha) * overlay[fg] + alpha * color[fg]

    return np.clip(overlay, 0, 255).astype(np.uint8)


def foreground_boundary(mask):
    binary = (mask > 0).astype(np.uint8)
    edges = cv2.Canny(binary * 255, 50, 150)
    return edges > 0


def overlay_boundary(image, mask):
    out = image.copy()
    boundary = foreground_boundary(mask)
    out[boundary] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def evaluate_all_validation_images(device):
    records = []

    for fold in range(5):
        print(f"Evaluating fold {fold}")

        val_csv = SPLIT_DIR / f"fold_{fold}_val.csv"
        val_df = pd.read_csv(val_csv)

        models = {}
        for model_name in MODEL_INFO:
            models[model_name] = build_model(model_name, fold, device)

        for _, row in val_df.iterrows():
            image = load_image(row["image_path"])
            true_mask = load_mask(row["mask_label_path"])

            record = {
                "fold": fold,
                "image_id": int(row["image_id"]),
                "image_path": row["image_path"],
                "mask_label_path": row["mask_label_path"],
                "has_c2": int(np.any(true_mask == 2)),
                "has_c4": int(np.any(true_mask == 4)),
                "foreground_pixels": int((true_mask > 0).sum()),
                "c2_pixels": int((true_mask == 2).sum()),
                "c4_pixels": int((true_mask == 4).sum()),
            }

            preds = {}

            for model_name, model in models.items():
                model_type = MODEL_INFO[model_name]["type"]
                pred = predict_mask(model, image, device, model_type)
                preds[model_name] = pred

                metrics = per_image_metrics(pred, true_mask)

                prefix = model_name.lower().replace(" ", "_").replace("-", "_")
                for k, v in metrics.items():
                    record[f"{prefix}_{k}"] = v

            record["proposed_minus_resunet_mean_dice"] = (
                record["proposed_amc_ordinal_mean_dice_present_fg"]
                - record["resunet_ds_mean_dice_present_fg"]
            )

            record["proposed_minus_resunet_c2_dice"] = (
                record["proposed_amc_ordinal_dice_c2"]
                - record["resunet_ds_dice_c2"]
            )

            record["resunet_minus_proposed_c4_dice"] = (
                record["resunet_ds_dice_c4"]
                - record["proposed_amc_ordinal_dice_c4"]
            )

            record["proposed_minus_deeplab_mean_dice"] = (
                record["proposed_amc_ordinal_mean_dice_present_fg"]
                - record["deeplabv3_resnet50_mean_dice_present_fg"]
            )

            records.append(record)

        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(records)

    return metrics_df


def choose_cases(metrics_df):
    selected = []

    candidate = metrics_df[metrics_df["foreground_pixels"] > 0].copy()

    # 1. Overall high-agreement case
    idx = candidate["proposed_amc_ordinal_mean_dice_present_fg"].idxmax()
    selected.append(("overall_best", idx))

    # 2. C2-sensitive gain
    c2_df = candidate[candidate["has_c2"] == 1].copy()
    if len(c2_df):
        idx = c2_df["proposed_minus_resunet_c2_dice"].idxmax()
        selected.append(("c2_gain", idx))

    # 3. C4 trade-off where ResUNet is better than proposed
    c4_df = candidate[candidate["has_c4"] == 1].copy()
    if len(c4_df):
        idx = c4_df["resunet_minus_proposed_c4_dice"].idxmax()
        selected.append(("c4_tradeoff", idx))

    # 4. Proposed vs external baseline gap
    idx = candidate["proposed_minus_deeplab_mean_dice"].idxmax()
    selected.append(("deeplab_gap", idx))

    # 5. Challenging case: proposed lower but foreground exists
    idx = candidate["proposed_amc_ordinal_mean_dice_present_fg"].idxmin()
    selected.append(("challenging_case", idx))

    rows = []
    used = set()

    for case_type, idx in selected:
        row = metrics_df.loc[idx].copy()
        key = (int(row["fold"]), int(row["image_id"]))

        if key in used:
            continue

        used.add(key)
        row["case_type"] = case_type
        row["case_label"] = CASE_LABELS.get(case_type, case_type)
        rows.append(row)

    return pd.DataFrame(rows)


def generate_case_predictions(case_df, device):
    case_outputs = []

    for _, case in case_df.iterrows():
        fold = int(case["fold"])

        print(f"Generating qualitative case: {case['case_type']} | fold {fold} | image {case['image_id']}")

        image = load_image(case["image_path"])
        true_mask = load_mask(case["mask_label_path"])

        models = {}
        preds = {}

        for model_name in MODEL_INFO:
            models[model_name] = build_model(model_name, fold, device)
            preds[model_name] = predict_mask(
                models[model_name],
                image,
                device,
                MODEL_INFO[model_name]["type"]
            )

        case_outputs.append({
            "case": case,
            "image": image,
            "true_mask": true_mask,
            "preds": preds,
        })

        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return case_outputs


def make_montage(case_outputs):
    columns = [
        "Image",
        "Ground truth",
        "ResUNet-DS",
        "Proposed",
        "DeepLabV3",
    ]

    n_rows = len(case_outputs)
    n_cols = len(columns)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13.2, 2.65 * n_rows),
        facecolor="white"
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, item in enumerate(case_outputs):
        case = item["case"]
        image = item["image"]
        true_mask = item["true_mask"]
        preds = item["preds"]

        panels = [
            image,
            overlay_mask(image, true_mask, alpha=0.50),
            overlay_mask(image, preds["ResUNet-DS"], alpha=0.50),
            overlay_mask(image, preds["Proposed AMC-Ordinal"], alpha=0.50),
            overlay_mask(image, preds["DeepLabV3-ResNet50"], alpha=0.50),
        ]

        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.axis("off")

            if r == 0:
                ax.text(
                    0.5,
                    1.04,
                    columns[c],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold"
                )

            if c == 0:
                label = case["case_label"]
                fold = int(case["fold"])
                image_id = int(case["image_id"])
                ax.text(
                    -0.05,
                    0.5,
                    f"{label}\nFold {fold}, ID {image_id}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    rotation=90
                )

    plt.subplots_adjust(wspace=0.02, hspace=0.08)

    out_png = OUT_FIG_DIR / "fig_qualitative_01_model_prediction_montage.png"
    out_pdf = OUT_FIG_DIR / "fig_qualitative_01_model_prediction_montage.pdf"

    manuscript_png = MANUSCRIPT_FIG_DIR / "fig_qualitative_01_model_prediction_montage.png"
    manuscript_pdf = MANUSCRIPT_FIG_DIR / "fig_qualitative_01_model_prediction_montage.pdf"

    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(manuscript_png, bbox_inches="tight", dpi=600)
    fig.savefig(manuscript_pdf, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {manuscript_png}")
    print(f"Saved: {manuscript_pdf}")


def make_legend_figure():
    fig, ax = plt.subplots(figsize=(5.8, 1.1), facecolor="white")
    ax.axis("off")

    labels = {
        1: "C1",
        2: "C2",
        3: "C3",
        4: "C4",
    }

    x = 0.05
    for cls in [1, 2, 3, 4]:
        color = np.array(CLASS_COLORS[cls]) / 255.0

        ax.add_patch(
            plt.Rectangle(
                (x, 0.35),
                0.07,
                0.28,
                transform=ax.transAxes,
                color=color,
                ec="#202428",
                lw=0.6
            )
        )

        ax.text(
            x + 0.085,
            0.49,
            labels[cls],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10
        )

        x += 0.22

    out_png = OUT_FIG_DIR / "fig_qualitative_00_class_color_legend.png"
    out_pdf = OUT_FIG_DIR / "fig_qualitative_00_class_color_legend.pdf"

    manuscript_png = MANUSCRIPT_FIG_DIR / "fig_qualitative_00_class_color_legend.png"
    manuscript_pdf = MANUSCRIPT_FIG_DIR / "fig_qualitative_00_class_color_legend.pdf"

    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(manuscript_png, bbox_inches="tight", dpi=600)
    fig.savefig(manuscript_pdf, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {manuscript_png}")
    print(f"Saved: {manuscript_pdf}")


def main():
    print("=" * 90)
    print("Generating qualitative prediction figures")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metrics_df = evaluate_all_validation_images(device)

    metrics_path = OUT_TABLE_DIR / "qualitative_per_image_metrics_all_models.csv"
    metrics_df.to_csv(metrics_path, index=False)
    metrics_df.to_csv(MANUSCRIPT_TABLE_DIR / "qualitative_per_image_metrics_all_models.csv", index=False)

    case_df = choose_cases(metrics_df)

    case_path = OUT_TABLE_DIR / "qualitative_selected_cases.csv"
    case_df.to_csv(case_path, index=False)
    case_df.to_csv(MANUSCRIPT_TABLE_DIR / "qualitative_selected_cases.csv", index=False)

    print()
    print("Selected cases:")
    print(case_df[[
        "case_type",
        "case_label",
        "fold",
        "image_id",
        "has_c2",
        "has_c4",
        "proposed_amc_ordinal_mean_dice_present_fg",
        "proposed_minus_resunet_c2_dice",
        "resunet_minus_proposed_c4_dice",
        "proposed_minus_deeplab_mean_dice",
    ]].to_string(index=False))

    case_outputs = generate_case_predictions(case_df, device)

    make_legend_figure()
    make_montage(case_outputs)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics_path": str(metrics_path),
        "selected_cases_path": str(case_path),
        "output_figure_dir": str(OUT_FIG_DIR),
        "manuscript_figure_dir": str(MANUSCRIPT_FIG_DIR),
        "num_selected_cases": int(len(case_df)),
    }

    report_path = REPORT_DIR / "12_qualitative_prediction_figures_summary.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("=" * 90)
    print("Qualitative figure generation completed")
    print("=" * 90)
    print(f"Metrics CSV: {metrics_path}")
    print(f"Selected cases CSV: {case_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
