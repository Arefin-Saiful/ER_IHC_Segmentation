import json
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

QUAL_CASES_CSV = PROJECT_ROOT / "outputs/tables/qualitative/qualitative_selected_cases.csv"

OUT_TABLE_DIR = PROJECT_ROOT / "outputs/tables/xai"
OUT_FIG_DIR = PROJECT_ROOT / "outputs/figures/xai"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "manuscript_assets/figures"
MANUSCRIPT_TABLE_DIR = PROJECT_ROOT / "manuscript_assets/tables"

for d in [OUT_TABLE_DIR, OUT_FIG_DIR, REPORT_DIR, MANUSCRIPT_FIG_DIR, MANUSCRIPT_TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5

DISPLAY_CASES = ["c2_gain", "c4_tradeoff", "challenging_case"]

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
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (58, 128, 191),
    2: (66, 156, 89),
    3: (214, 162, 44),
    4: (181, 71, 56),
}

ERROR_COLORS = {
    0: (28, 28, 28),      # correct
    1: (66, 135, 245),    # under-estimation
    2: (235, 144, 52),    # over-estimation
    3: (220, 60, 60),     # missed foreground -> predicted background
    4: (146, 89, 201),    # false foreground on true background
}

ERROR_LABELS = {
    0: "Correct",
    1: "Under-estimation",
    2: "Over-estimation",
    3: "Missed foreground",
    4: "False foreground",
}

CASE_LABELS = {
    "c2_gain": "C2-sensitive gain",
    "c4_tradeoff": "C4 trade-off",
    "challenging_case": "Challenging case",
}

TEXT_COLOR = "#1F2933"
PANEL_BG = "#FBFAF7"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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


def instantiate_resunet_like(module):
    candidate_names = [
        "ResUNetDS",
        "ProposedAMCOrdinalNet",
        "ResUNet",
        "UNet",
    ]
    for name in candidate_names:
        if hasattr(module, name):
            cls = getattr(module, name)
            try:
                return cls(in_channels=3, num_classes=NUM_CLASSES, base_ch=32)
            except TypeError:
                try:
                    return cls(in_channels=3, num_classes=NUM_CLASSES)
                except TypeError:
                    return cls(num_classes=NUM_CLASSES)
    raise AttributeError("Could not find a supported model class in module.")


def build_model(model_name, fold, device):
    info = MODEL_INFO[model_name]
    module = import_module(info["script"], f"module_{model_name.replace(' ', '_').replace('-', '_')}_{fold}")

    ckpt_path = Path(str(info["checkpoint_pattern"]).format(fold=fold))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_name}, fold {fold}: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if info["type"] == "resunet":
        model = instantiate_resunet_like(module)
    else:
        raise ValueError(f"Unsupported model type: {info['type']}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def forward_model(model, image, device):
    x = normalize_image(image)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    if device.type == "cuda":
        with torch.cuda.amp.autocast():
            outputs = model(x)
    else:
        outputs = model(x)

    if isinstance(outputs, (tuple, list)):
        logits = outputs[0]
    else:
        logits = outputs

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
    pred = np.argmax(probs, axis=0).astype(np.uint8)

    return pred, probs


def dice_for_class(pred, true, cls):
    p = pred == cls
    t = true == cls
    denom = p.sum() + t.sum()
    if denom == 0:
        return np.nan
    return float(2.0 * np.logical_and(p, t).sum() / denom)


def mean_dice_present(pred, true):
    vals = []
    for cls in range(1, NUM_CLASSES):
        d = dice_for_class(pred, true, cls)
        if not np.isnan(d):
            vals.append(d)
    return float(np.mean(vals)) if len(vals) else np.nan


def entropy_map(prob_map):
    eps = 1e-8
    ent = -np.sum(prob_map * np.log(prob_map + eps), axis=0)
    ent = ent / np.log(prob_map.shape[0])
    return ent.astype(np.float32)


def make_ordinal_error_map(true_mask, pred_mask):
    err = np.zeros_like(true_mask, dtype=np.uint8)

    correct = pred_mask == true_mask
    err[correct] = 0

    missed_fg = (true_mask > 0) & (pred_mask == 0)
    err[missed_fg] = 3

    false_fg = (true_mask == 0) & (pred_mask > 0)
    err[false_fg] = 4

    under = (true_mask > 0) & (pred_mask > 0) & (pred_mask < true_mask)
    err[under] = 1

    over = (true_mask > 0) & (pred_mask > 0) & (pred_mask > true_mask)
    err[over] = 2

    return err


def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        out[mask == cls] = color
    return out


def colorize_error_map(err_map):
    h, w = err_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for code, color in ERROR_COLORS.items():
        out[err_map == code] = color
    return out


def overlay_mask(image, mask, alpha=0.48):
    color = colorize_mask(mask)
    fg = mask > 0
    out = image.copy().astype(np.float32)
    out[fg] = (1 - alpha) * out[fg] + alpha * color[fg]
    return np.clip(out, 0, 255).astype(np.uint8)


def safe_mean(arr, cond):
    vals = arr[cond]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def choose_cases():
    if not QUAL_CASES_CSV.exists():
        raise FileNotFoundError(f"Missing qualitative cases CSV: {QUAL_CASES_CSV}")

    df = pd.read_csv(QUAL_CASES_CSV)

    selected = []
    for case_type in DISPLAY_CASES:
        sub = df[df["case_type"] == case_type].copy()
        if len(sub):
            selected.append(sub.iloc[0])

    if len(selected) == 0:
        raise RuntimeError("No selected qualitative cases found for XAI.")

    out_df = pd.DataFrame(selected).reset_index(drop=True)
    return out_df


def build_models_for_fold_cache(folds, device):
    cache = {}
    for fold in folds:
        cache[fold] = {}
        for model_name in MODEL_INFO:
            print(f"Loading {model_name} | fold {fold}")
            cache[fold][model_name] = build_model(model_name, fold, device)
    return cache


def generate_xai_outputs(case_df, device):
    folds = sorted(case_df["fold"].astype(int).unique().tolist())
    model_cache = build_models_for_fold_cache(folds, device)

    outputs = []
    summary_rows = []

    for _, case in case_df.iterrows():
        fold = int(case["fold"])
        image_id = int(case["image_id"])

        print(f"Processing XAI case | {case['case_type']} | fold {fold} | image {image_id}")

        image = load_image(case["image_path"])
        true_mask = load_mask(case["mask_label_path"])

        resunet_pred, resunet_probs = forward_model(model_cache[fold]["ResUNet-DS"], image, device)
        proposed_pred, proposed_probs = forward_model(model_cache[fold]["Proposed AMC-Ordinal"], image, device)

        proposed_entropy = entropy_map(proposed_probs)
        proposed_err_map = make_ordinal_error_map(true_mask, proposed_pred)

        row = {
            "case_type": case["case_type"],
            "case_label": CASE_LABELS.get(case["case_type"], case["case_type"]),
            "fold": fold,
            "image_id": image_id,
            "image_path": case["image_path"],
            "mask_label_path": case["mask_label_path"],
            "has_c2": int(np.any(true_mask == 2)),
            "has_c4": int(np.any(true_mask == 4)),
            "resunet_mean_dice_present_fg": mean_dice_present(resunet_pred, true_mask),
            "proposed_mean_dice_present_fg": mean_dice_present(proposed_pred, true_mask),
            "resunet_dice_c2": dice_for_class(resunet_pred, true_mask, 2),
            "proposed_dice_c2": dice_for_class(proposed_pred, true_mask, 2),
            "resunet_dice_c4": dice_for_class(resunet_pred, true_mask, 4),
            "proposed_dice_c4": dice_for_class(proposed_pred, true_mask, 4),
            "delta_mean_dice_present_fg": mean_dice_present(proposed_pred, true_mask) - mean_dice_present(resunet_pred, true_mask),
            "delta_dice_c2": dice_for_class(proposed_pred, true_mask, 2) - dice_for_class(resunet_pred, true_mask, 2),
            "delta_dice_c4": dice_for_class(proposed_pred, true_mask, 4) - dice_for_class(resunet_pred, true_mask, 4),
            "proposed_mean_prob_gt_c2": safe_mean(proposed_probs[2], true_mask == 2),
            "proposed_mean_prob_gt_c4": safe_mean(proposed_probs[4], true_mask == 4),
            "proposed_mean_uncertainty_fg": safe_mean(proposed_entropy, true_mask > 0),
            "proposed_mean_uncertainty_error_pixels": safe_mean(proposed_entropy, proposed_err_map != 0),
            "proposed_error_rate_fg": float(np.mean(proposed_err_map[true_mask > 0] != 0)) if np.any(true_mask > 0) else np.nan,
        }
        summary_rows.append(row)

        outputs.append({
            "case_meta": row,
            "image": image,
            "true_mask": true_mask,
            "resunet_pred": resunet_pred,
            "proposed_pred": proposed_pred,
            "proposed_prob_c2": proposed_probs[2],
            "proposed_prob_c4": proposed_probs[4],
            "proposed_entropy": proposed_entropy,
            "proposed_err_map": proposed_err_map,
        })

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return outputs, pd.DataFrame(summary_rows)


def save_legends():
    fig, ax = plt.subplots(figsize=(9.6, 1.8), facecolor="white")
    ax.axis("off")

    ax.text(0.01, 0.85, "Mask classes", transform=ax.transAxes, ha="left", va="center", fontsize=10, fontweight="bold")
    x = 0.01
    for cls in [1, 2, 3, 4]:
        color = np.array(CLASS_COLORS[cls]) / 255.0
        ax.add_patch(Rectangle((x, 0.50), 0.04, 0.18, transform=ax.transAxes, color=color, ec="#202428", lw=0.6))
        ax.text(x + 0.05, 0.59, f"C{cls}", transform=ax.transAxes, ha="left", va="center", fontsize=9)
        x += 0.12

    ax.text(0.48, 0.85, "Ordinal error map", transform=ax.transAxes, ha="left", va="center", fontsize=10, fontweight="bold")
    x = 0.48
    for code in [0, 1, 2, 3, 4]:
        color = np.array(ERROR_COLORS[code]) / 255.0
        ax.add_patch(Rectangle((x, 0.50), 0.03, 0.18, transform=ax.transAxes, color=color, ec="#202428", lw=0.6))
        ax.text(x + 0.035, 0.59, ERROR_LABELS[code], transform=ax.transAxes, ha="left", va="center", fontsize=8.5)
        x += 0.11

    out_png = OUT_FIG_DIR / "fig_xai_00_legends.png"
    out_pdf = OUT_FIG_DIR / "fig_xai_00_legends.pdf"
    man_png = MANUSCRIPT_FIG_DIR / "fig_xai_00_legends.png"
    man_pdf = MANUSCRIPT_FIG_DIR / "fig_xai_00_legends.pdf"

    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(man_png, bbox_inches="tight", dpi=600)
    fig.savefig(man_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {man_png}")
    print(f"Saved: {man_pdf}")


def make_xai_figure(xai_outputs):
    columns = [
        "Image",
        "Ground truth",
        "ResUNet-DS",
        "Proposed",
        "C2 probability",
        "C4 probability",
        "Uncertainty",
        "Ordinal error",
    ]

    n_rows = len(xai_outputs)
    n_cols = len(columns)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16.5, 3.15 * n_rows),
        facecolor="white"
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, item in enumerate(xai_outputs):
        meta = item["case_meta"]

        delta_c2 = meta["delta_dice_c2"]
        delta_c4 = meta["delta_dice_c4"]
        delta_mean = meta["delta_mean_dice_present_fg"]

        row_label = (
            f"{meta['case_label']}\n"
            f"Fold {meta['fold']}, ID {meta['image_id']}\n"
            f"ΔMean={delta_mean:+.3f} | ΔC2={delta_c2:+.3f} | ΔC4={delta_c4:+.3f}"
        )

        panels = [
            ("rgb", item["image"]),
            ("rgb", overlay_mask(item["image"], item["true_mask"], alpha=0.50)),
            ("rgb", overlay_mask(item["image"], item["resunet_pred"], alpha=0.50)),
            ("rgb", overlay_mask(item["image"], item["proposed_pred"], alpha=0.50)),
            ("heat_c2", item["proposed_prob_c2"]),
            ("heat_c4", item["proposed_prob_c4"]),
            ("heat_unc", item["proposed_entropy"]),
            ("rgb", colorize_error_map(item["proposed_err_map"])),
        ]

        for c, (ptype, panel) in enumerate(panels):
            ax = axes[r, c]

            if ptype == "rgb":
                ax.imshow(panel)
            elif ptype == "heat_c2":
                ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
            elif ptype == "heat_c4":
                ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
            elif ptype == "heat_unc":
                ax.imshow(panel, cmap="viridis", vmin=0, vmax=1)

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
                ax.text(
                    -0.08,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.3,
                    rotation=90
                )

    plt.subplots_adjust(wspace=0.02, hspace=0.08)

    out_png = OUT_FIG_DIR / "fig_xai_01_probability_uncertainty_montage.png"
    out_pdf = OUT_FIG_DIR / "fig_xai_01_probability_uncertainty_montage.pdf"
    man_png = MANUSCRIPT_FIG_DIR / "fig_xai_01_probability_uncertainty_montage.png"
    man_pdf = MANUSCRIPT_FIG_DIR / "fig_xai_01_probability_uncertainty_montage.pdf"

    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(man_png, bbox_inches="tight", dpi=600)
    fig.savefig(man_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {man_png}")
    print(f"Saved: {man_pdf}")


def main():
    print("=" * 90)
    print("Generating XAI probability, uncertainty, and ordinal error figures")
    print("=" * 90)

    setup_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    case_df = choose_cases()

    print()
    print("Selected XAI cases:")
    print(case_df[["case_type", "case_label", "fold", "image_id", "has_c2", "has_c4"]].to_string(index=False))

    xai_outputs, summary_df = generate_xai_outputs(case_df, device)

    summary_path = OUT_TABLE_DIR / "xai_case_summary.csv"
    man_summary_path = MANUSCRIPT_TABLE_DIR / "xai_case_summary.csv"

    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(man_summary_path, index=False)

    save_legends()
    make_xai_figure(xai_outputs)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "xai_case_summary_path": str(summary_path),
        "manuscript_xai_case_summary_path": str(man_summary_path),
        "figure_dir": str(OUT_FIG_DIR),
        "manuscript_figure_dir": str(MANUSCRIPT_FIG_DIR),
        "num_cases": int(len(summary_df)),
        "cases": summary_df[["case_type", "case_label", "fold", "image_id"]].to_dict(orient="records"),
    }

    report_path = REPORT_DIR / "14_xai_probability_uncertainty_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("XAI case summary:")
    print(summary_df.to_string(index=False))

    print()
    print("=" * 90)
    print("XAI figure generation completed")
    print("=" * 90)
    print(f"Summary CSV: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
