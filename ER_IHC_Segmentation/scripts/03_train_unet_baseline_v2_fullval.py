import os
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

SPLIT_DIR = PROJECT_ROOT / "data/splits"
OUT_ROOT = PROJECT_ROOT / "outputs"
LOG_DIR = OUT_ROOT / "logs"
CKPT_DIR = OUT_ROOT / "checkpoints/unet_baseline"
METRIC_DIR = OUT_ROOT / "metrics/unet_baseline"
FIG_DIR = OUT_ROOT / "figures/results/unet_baseline"
REPORT_DIR = OUT_ROOT / "reports"

NUM_CLASSES = 5
IGNORE_BG_FOR_MEAN = True

CLASS_NAMES = {
    0: "Background",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_dirs():
    for d in [LOG_DIR, CKPT_DIR, METRIC_DIR, FIG_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_image(path):
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def random_crop(image, mask, crop_size):
    h, w = mask.shape

    if h < crop_size or w < crop_size:
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)

        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect"
        )
        mask = np.pad(
            mask,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0
        )
        h, w = mask.shape

    y = np.random.randint(0, h - crop_size + 1)
    x = np.random.randint(0, w - crop_size + 1)

    return image[y:y + crop_size, x:x + crop_size], mask[y:y + crop_size, x:x + crop_size]


def center_crop_or_resize(image, mask, size):
    h, w = mask.shape

    if h >= size and w >= size:
        y = (h - size) // 2
        x = (w - size) // 2
        image = image[y:y + size, x:x + size]
        mask = mask[y:y + size, x:x + size]
        return image, mask

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def apply_geometric_augmentation(image, mask):
    if np.random.rand() < 0.5:
        image = np.ascontiguousarray(np.flip(image, axis=1))
        mask = np.ascontiguousarray(np.flip(mask, axis=1))

    if np.random.rand() < 0.5:
        image = np.ascontiguousarray(np.flip(image, axis=0))
        mask = np.ascontiguousarray(np.flip(mask, axis=0))

    if np.random.rand() < 0.5:
        k = np.random.randint(0, 4)
        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))

    if np.random.rand() < 0.45:
        h, w = mask.shape
        angle = np.random.uniform(-12, 12)
        scale = np.random.uniform(0.92, 1.08)
        tx = np.random.uniform(-0.04, 0.04) * w
        ty = np.random.uniform(-0.04, 0.04) * h

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        mask = cv2.warpAffine(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    return image, mask


def apply_stain_light_augmentation(image):
    image = image.astype(np.float32)

    if np.random.rand() < 0.45:
        contrast = np.random.uniform(0.88, 1.12)
        brightness = np.random.uniform(-10, 10)
        image = image * contrast + brightness

    if np.random.rand() < 0.35:
        gamma = 2.0 ** np.random.uniform(-0.22, 0.22)
        image_norm = np.clip(image / 255.0, 0, 1)
        image = (image_norm ** gamma) * 255.0

    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    if np.random.rand() < 0.35:
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] += np.random.uniform(-3, 3)
        hsv[:, :, 1] *= np.random.uniform(0.90, 1.10)
        hsv[:, :, 2] *= np.random.uniform(0.92, 1.08)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        image_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if np.random.rand() < 0.15:
        k = random.choice([3, 5])
        image_uint8 = cv2.GaussianBlur(image_uint8, (k, k), 0)

    if np.random.rand() < 0.15:
        noise = np.random.normal(0, 3.0, size=image_uint8.shape)
        image_uint8 = np.clip(image_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image_uint8


def normalize_image(image):
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))

    return image.astype(np.float32)


class ERIHCDataset(Dataset):
    def __init__(self, csv_path, crop_size=320, mode="train", aug_mode="full"):
        self.df = pd.read_csv(csv_path)
        self.crop_size = crop_size
        self.mode = mode
        self.aug_mode = aug_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = read_image(row["image_path"])
        mask = read_mask(row["mask_label_path"])

        if self.mode == "train":
            image, mask = random_crop(image, mask, self.crop_size)

            if self.aug_mode in ["geometric", "full"]:
                image, mask = apply_geometric_augmentation(image, mask)

            if self.aug_mode in ["stain_light", "full"]:
                image = apply_stain_light_augmentation(image)

        else:
            # Final validation uses the complete ER-IHC patch instead of a center crop.
            # The dataset patches are 512x512, which is compatible with the U-Net downsampling path.
            # This produces manuscript-valid full-patch validation metrics.
            pass

        image = normalize_image(image)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": int(row["image_id"]),
        }


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base_ch=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


def compute_class_weights(train_csv):
    df = pd.read_csv(train_csv)

    counts = np.array([df[f"pix_c{i}"].sum() for i in range(NUM_CLASSES)], dtype=np.float64)
    freq = counts / counts.sum()

    weights = 1.0 / np.sqrt(freq + 1e-8)
    weights = weights / weights.mean()

    weights = np.clip(weights, 0.25, 4.0)

    return torch.tensor(weights, dtype=torch.float32)


def soft_dice_loss(logits, targets, include_background=False, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float()

    if include_background:
        class_ids = list(range(NUM_CLASSES))
    else:
        class_ids = list(range(1, NUM_CLASSES))

    losses = []

    for cls in class_ids:
        p = probs[:, cls]
        t = targets_onehot[:, cls]

        intersection = (p * t).sum(dim=(1, 2))
        denominator = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))

        dice = (2 * intersection + eps) / (denominator + eps)
        losses.append(1.0 - dice)

    return torch.stack(losses, dim=0).mean()


def compute_metrics_from_confusion(confusion):
    metrics = {}

    dice_values = []
    iou_values = []
    precision_values = []
    recall_values = []

    for cls in range(NUM_CLASSES):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        metrics[f"dice_c{cls}"] = float(dice)
        metrics[f"iou_c{cls}"] = float(iou)
        metrics[f"precision_c{cls}"] = float(precision)
        metrics[f"recall_c{cls}"] = float(recall)

        if cls != 0:
            dice_values.append(dice)
            iou_values.append(iou)
            precision_values.append(precision)
            recall_values.append(recall)

    metrics["mean_dice_no_bg"] = float(np.mean(dice_values))
    metrics["mean_iou_no_bg"] = float(np.mean(iou_values))
    metrics["macro_precision_no_bg"] = float(np.mean(precision_values))
    metrics["macro_recall_no_bg"] = float(np.mean(recall_values))
    metrics["minority_dice_c2_c3_c4"] = float(np.mean([metrics["dice_c2"], metrics["dice_c3"], metrics["dice_c4"]]))

    return metrics


def update_confusion_matrix(confusion, preds, targets):
    preds = preds.detach().cpu().numpy().astype(np.int64).ravel()
    targets = targets.detach().cpu().numpy().astype(np.int64).ravel()

    valid = (targets >= 0) & (targets < NUM_CLASSES)
    preds = preds[valid]
    targets = targets[valid]

    inds = NUM_CLASSES * targets + preds
    cm = np.bincount(inds, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

    confusion += cm

    return confusion


def train_one_epoch(model, loader, optimizer, scaler, device, class_weights, ce_weight, dice_weight):
    model.train()

    running_loss = 0.0
    n_batches = 0

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss_ce = ce_loss_fn(logits, masks)
            loss_dice = soft_dice_loss(logits, masks, include_background=False)
            loss = ce_weight * loss_ce + dice_weight * loss_dice

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, class_weights, ce_weight, dice_weight):
    model.eval()

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    running_loss = 0.0
    n_batches = 0

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss_ce = ce_loss_fn(logits, masks)
            loss_dice = soft_dice_loss(logits, masks, include_background=False)
            loss = ce_weight * loss_ce + dice_weight * loss_dice

        preds = torch.argmax(logits, dim=1)

        confusion = update_confusion_matrix(confusion, preds, masks)

        running_loss += loss.item()
        n_batches += 1

    metrics = compute_metrics_from_confusion(confusion)
    metrics["val_loss"] = running_loss / max(n_batches, 1)

    return metrics, confusion


def plot_learning_curve(history, output_path):
    df = pd.DataFrame(history)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))

    ax1.plot(df["epoch"], df["train_loss"], linewidth=1.8, label="Train loss")
    ax1.plot(df["epoch"], df["val_loss"], linewidth=1.8, label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.45)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["mean_dice_no_bg"], linewidth=1.8, linestyle="--", label="Mean Dice")
    ax2.plot(df["epoch"], df["minority_dice_c2_c3_c4"], linewidth=1.8, linestyle="--", label="Minority Dice")
    ax2.set_ylabel("Dice")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="center right")

    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def save_confusion_matrix(confusion, output_path):
    df = pd.DataFrame(
        confusion.astype(int),
        index=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        columns=[CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    )
    df.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-mode", type=str, default="full", choices=["none", "geometric", "stain_light", "full"])
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    args = parser.parse_args()

    setup_dirs()
    seed_everything(args.seed)

    train_csv = SPLIT_DIR / f"fold_{args.fold}_train.csv"
    val_csv = SPLIT_DIR / f"fold_{args.fold}_val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train split: {train_csv}")

    if not val_csv.exists():
        raise FileNotFoundError(f"Missing val split: {val_csv}")

    run_name = f"unet_baseline_fullval_fold{args.fold}_aug-{args.aug_mode}_base{args.base_ch}_crop{args.crop_size}"
    run_ckpt_dir = CKPT_DIR / run_name
    run_metric_dir = METRIC_DIR / run_name
    run_fig_dir = FIG_DIR / run_name

    for d in [run_ckpt_dir, run_metric_dir, run_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("Phase 3A: U-Net baseline training")
    print("=" * 90)
    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Crop size: {args.crop_size}")
    print(f"Base channels: {args.base_ch}")
    print(f"Augmentation mode: {args.aug_mode}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")

    train_dataset = ERIHCDataset(
        train_csv,
        crop_size=args.crop_size,
        mode="train",
        aug_mode=args.aug_mode
    )

    val_dataset = ERIHCDataset(
        val_csv,
        crop_size=args.crop_size,
        mode="val",
        aug_mode="none"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_ch=args.base_ch
    ).to(device)

    class_weights = compute_class_weights(train_csv)

    print("Class weights:")
    for i, w in enumerate(class_weights.tolist()):
        print(f"  C{i}: {w:.4f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -1.0
    best_epoch = -1
    best_metrics = None

    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            class_weights,
            args.ce_weight,
            args.dice_weight
        )

        val_metrics, confusion = validate(
            model,
            val_loader,
            device,
            class_weights,
            args.ce_weight,
            args.dice_weight
        )

        selection_score = (
            0.60 * val_metrics["minority_dice_c2_c3_c4"]
            + 0.40 * val_metrics["mean_dice_no_bg"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_score": selection_score,
            **val_metrics,
            "epoch_seconds": time.time() - epoch_start,
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"mean_dice={val_metrics['mean_dice_no_bg']:.4f} | "
            f"minority_dice={val_metrics['minority_dice_c2_c3_c4']:.4f} | "
            f"score={selection_score:.4f} | "
            f"time={row['epoch_seconds']:.1f}s"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_metrics = row.copy()

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "args": vars(args),
                "class_weights": class_weights.tolist(),
                "metrics": best_metrics,
            }

            torch.save(checkpoint, run_ckpt_dir / "best_model.pt")
            save_confusion_matrix(confusion, run_metric_dir / "best_confusion_matrix.csv")

            print(f"  Saved new best checkpoint at epoch {epoch}")

        pd.DataFrame(history).to_csv(run_metric_dir / "training_history.csv", index=False)

    total_time = time.time() - start_time

    history_path = run_metric_dir / "training_history.csv"
    best_metrics_path = run_metric_dir / "best_metrics.json"
    config_path = run_metric_dir / "run_config.json"
    curve_path = run_fig_dir / "learning_curve.png"

    with open(best_metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=4)

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    plot_learning_curve(history, curve_path)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "total_time_seconds": total_time,
        "checkpoint": str(run_ckpt_dir / "best_model.pt"),
        "history_path": str(history_path),
        "best_metrics_path": str(best_metrics_path),
        "learning_curve": str(curve_path),
        "best_metrics": best_metrics,
    }

    summary_path = REPORT_DIR / f"03_unet_baseline_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 90)
    print("Training completed")
    print("=" * 90)
    print(f"Best epoch: {best_epoch}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best metrics saved: {best_metrics_path}")
    print(f"Checkpoint saved: {run_ckpt_dir / 'best_model.pt'}")
    print(f"Learning curve saved: {curve_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
