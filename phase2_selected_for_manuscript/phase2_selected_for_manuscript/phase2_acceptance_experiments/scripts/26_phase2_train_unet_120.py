import json
import time
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")
PHASE2_ROOT = PROJECT_ROOT / "phase2_acceptance_experiments"

BASE_SCRIPT = PROJECT_ROOT / "scripts/05_train_proposed_amc_ordinal.py"

SPLIT_DIR = PROJECT_ROOT / "data/splits"

CKPT_DIR = PHASE2_ROOT / "outputs/checkpoints/unet_120"
METRIC_DIR = PHASE2_ROOT / "outputs/metrics/unet_120"
FIG_DIR = PHASE2_ROOT / "outputs/figures/unet_120"
REPORT_DIR = PHASE2_ROOT / "outputs/reports"

NUM_CLASSES = 5


def import_base_module():
    spec = importlib.util.spec_from_file_location("base_proposed", str(BASE_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = import_base_module()


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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PlainUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base_ch=32):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch, base_ch)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits, [logits]


def setup_dirs():
    for d in [CKPT_DIR, METRIC_DIR, FIG_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-mode", type=str, default="full")

    # Keep loss settings comparable and simple for U-Net baseline.
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.0)

    # Disabled for plain U-Net baseline.
    parser.add_argument("--ft-weight", type=float, default=0.0)
    parser.add_argument("--ft-alpha", type=float, default=0.70)
    parser.add_argument("--ft-beta", type=float, default=0.30)
    parser.add_argument("--ft-gamma", type=float, default=2.0)
    parser.add_argument("--ft-warmup", type=int, default=5)
    parser.add_argument("--ft-ramp", type=int, default=10)

    parser.add_argument("--ordinal-weight", type=float, default=0.0)
    parser.add_argument("--ordinal-warmup", type=int, default=10)
    parser.add_argument("--ordinal-ramp", type=int, default=20)

    # Required by imported AMC update functions, but not used here.
    parser.add_argument("--amc-p0", type=float, default=0.0)
    parser.add_argument("--amc-p-min", type=float, default=0.0)
    parser.add_argument("--amc-p-max", type=float, default=0.0)
    parser.add_argument("--amc-target", type=float, default=0.70)
    parser.add_argument("--amc-gain", type=float, default=0.20)

    args = parser.parse_args()

    setup_dirs()
    base.seed_everything(args.seed)

    train_csv = SPLIT_DIR / f"fold_{args.fold}_train.csv"
    val_csv = SPLIT_DIR / f"fold_{args.fold}_val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(train_csv)

    if not val_csv.exists():
        raise FileNotFoundError(val_csv)

    run_name = (
        f"phase2_unet120_fullval_fold{args.fold}"
        f"_aug-{args.aug_mode}"
        f"_base{args.base_ch}"
        f"_crop{args.crop_size}"
    )

    run_ckpt_dir = CKPT_DIR / run_name
    run_metric_dir = METRIC_DIR / run_name
    run_fig_dir = FIG_DIR / run_name

    for d in [run_ckpt_dir, run_metric_dir, run_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = base.compute_class_weights(train_csv)

    print("=" * 90)
    print("Phase 2B: U-Net baseline retraining for 120 epochs")
    print("=" * 90)
    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Crop size: {args.crop_size}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    print("Checkpoint selection:")
    print("  score = 0.60 * minority_dice + 0.30 * mean_dice + 0.10 * weighted_kappa")

    print("Class weights:")
    for i, w in enumerate(class_weights.tolist()):
        print(f"  C{i}: {w:.4f}")

    train_dataset = base.ERIHCDataset(
        train_csv,
        crop_size=args.crop_size,
        mode="train",
        aug_mode=args.aug_mode,
        use_amc=False,
    )

    val_dataset = base.ERIHCDataset(
        val_csv,
        crop_size=args.crop_size,
        mode="val",
        aug_mode="none",
        use_amc=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = PlainUNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_ch=args.base_ch
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -1.0
    best_epoch = -1
    best_metrics = None
    history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_stats = base.train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            class_weights=class_weights,
            epoch=epoch,
            args=args,
        )

        val_metrics, confusion = base.validate(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            args=args,
        )

        selection_score = (
            0.60 * val_metrics["minority_dice_c2_c3_c4"]
            + 0.30 * val_metrics["mean_dice_no_bg"]
            + 0.10 * val_metrics["weighted_kappa_fg"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_dice": train_stats["dice"],
            "train_focal_tversky": train_stats.get("ft", 0.0),
            "train_ordinal": train_stats.get("ordinal", 0.0),
            "train_aux": train_stats.get("aux", 0.0),
            "lambda_ft": train_stats.get("lambda_ft", 0.0),
            "lambda_ord": train_stats.get("lambda_ord", 0.0),
            "amc_p": 0.0,
            "selection_score": selection_score,
            **val_metrics,
            "epoch_seconds": time.time() - epoch_start,
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_stats['loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"mean_dice={val_metrics['mean_dice_no_bg']:.4f} | "
            f"minority_dice={val_metrics['minority_dice_c2_c3_c4']:.4f} | "
            f"dice_c2={val_metrics['dice_c2']:.4f} | "
            f"dice_c4={val_metrics['dice_c4']:.4f} | "
            f"kappa={val_metrics['weighted_kappa_fg']:.4f} | "
            f"ord_mae={val_metrics['ordinal_mae_fg']:.4f} | "
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
            base.save_confusion_matrix(confusion, run_metric_dir / "best_confusion_matrix.csv")

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

    base.plot_learning_curve(history, curve_path)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 2B",
        "model": "U-Net 120 epochs",
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

    summary_path = REPORT_DIR / f"26_phase2_unet120_{run_name}_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 90)
    print("Phase 2B U-Net 120 training completed")
    print("=" * 90)
    print(f"Fold: {args.fold}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best metrics saved: {best_metrics_path}")
    print(f"Checkpoint saved: {run_ckpt_dir / 'best_model.pt'}")
    print(f"Learning curve saved: {curve_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
