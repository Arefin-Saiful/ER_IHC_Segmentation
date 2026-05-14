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

CKPT_DIR = OUT_ROOT / "checkpoints/proposed_amc_ordinal"
METRIC_DIR = OUT_ROOT / "metrics/proposed_amc_ordinal"
FIG_DIR = OUT_ROOT / "figures/results/proposed_amc_ordinal"
REPORT_DIR = OUT_ROOT / "reports"

NUM_CLASSES = 5

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
    for d in [CKPT_DIR, METRIC_DIR, FIG_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_image(path):
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def pad_if_needed(image, mask, crop_size):
    h, w = mask.shape

    if h >= crop_size and w >= crop_size:
        return image, mask

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

    return image, mask


def random_crop(image, mask, crop_size):
    image, mask = pad_if_needed(image, mask, crop_size)

    h, w = mask.shape

    y = np.random.randint(0, h - crop_size + 1)
    x = np.random.randint(0, w - crop_size + 1)

    return image[y:y + crop_size, x:x + crop_size], mask[y:y + crop_size, x:x + crop_size]


def crop_around_point(image, mask, cy, cx, crop_size):
    image, mask = pad_if_needed(image, mask, crop_size)

    h, w = mask.shape

    y = int(cy - crop_size // 2)
    x = int(cx - crop_size // 2)

    y = max(0, min(y, h - crop_size))
    x = max(0, min(x, w - crop_size))

    return image[y:y + crop_size, x:x + crop_size], mask[y:y + crop_size, x:x + crop_size]


def choose_class_center(mask, candidate_classes, class_sampling_weights=None):
    available_classes = []

    for cls in candidate_classes:
        if np.any(mask == cls):
            available_classes.append(cls)

    if len(available_classes) == 0:
        return None

    if class_sampling_weights is None:
        selected_class = random.choice(available_classes)
    else:
        weights = np.array([class_sampling_weights.get(cls, 1.0) for cls in available_classes], dtype=np.float64)
        weights = weights / weights.sum()
        selected_class = int(np.random.choice(available_classes, p=weights))

    coords = np.argwhere(mask == selected_class)

    if len(coords) == 0:
        return None

    idx = np.random.randint(0, len(coords))
    cy, cx = coords[idx]

    return int(cy), int(cx), int(selected_class)


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
    def __init__(
        self,
        csv_path,
        crop_size=320,
        mode="train",
        aug_mode="full",
        use_amc=False,
        amc_p=0.25,
        class_sampling_weights=None
    ):
        self.df = pd.read_csv(csv_path)
        self.crop_size = crop_size
        self.mode = mode
        self.aug_mode = aug_mode
        self.use_amc = use_amc
        self.amc_p = amc_p
        self.class_sampling_weights = class_sampling_weights or {}

    def set_amc_probability(self, value):
        self.amc_p = float(value)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = read_image(row["image_path"])
        mask = read_mask(row["mask_label_path"])

        sampled_class = -1

        if self.mode == "train":
            if self.use_amc:
                r = np.random.rand()

                if r < self.amc_p:
                    center = choose_class_center(
                        mask,
                        candidate_classes=[2, 3, 4],
                        class_sampling_weights=self.class_sampling_weights
                    )

                    if center is None:
                        center = choose_class_center(
                            mask,
                            candidate_classes=[1, 2, 3, 4],
                            class_sampling_weights=None
                        )

                    if center is not None:
                        cy, cx, sampled_class = center
                        image, mask = crop_around_point(image, mask, cy, cx, self.crop_size)
                    else:
                        image, mask = random_crop(image, mask, self.crop_size)

                elif r < self.amc_p + 0.20:
                    center = choose_class_center(
                        mask,
                        candidate_classes=[1, 2, 3, 4],
                        class_sampling_weights=None
                    )

                    if center is not None:
                        cy, cx, sampled_class = center
                        image, mask = crop_around_point(image, mask, cy, cx, self.crop_size)
                    else:
                        image, mask = random_crop(image, mask, self.crop_size)

                else:
                    image, mask = random_crop(image, mask, self.crop_size)
            else:
                image, mask = random_crop(image, mask, self.crop_size)

            if self.aug_mode in ["geometric", "full"]:
                image, mask = apply_geometric_augmentation(image, mask)

            if self.aug_mode in ["stain_light", "full"]:
                image = apply_stain_light_augmentation(image)

        else:
            pass

        image = normalize_image(image)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": int(row["image_id"]),
            "sampled_class": int(sampled_class),
        }


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.relu(out + identity)

        return out


class SCSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        reduced = max(channels // reduction, 4)

        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResidualBlock(out_ch + skip_ch, out_ch)
        self.scse = SCSEBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        x = self.scse(x)
        return x


class ResUNetDS(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base_ch=32):
        super().__init__()

        self.enc1 = nn.Sequential(
            ResidualBlock(in_channels, base_ch),
            SCSEBlock(base_ch)
        )

        self.enc2 = nn.Sequential(
            ResidualBlock(base_ch, base_ch * 2),
            SCSEBlock(base_ch * 2)
        )

        self.enc3 = nn.Sequential(
            ResidualBlock(base_ch * 2, base_ch * 4),
            SCSEBlock(base_ch * 4)
        )

        self.enc4 = nn.Sequential(
            ResidualBlock(base_ch * 4, base_ch * 8),
            SCSEBlock(base_ch * 8)
        )

        self.pool = nn.MaxPool2d(2)

        self.bridge = nn.Sequential(
            ResidualBlock(base_ch * 8, base_ch * 16),
            SCSEBlock(base_ch * 16)
        )

        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, num_classes, kernel_size=1)

        self.aux4 = nn.Conv2d(base_ch * 8, num_classes, kernel_size=1)
        self.aux3 = nn.Conv2d(base_ch * 4, num_classes, kernel_size=1)
        self.aux2 = nn.Conv2d(base_ch * 2, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bridge(self.pool(e4))

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        main_logits = self.out(d1)

        aux_logits = [
            self.aux4(d4),
            self.aux3(d3),
            self.aux2(d2),
        ]

        return main_logits, aux_logits


def compute_class_weights(train_csv):
    df = pd.read_csv(train_csv)

    counts = np.array([df[f"pix_c{i}"].sum() for i in range(NUM_CLASSES)], dtype=np.float64)
    freq = counts / counts.sum()

    weights = 1.0 / np.sqrt(freq + 1e-8)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 4.0)

    return torch.tensor(weights, dtype=torch.float32)


def compute_minority_sampling_weights(train_csv, c2_boost=1.00, c3_boost=1.00, c4_boost=1.75):
    df = pd.read_csv(train_csv)

    counts = {
        2: float(df["pix_c2"].sum()),
        3: float(df["pix_c3"].sum()),
        4: float(df["pix_c4"].sum()),
    }

    boosts = {
        2: float(c2_boost),
        3: float(c3_boost),
        4: float(c4_boost),
    }

    raw = {}
    for cls in counts:
        raw[cls] = boosts[cls] * (1.0 / np.sqrt(counts[cls] + 1.0))

    total = sum(raw.values())

    return {cls: raw[cls] / total for cls in raw}


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


def focal_tversky_loss(logits, targets, alpha=0.7, beta=0.3, gamma=2.0, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float()

    losses = []

    for cls in range(1, NUM_CLASSES):
        p = probs[:, cls]
        t = targets_onehot[:, cls]

        tp = (p * t).sum(dim=(1, 2))
        fp = (p * (1.0 - t)).sum(dim=(1, 2))
        fn = ((1.0 - p) * t).sum(dim=(1, 2))

        tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        loss = torch.pow(1.0 - tversky, gamma)

        losses.append(loss)

    return torch.stack(losses, dim=0).mean()


def ordinal_emd_loss_foreground(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)

    fg_mask = targets > 0

    if fg_mask.sum() == 0:
        return logits.sum() * 0.0

    fg_probs = probs[:, 1:5, :, :]
    fg_probs = fg_probs.permute(0, 2, 3, 1)
    fg_probs = fg_probs[fg_mask]

    fg_probs = fg_probs / (fg_probs.sum(dim=1, keepdim=True) + eps)

    fg_targets = targets[fg_mask] - 1
    target_onehot = F.one_hot(fg_targets, num_classes=4).float()

    pred_cdf = torch.cumsum(fg_probs, dim=1)
    target_cdf = torch.cumsum(target_onehot, dim=1)

    loss = torch.abs(pred_cdf - target_cdf).mean()

    return loss


def deep_supervision_loss(aux_logits, targets, ce_loss_fn):
    total = 0.0

    for aux in aux_logits:
        aux_up = F.interpolate(aux, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        loss_ce = ce_loss_fn(aux_up, targets)
        loss_dice = soft_dice_loss(aux_up, targets, include_background=False)

        total = total + loss_ce + loss_dice

    return total / max(len(aux_logits), 1)


def ramp_weight(epoch, warmup_epochs, ramp_epochs, max_weight):
    if epoch <= warmup_epochs:
        return 0.0

    progress = (epoch - warmup_epochs) / max(ramp_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)

    return max_weight * progress


def compute_ordinal_metrics(confusion):
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

    metrics.update(compute_ordinal_metrics(confusion))

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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    class_weights,
    epoch,
    args
):
    model.train()

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    running = {
        "loss": 0.0,
        "ce": 0.0,
        "dice": 0.0,
        "ft": 0.0,
        "ordinal": 0.0,
        "aux": 0.0,
    }

    n_batches = 0

    lambda_ft = ramp_weight(epoch, args.ft_warmup, args.ft_ramp, args.ft_weight)
    lambda_ord = ramp_weight(epoch, args.ordinal_warmup, args.ordinal_ramp, args.ordinal_weight)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits, aux_logits = model(images)

            loss_ce = ce_loss_fn(logits, masks)
            loss_dice = soft_dice_loss(logits, masks, include_background=False)
            loss_ft = focal_tversky_loss(
                logits,
                masks,
                alpha=args.ft_alpha,
                beta=args.ft_beta,
                gamma=args.ft_gamma
            )
            loss_ord = ordinal_emd_loss_foreground(logits, masks)
            loss_aux = deep_supervision_loss(aux_logits, masks, ce_loss_fn)

            loss = (
                args.ce_weight * loss_ce
                + args.dice_weight * loss_dice
                + lambda_ft * loss_ft
                + lambda_ord * loss_ord
                + args.aux_weight * loss_aux
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running["loss"] += loss.item()
        running["ce"] += loss_ce.item()
        running["dice"] += loss_dice.item()
        running["ft"] += loss_ft.item()
        running["ordinal"] += loss_ord.item()
        running["aux"] += loss_aux.item()

        n_batches += 1

    out = {k: v / max(n_batches, 1) for k, v in running.items()}
    out["lambda_ft"] = lambda_ft
    out["lambda_ord"] = lambda_ord

    return out


@torch.no_grad()
def validate(model, loader, device, class_weights, args):
    model.eval()

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    running_loss = 0.0
    n_batches = 0

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits, _ = model(images)

            loss_ce = ce_loss_fn(logits, masks)
            loss_dice = soft_dice_loss(logits, masks, include_background=False)
            loss = args.ce_weight * loss_ce + args.dice_weight * loss_dice

        preds = torch.argmax(logits, dim=1)
        confusion = update_confusion_matrix(confusion, preds, masks)

        running_loss += loss.item()
        n_batches += 1

    metrics = compute_metrics_from_confusion(confusion)
    metrics["val_loss"] = running_loss / max(n_batches, 1)

    return metrics, confusion


def update_amc_probability(current_p, minority_dice, args):
    if np.isnan(minority_dice):
        return current_p

    new_p = current_p + args.amc_gain * (args.amc_target - minority_dice)
    new_p = max(args.amc_p_min, min(args.amc_p_max, new_p))

    return float(new_p)


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
    ax2.plot(df["epoch"], df["amc_p"], linewidth=1.4, linestyle=":", label="AMC probability")
    ax2.set_ylabel("Dice / AMC probability")

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
    parser.add_argument("--epochs", type=int, default=120)
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
    parser.add_argument("--aux-weight", type=float, default=0.10)

    parser.add_argument("--ft-weight", type=float, default=0.50)
    parser.add_argument("--ft-alpha", type=float, default=0.70)
    parser.add_argument("--ft-beta", type=float, default=0.30)
    parser.add_argument("--ft-gamma", type=float, default=2.0)
    parser.add_argument("--ft-warmup", type=int, default=5)
    parser.add_argument("--ft-ramp", type=int, default=10)

    parser.add_argument("--ordinal-weight", type=float, default=0.10)
    parser.add_argument("--ordinal-warmup", type=int, default=10)
    parser.add_argument("--ordinal-ramp", type=int, default=20)

    parser.add_argument("--amc-p0", type=float, default=0.25)
    parser.add_argument("--amc-p-min", type=float, default=0.20)
    parser.add_argument("--amc-p-max", type=float, default=0.85)
    parser.add_argument("--amc-target", type=float, default=0.70)
    parser.add_argument("--amc-gain", type=float, default=0.20)

    parser.add_argument("--c2-sample-boost", type=float, default=1.00)
    parser.add_argument("--c3-sample-boost", type=float, default=1.00)
    parser.add_argument("--c4-sample-boost", type=float, default=1.75)

    parser.add_argument("--score-minority-weight", type=float, default=0.40)
    parser.add_argument("--score-mean-weight", type=float, default=0.25)
    parser.add_argument("--score-c4-weight", type=float, default=0.20)
    parser.add_argument("--score-kappa-weight", type=float, default=0.10)
    parser.add_argument("--score-ord-mae-weight", type=float, default=0.05)

    args = parser.parse_args()

    setup_dirs()
    seed_everything(args.seed)

    train_csv = SPLIT_DIR / f"fold_{args.fold}_train.csv"
    val_csv = SPLIT_DIR / f"fold_{args.fold}_val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train split: {train_csv}")

    if not val_csv.exists():
        raise FileNotFoundError(f"Missing val split: {val_csv}")

    run_name = (
        f"proposed_amc_ordinal_fullval_fold{args.fold}"
        f"_aug-{args.aug_mode}"
        f"_base{args.base_ch}"
        f"_crop{args.crop_size}"
        f"_ord{args.ordinal_weight}"
        f"_ft{args.ft_weight}"
        f"_amcmax{args.amc_p_max}"
        f"_gain{args.amc_gain}"
        f"_target{args.amc_target}"
        f"_c4protect"
        f"_c2b{args.c2_sample_boost}"
        f"_c3b{args.c3_sample_boost}"
        f"_c4b{args.c4_sample_boost}"
        f"_scorec4{args.score_c4_weight}"
    )

    run_ckpt_dir = CKPT_DIR / run_name
    run_metric_dir = METRIC_DIR / run_name
    run_fig_dir = FIG_DIR / run_name

    for d in [run_ckpt_dir, run_metric_dir, run_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = compute_class_weights(train_csv)
    minority_sampling_weights = compute_minority_sampling_weights(
        train_csv,
        c2_boost=args.c2_sample_boost,
        c3_boost=args.c3_sample_boost,
        c4_boost=args.c4_sample_boost,
    )

    print("=" * 90)
    print("Phase 5: Proposed ResUNet-DS + AMC + ordinal EMD training")
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

    print("Class weights:")
    for i, w in enumerate(class_weights.tolist()):
        print(f"  C{i}: {w:.4f}")

    print("Minority crop sampling weights:")
    for k, v in minority_sampling_weights.items():
        print(f"  C{k}: {v:.4f}")

    train_dataset = ERIHCDataset(
        train_csv,
        crop_size=args.crop_size,
        mode="train",
        aug_mode=args.aug_mode,
        use_amc=True,
        amc_p=args.amc_p0,
        class_sampling_weights=minority_sampling_weights
    )

    val_dataset = ERIHCDataset(
        val_csv,
        crop_size=args.crop_size,
        mode="val",
        aug_mode="none",
        use_amc=False
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

    model = ResUNetDS(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_ch=args.base_ch
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -1.0
    best_epoch = -1
    best_metrics = None

    current_amc_p = args.amc_p0

    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_dataset.set_amc_probability(current_amc_p)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            class_weights=class_weights,
            epoch=epoch,
            args=args
        )

        val_metrics, confusion = validate(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            args=args
        )

        selection_score = (
            args.score_minority_weight * val_metrics["minority_dice_c2_c3_c4"]
            + args.score_mean_weight * val_metrics["mean_dice_no_bg"]
            + args.score_c4_weight * val_metrics["dice_c4"]
            + args.score_kappa_weight * val_metrics["weighted_kappa_fg"]
            - args.score_ord_mae_weight * val_metrics["ordinal_mae_fg"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_dice": train_stats["dice"],
            "train_focal_tversky": train_stats["ft"],
            "train_ordinal": train_stats["ordinal"],
            "train_aux": train_stats["aux"],
            "lambda_ft": train_stats["lambda_ft"],
            "lambda_ord": train_stats["lambda_ord"],
            "amc_p": current_amc_p,
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
            f"kappa={val_metrics['weighted_kappa_fg']:.4f} | "
            f"ord_mae={val_metrics['ordinal_mae_fg']:.4f} | "
            f"amc_p={current_amc_p:.3f} | "
            f"ord_w={train_stats['lambda_ord']:.3f} | "
            f"ft_w={train_stats['lambda_ft']:.3f} | "
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
                "minority_sampling_weights": minority_sampling_weights,
                "metrics": best_metrics,
            }

            torch.save(checkpoint, run_ckpt_dir / "best_model.pt")
            save_confusion_matrix(confusion, run_metric_dir / "best_confusion_matrix.csv")

            print(f"  Saved new best checkpoint at epoch {epoch}")

        current_amc_p = update_amc_probability(
            current_p=current_amc_p,
            minority_dice=val_metrics["minority_dice_c2_c3_c4"],
            args=args
        )

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

    summary_path = REPORT_DIR / f"05_proposed_amc_ordinal_{run_name}_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 90)
    print("Proposed model training completed")
    print("=" * 90)
    print(f"Best epoch: {best_epoch}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best metrics saved: {best_metrics_path}")
    print(f"Checkpoint saved: {run_ckpt_dir / 'best_model.pt'}")
    print(f"Learning curve saved: {curve_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
