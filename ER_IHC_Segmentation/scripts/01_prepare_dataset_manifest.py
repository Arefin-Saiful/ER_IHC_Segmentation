import os
import re
import json
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

IMAGE_ZIP = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Image.zip")
MASK_ZIP = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Mask.zip")

RAW_IMAGE_DIR = PROJECT_ROOT / "data/raw/images_zip_extract"
RAW_MASK_DIR = PROJECT_ROOT / "data/raw/masks_zip_extract"

PROCESSED_MASK_LABEL_DIR = PROJECT_ROOT / "data/processed/masks_label"

MANIFEST_DIR = PROJECT_ROOT / "data/manifests"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/eda"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

COLOR_TO_LABEL = {
    (0, 0, 0): 0,
    (0, 159, 255): 1,
    (0, 255, 0): 2,
    (255, 216, 0): 3,
    (255, 0, 0): 4,
}

LABEL_TO_NAME = {
    0: "Background",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}

def extract_zip_if_needed(zip_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_file = output_dir / ".extracted_ok"

    if marker_file.exists():
        print(f"Already extracted, skipping: {zip_path}")
        return

    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    marker_file.write_text(f"Extracted from {zip_path} at {datetime.now().isoformat()}\n")
    print(f"Extraction completed: {output_dir}")

def find_image_files(root_dir):
    files = []
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return sorted(files)

def extract_id_from_filename(path):
    numbers = re.findall(r"\d+", path.stem)
    if len(numbers) == 0:
        return None
    return int(numbers[-1])

def convert_mask_to_label(mask_path):
    mask_img = Image.open(mask_path)

    if mask_img.mode == "RGBA":
        mask_arr_rgba = np.array(mask_img)
        alpha = mask_arr_rgba[:, :, 3]
        rgb = mask_arr_rgba[:, :, :3]
    else:
        rgb = np.array(mask_img.convert("RGB"))
        alpha = np.ones(rgb.shape[:2], dtype=np.uint8) * 255

    h, w, _ = rgb.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)

    known = np.zeros((h, w), dtype=bool)

    for color, label in COLOR_TO_LABEL.items():
        color_array = np.array(color, dtype=np.uint8)
        match = np.all(rgb == color_array, axis=-1)
        label_mask[match] = label
        known = known | match

    transparent_background = alpha == 0
    label_mask[transparent_background] = 0
    known = known | transparent_background

    unknown_pixels = int((~known).sum())

    unknown_colors = []
    if unknown_pixels > 0:
        unknown_rgb = rgb[~known]
        unique_unknown = np.unique(unknown_rgb.reshape(-1, 3), axis=0)
        unknown_colors = [tuple(map(int, color)) for color in unique_unknown[:30]]

    return label_mask, unknown_pixels, unknown_colors

def main():
    print("=" * 90)
    print("Phase 1: ER-IHC dataset extraction, mask conversion, and manifest creation")
    print("=" * 90)

    for directory in [
        RAW_IMAGE_DIR,
        RAW_MASK_DIR,
        PROCESSED_MASK_LABEL_DIR,
        MANIFEST_DIR,
        TABLE_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not IMAGE_ZIP.exists():
        raise FileNotFoundError(f"Image zip not found: {IMAGE_ZIP}")

    if not MASK_ZIP.exists():
        raise FileNotFoundError(f"Mask zip not found: {MASK_ZIP}")

    extract_zip_if_needed(IMAGE_ZIP, RAW_IMAGE_DIR)
    extract_zip_if_needed(MASK_ZIP, RAW_MASK_DIR)

    image_files = find_image_files(RAW_IMAGE_DIR)
    mask_files = find_image_files(RAW_MASK_DIR)

    print(f"Found image files: {len(image_files)}")
    print(f"Found mask files : {len(mask_files)}")

    image_map = {}
    for path in image_files:
        image_id = extract_id_from_filename(path)
        if image_id is not None:
            image_map[image_id] = path

    mask_map = {}
    for path in mask_files:
        mask_id = extract_id_from_filename(path)
        if mask_id is not None:
            mask_map[mask_id] = path

    common_ids = sorted(set(image_map.keys()) & set(mask_map.keys()))
    image_only_ids = sorted(set(image_map.keys()) - set(mask_map.keys()))
    mask_only_ids = sorted(set(mask_map.keys()) - set(image_map.keys()))

    print(f"Matched image-mask pairs: {len(common_ids)}")
    print(f"Image-only IDs: {len(image_only_ids)}")
    print(f"Mask-only IDs : {len(mask_only_ids)}")

    rows = []
    unknown_records = []

    for image_id in common_ids:
        image_path = image_map[image_id]
        mask_path = mask_map[image_id]

        label_mask, unknown_pixels, unknown_colors = convert_mask_to_label(mask_path)

        label_mask_path = PROCESSED_MASK_LABEL_DIR / f"mask_{image_id:04d}.png"
        Image.fromarray(label_mask, mode="L").save(label_mask_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        total_pixels = int(label_mask.size)

        class_counts = {}
        class_ratios = {}

        for cls in range(5):
            pixel_count = int((label_mask == cls).sum())
            class_counts[f"pix_c{cls}"] = pixel_count
            class_ratios[f"ratio_c{cls}"] = pixel_count / total_pixels

        foreground_pixels = sum(class_counts[f"pix_c{cls}"] for cls in [1, 2, 3, 4])
        minority_pixels = sum(class_counts[f"pix_c{cls}"] for cls in [2, 3, 4])

        row = {
            "image_id": image_id,
            "image_path": str(image_path),
            "mask_original_path": str(mask_path),
            "mask_label_path": str(label_mask_path),
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "foreground_pixels": int(foreground_pixels),
            "foreground_ratio": foreground_pixels / total_pixels,
            "minority_pixels_c2_c3_c4": int(minority_pixels),
            "minority_ratio_c2_c3_c4": minority_pixels / total_pixels,
            "unknown_pixels": unknown_pixels,
        }

        row.update(class_counts)
        row.update(class_ratios)

        rows.append(row)

        if unknown_pixels > 0:
            unknown_records.append({
                "image_id": image_id,
                "mask_path": str(mask_path),
                "unknown_pixels": unknown_pixels,
                "unknown_colors_first_30": str(unknown_colors),
            })

    manifest = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)

    manifest_path = MANIFEST_DIR / "manifest_er_ihc.csv"
    manifest.to_csv(manifest_path, index=False)

    class_summary_rows = []
    total_dataset_pixels = manifest["total_pixels"].sum()

    for cls in range(5):
        total_class_pixels = int(manifest[f"pix_c{cls}"].sum())
        class_summary_rows.append({
            "class_id": cls,
            "class_name": LABEL_TO_NAME[cls],
            "total_pixels": total_class_pixels,
            "dataset_pixel_ratio": total_class_pixels / total_dataset_pixels,
            "images_present": int((manifest[f"pix_c{cls}"] > 0).sum()),
            "mean_image_ratio": float(manifest[f"ratio_c{cls}"].mean()),
            "std_image_ratio": float(manifest[f"ratio_c{cls}"].std()),
            "min_image_ratio": float(manifest[f"ratio_c{cls}"].min()),
            "max_image_ratio": float(manifest[f"ratio_c{cls}"].max()),
        })

    class_summary = pd.DataFrame(class_summary_rows)

    class_summary_path = TABLE_DIR / "class_distribution_summary.csv"
    class_summary.to_csv(class_summary_path, index=False)

    unknown_df = pd.DataFrame(unknown_records)
    unknown_path = TABLE_DIR / "unknown_mask_colors.csv"
    unknown_df.to_csv(unknown_path, index=False)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "image_zip": str(IMAGE_ZIP),
        "mask_zip": str(MASK_ZIP),
        "raw_image_files": len(image_files),
        "raw_mask_files": len(mask_files),
        "matched_pairs": len(common_ids),
        "image_only_ids": image_only_ids,
        "mask_only_ids": mask_only_ids,
        "masks_with_unknown_pixels": int((manifest["unknown_pixels"] > 0).sum()),
        "total_unknown_pixels": int(manifest["unknown_pixels"].sum()),
        "manifest_path": str(manifest_path),
        "class_summary_path": str(class_summary_path),
        "unknown_colors_path": str(unknown_path),
    }

    summary_path = REPORT_DIR / "01_dataset_preparation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 90)
    print("Dataset preparation completed")
    print("=" * 90)
    print(f"Manifest saved: {manifest_path}")
    print(f"Class summary saved: {class_summary_path}")
    print(f"Unknown colors saved: {unknown_path}")
    print(f"Summary saved: {summary_path}")
    print()
    print("Class distribution summary:")
    print(class_summary.to_string(index=False))
    print()
    print("First five manifest rows:")
    print(manifest.head().to_string(index=False))

if __name__ == "__main__":
    main()
