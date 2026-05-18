from pathlib import Path
import json
import pandas as pd
import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")
DS = PROJECT_ROOT / "phase2_acceptance_experiments/nnunet_workspace/nnUNet_raw/Dataset501_ERIHCSeg"

imagesTr = DS / "imagesTr"
labelsTr = DS / "labelsTr"
dataset_json = DS / "dataset.json"

print("=" * 90)
print("nnU-Net raw dataset debug")
print("=" * 90)
print("Dataset:", DS)
print("dataset.json exists:", dataset_json.exists())

if dataset_json.exists():
    print(json.dumps(json.load(open(dataset_json)), indent=4))

image_files = sorted(imagesTr.glob("*.png"))
label_files = sorted(labelsTr.glob("*.png"))

print("\nCounts")
print("imagesTr:", len(image_files))
print("labelsTr:", len(label_files))

image_cases = {p.name.replace("_0000.png", "") for p in image_files}
label_cases = {p.name.replace(".png", "") for p in label_files}

print("image-only cases:", sorted(image_cases - label_cases)[:20])
print("label-only cases:", sorted(label_cases - image_cases)[:20])

bad_rows = []
unique_values_all = set()
shape_rows = []

for case in sorted(image_cases & label_cases):
    img_path = imagesTr / f"{case}_0000.png"
    lbl_path = labelsTr / f"{case}.png"

    row = {
        "case": case,
        "image_path": str(img_path),
        "label_path": str(lbl_path),
        "image_exists": img_path.exists(),
        "label_exists": lbl_path.exists(),
        "image_is_symlink": img_path.is_symlink(),
        "label_is_symlink": lbl_path.is_symlink(),
        "error": "",
    }

    try:
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        img_arr = np.array(img)
        lbl_arr = np.array(lbl)

        row["image_mode"] = img.mode
        row["label_mode"] = lbl.mode
        row["image_shape"] = str(img_arr.shape)
        row["label_shape"] = str(lbl_arr.shape)
        row["image_dtype"] = str(img_arr.dtype)
        row["label_dtype"] = str(lbl_arr.dtype)

        unique_vals = sorted(np.unique(lbl_arr).tolist())
        row["label_unique"] = str(unique_vals)
        unique_values_all.update(unique_vals)

        image_hw = img_arr.shape[:2]
        label_hw = lbl_arr.shape[:2]

        row["same_hw"] = image_hw == label_hw
        row["label_min"] = int(lbl_arr.min())
        row["label_max"] = int(lbl_arr.max())

        if image_hw != label_hw:
            row["error"] += "SHAPE_MISMATCH; "

        unexpected = [v for v in unique_vals if v not in [0, 1, 2, 3, 4]]
        if unexpected:
            row["error"] += f"UNEXPECTED_LABELS_{unexpected}; "

        if img.mode not in ["RGB", "RGBA", "L"]:
            row["error"] += f"UNUSUAL_IMAGE_MODE_{img.mode}; "

        if lbl_arr.ndim != 2:
            row["error"] += "LABEL_NOT_2D; "

    except Exception as e:
        row["error"] += repr(e)

    shape_rows.append(row)

    if row["error"]:
        bad_rows.append(row)

df = pd.DataFrame(shape_rows)
bad = pd.DataFrame(bad_rows)

out_dir = PROJECT_ROOT / "phase2_acceptance_experiments/outputs/tables"
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out_dir / "29e_nnunet_dataset_integrity_all_cases.csv", index=False)
bad.to_csv(out_dir / "29e_nnunet_dataset_integrity_bad_cases.csv", index=False)

print("\nAll label values found:")
print(sorted(unique_values_all))

print("\nBad cases:", len(bad_rows))
if len(bad_rows):
    print(bad[["case", "image_mode", "label_mode", "image_shape", "label_shape", "label_unique", "error"]].head(40).to_string(index=False))
else:
    print("No PIL-level problems found.")

print("\nShape/mode summary:")
print(df.groupby(["image_mode", "label_mode", "image_shape", "label_shape"]).size().reset_index(name="count").to_string(index=False))

print("\nSaved:")
print(out_dir / "29e_nnunet_dataset_integrity_all_cases.csv")
print(out_dir / "29e_nnunet_dataset_integrity_bad_cases.csv")
