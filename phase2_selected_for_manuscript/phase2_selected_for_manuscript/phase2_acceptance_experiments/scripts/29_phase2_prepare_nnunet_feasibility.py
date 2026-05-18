import json
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")
PHASE2_ROOT = PROJECT_ROOT / "phase2_acceptance_experiments"

SPLIT_DIR = PROJECT_ROOT / "data/splits"

NNUNET_ROOT = PHASE2_ROOT / "nnunet_workspace"
NNUNET_RAW = NNUNET_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_ROOT / "nnUNet_results"

DATASET_ID = 501
DATASET_NAME = "Dataset501_ERIHCSeg"
DATASET_DIR = NNUNET_RAW / DATASET_NAME

IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"

OUT_REPORT_DIR = PHASE2_ROOT / "outputs/reports"
OUT_TABLE_DIR = PHASE2_ROOT / "outputs/tables"

OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=20)
        return True, out.strip()
    except Exception as e:
        return False, str(e)


def check_nnunet():
    checks = {}

    ok_import, out_import = run_cmd("python3 - <<'PY'\nimport nnunetv2\nprint(nnunetv2.__file__)\nPY")
    checks["python_import_nnunetv2"] = {
        "ok": ok_import,
        "output": out_import,
    }

    for cli in [
        "nnUNetv2_plan_and_preprocess",
        "nnUNetv2_train",
        "nnUNetv2_predict",
        "nnUNetv2_evaluate_folder",
    ]:
        path = shutil.which(cli)
        checks[cli] = {
            "ok": path is not None,
            "path": path,
        }

    return checks


def read_all_cases():
    # Use fold_0_train + fold_0_val as the full 220-case set.
    # These should together cover the complete dataset exactly once.
    train = pd.read_csv(SPLIT_DIR / "fold_0_train.csv")
    val = pd.read_csv(SPLIT_DIR / "fold_0_val.csv")

    df = pd.concat([train, val], ignore_index=True)

    if "image_id" not in df.columns:
        raise ValueError("Expected column `image_id` in split CSV.")

    if "image_path" not in df.columns:
        raise ValueError("Expected column `image_path` in split CSV.")

    if "mask_label_path" not in df.columns:
        raise ValueError("Expected column `mask_label_path` in split CSV.")

    df = df.drop_duplicates(subset=["image_id"]).sort_values("image_id").reset_index(drop=True)

    if len(df) != 220:
        print(f"Warning: expected 220 cases, found {len(df)} cases.")

    return df


def case_name(image_id):
    return f"ERIHC_{int(image_id):04d}"


def safe_symlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        os.symlink(src, dst)
        return "symlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def prepare_raw_dataset(df):
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS, IMAGES_TR, LABELS_TR]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, row in df.iterrows():
        cid = case_name(row["image_id"])

        src_img = Path(row["image_path"])
        src_lbl = Path(row["mask_label_path"])

        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")

        if not src_lbl.exists():
            raise FileNotFoundError(f"Missing label: {src_lbl}")

        dst_img = IMAGES_TR / f"{cid}_0000.png"
        dst_lbl = LABELS_TR / f"{cid}.png"

        mode_img = safe_symlink_or_copy(src_img, dst_img)
        mode_lbl = safe_symlink_or_copy(src_lbl, dst_lbl)

        rows.append({
            "image_id": int(row["image_id"]),
            "case": cid,
            "source_image": str(src_img),
            "source_label": str(src_lbl),
            "nnunet_image": str(dst_img),
            "nnunet_label": str(dst_lbl),
            "image_mode": mode_img,
            "label_mode": mode_lbl,
        })

    mapping = pd.DataFrame(rows)

    mapping_path = OUT_TABLE_DIR / "phase2_nnunet_case_mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    dataset_json = {
        "channel_names": {
            "0": "red",
            "1": "green",
            "2": "blue"
        },
        "labels": {
            "background": 0,
            "C1": 1,
            "C2": 2,
            "C3": 3,
            "C4": 4
        },
        "numTraining": int(len(mapping)),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO"
    }

    dataset_json_path = DATASET_DIR / "dataset.json"

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    return mapping_path, dataset_json_path


def create_env_script():
    env_path = NNUNET_ROOT / "activate_nnunet_env.sh"

    content = f'''#!/usr/bin/env bash
export nnUNet_raw="{NNUNET_RAW}"
export nnUNet_preprocessed="{NNUNET_PREPROCESSED}"
export nnUNet_results="{NNUNET_RESULTS}"

echo "nnU-Net environment variables set:"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
'''

    env_path.write_text(content)
    env_path.chmod(0o755)

    return env_path


def create_custom_split_script():
    p = PHASE2_ROOT / "scripts/29b_create_nnunet_custom_splits.py"

    content = f'''import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")
PHASE2_ROOT = PROJECT_ROOT / "phase2_acceptance_experiments"

SPLIT_DIR = PROJECT_ROOT / "data/splits"
NNUNET_PREPROCESSED = PHASE2_ROOT / "nnunet_workspace/nnUNet_preprocessed"
DATASET_NAME = "{DATASET_NAME}"

OUT_PATH = NNUNET_PREPROCESSED / DATASET_NAME / "splits_final.json"

def case_name(image_id):
    return f"ERIHC_{{int(image_id):04d}}"

splits = []

for fold in range(5):
    train = pd.read_csv(SPLIT_DIR / f"fold_{{fold}}_train.csv")
    val = pd.read_csv(SPLIT_DIR / f"fold_{{fold}}_val.csv")

    train_cases = [case_name(x) for x in train["image_id"].tolist()]
    val_cases = [case_name(x) for x in val["image_id"].tolist()]

    splits.append({{
        "train": train_cases,
        "val": val_cases
    }})

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PATH, "w") as f:
    json.dump(splits, f, indent=4)

print("Custom five-fold splits saved:")
print(OUT_PATH)
for i, s in enumerate(splits):
    print(f"Fold {{i}}: train={{len(s['train'])}}, val={{len(s['val'])}}")
'''

    p.write_text(content)
    return p


def create_train_runner_scripts():
    fold0_runner = PHASE2_ROOT / "scripts/run_29c_nnunet_train_fold0_feasibility.sh"
    all_runner = PHASE2_ROOT / "scripts/run_29d_nnunet_train_all_folds.sh"

    fold0_runner.write_text(f'''#!/usr/bin/env bash
set -euo pipefail

cd "{PROJECT_ROOT}"

source "{NNUNET_ROOT}/activate_nnunet_env.sh"

echo "Planning and preprocessing Dataset {DATASET_ID} if needed..."
nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d --verify_dataset_integrity

echo "Creating custom five-fold split..."
python3 phase2_acceptance_experiments/scripts/29b_create_nnunet_custom_splits.py

echo "Training nnU-Net fold 0 feasibility run..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {DATASET_ID} 2d 0

echo "nnU-Net fold 0 feasibility training completed."
''')
    fold0_runner.chmod(0o755)

    all_runner.write_text(f'''#!/usr/bin/env bash
set -euo pipefail

cd "{PROJECT_ROOT}"

source "{NNUNET_ROOT}/activate_nnunet_env.sh"

echo "Planning and preprocessing Dataset {DATASET_ID} if needed..."
nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d --verify_dataset_integrity

echo "Creating custom five-fold split..."
python3 phase2_acceptance_experiments/scripts/29b_create_nnunet_custom_splits.py

for FOLD in 0 1 2 3 4
do
  echo "================================================================================"
  echo "Training nnU-Net 2d | Fold $FOLD"
  echo "================================================================================"
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {DATASET_ID} 2d "$FOLD"
done

echo "All nnU-Net folds completed."
''')
    all_runner.chmod(0o755)

    return fold0_runner, all_runner


def main():
    print("=" * 90)
    print("Phase 2E: nnU-Net feasibility preparation")
    print("=" * 90)

    checks = check_nnunet()

    df = read_all_cases()
    mapping_path, dataset_json_path = prepare_raw_dataset(df)

    env_script = create_env_script()
    split_script = create_custom_split_script()
    fold0_runner, all_runner = create_train_runner_scripts()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase 2E",
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "nnunet_root": str(NNUNET_ROOT),
        "nnunet_raw": str(NNUNET_RAW),
        "nnunet_preprocessed": str(NNUNET_PREPROCESSED),
        "nnunet_results": str(NNUNET_RESULTS),
        "dataset_dir": str(DATASET_DIR),
        "imagesTr": str(IMAGES_TR),
        "labelsTr": str(LABELS_TR),
        "num_cases": int(len(df)),
        "case_mapping": str(mapping_path),
        "dataset_json": str(dataset_json_path),
        "env_script": str(env_script),
        "custom_split_script": str(split_script),
        "fold0_runner": str(fold0_runner),
        "all_folds_runner": str(all_runner),
        "nnunet_checks": checks,
    }

    report_path = OUT_REPORT_DIR / "29_phase2_nnunet_feasibility_preparation_summary.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("nnU-Net availability checks:")
    print(json.dumps(checks, indent=4))

    print()
    print("Prepared nnU-Net raw dataset:")
    print(DATASET_DIR)
    print(f"Cases: {len(df)}")
    print(f"Dataset JSON: {dataset_json_path}")
    print(f"Case mapping: {mapping_path}")
    print(f"Environment script: {env_script}")
    print(f"Fold-0 runner: {fold0_runner}")
    print(f"All-fold runner: {all_runner}")
    print(f"Report: {report_path}")

    if not checks["python_import_nnunetv2"]["ok"] or not checks["nnUNetv2_train"]["ok"]:
        print()
        print("WARNING: nnUNetv2 is not fully available.")
        print("Do not run training yet. Install/check nnUNetv2 first.")
    else:
        print()
        print("nnUNetv2 appears available. Next step: run the fold-0 feasibility runner first.")


if __name__ == "__main__":
    main()
