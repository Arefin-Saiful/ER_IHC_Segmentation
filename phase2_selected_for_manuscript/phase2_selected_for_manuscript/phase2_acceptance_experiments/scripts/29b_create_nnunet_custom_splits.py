import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")
PHASE2_ROOT = PROJECT_ROOT / "phase2_acceptance_experiments"

SPLIT_DIR = PROJECT_ROOT / "data/splits"
NNUNET_PREPROCESSED = PHASE2_ROOT / "nnunet_workspace/nnUNet_preprocessed"
DATASET_NAME = "Dataset501_ERIHCSeg"

OUT_PATH = NNUNET_PREPROCESSED / DATASET_NAME / "splits_final.json"

def case_name(image_id):
    return f"ERIHC_{int(image_id):04d}"

splits = []

for fold in range(5):
    train = pd.read_csv(SPLIT_DIR / f"fold_{fold}_train.csv")
    val = pd.read_csv(SPLIT_DIR / f"fold_{fold}_val.csv")

    train_cases = [case_name(x) for x in train["image_id"].tolist()]
    val_cases = [case_name(x) for x in val["image_id"].tolist()]

    splits.append({
        "train": train_cases,
        "val": val_cases
    })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PATH, "w") as f:
    json.dump(splits, f, indent=4)

print("Custom five-fold splits saved:")
print(OUT_PATH)
for i, s in enumerate(splits):
    print(f"Fold {i}: train={len(s['train'])}, val={len(s['val'])}")
