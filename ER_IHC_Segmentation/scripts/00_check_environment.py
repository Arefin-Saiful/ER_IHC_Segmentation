import os
import sys
import json
import zipfile
import shutil
import platform
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

IMAGE_ZIP = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Image.zip")
MASK_ZIP = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Mask.zip")

REPORT_DIR = PROJECT_ROOT / "outputs/reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def gb(x):
    return x / (1024 ** 3)

def check_zip(zip_path, name):
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"Path: {zip_path}")

    if not zip_path.exists():
        print("Status: NOT FOUND")
        return {
            "exists": False,
            "path": str(zip_path)
        }

    size_gb = gb(zip_path.stat().st_size)
    print("Status: FOUND")
    print(f"Size: {size_gb:.4f} GB")

    result = {
        "exists": True,
        "path": str(zip_path),
        "size_gb": size_gb,
        "is_zip": zipfile.is_zipfile(zip_path)
    }

    if not zipfile.is_zipfile(zip_path):
        print("Warning: File is not a valid zip.")
        return result

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        image_like = [
            x for x in names
            if x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]

        total_uncompressed = sum(info.file_size for info in z.infolist())

        print(f"Total entries: {len(names)}")
        print(f"Image-like entries: {len(image_like)}")
        print(f"Uncompressed size: {gb(total_uncompressed):.4f} GB")
        print("First 10 image-like files:")
        for item in image_like[:10]:
            print(f"  {item}")

        result.update({
            "total_entries": len(names),
            "image_like_entries": len(image_like),
            "uncompressed_size_gb": gb(total_uncompressed),
            "first_10_image_like_files": image_like[:10]
        })

    return result

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "version_not_available")
        print(f"{package_name}: FOUND ({version})")
        return True
    except Exception as e:
        print(f"{package_name}: MISSING or ERROR ({e})")
        return False

def main():
    print("=" * 80)
    print("ER-IHC Q1 Environment Check")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    print()
    print("=" * 80)
    print("Disk check")
    print("=" * 80)
    usage = shutil.disk_usage(PROJECT_ROOT)
    print(f"Total: {gb(usage.total):.2f} GB")
    print(f"Used : {gb(usage.used):.2f} GB")
    print(f"Free : {gb(usage.free):.2f} GB")

    if gb(usage.free) < 15:
        print("Warning: Free space is very low for training.")
    elif gb(usage.free) < 30:
        print("Warning: Free space is limited. We will use low-storage mode.")
    else:
        print("Storage is acceptable.")

    print()
    image_zip_info = check_zip(IMAGE_ZIP, "Image zip check")
    print()
    mask_zip_info = check_zip(MASK_ZIP, "Mask zip check")

    print()
    print("=" * 80)
    print("Package check")
    print("=" * 80)
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "Pillow": "PIL",
        "matplotlib": "matplotlib",
        "sklearn": "sklearn",
        "cv2": "cv2",
        "torch": "torch",
        "torchvision": "torchvision"
    }

    package_status = {}
    for package_name, import_name in packages.items():
        package_status[package_name] = check_package(package_name, import_name)

    print()
    print("=" * 80)
    print("CUDA check")
    print("=" * 80)
    cuda_info = {}

    try:
        import torch
        cuda_info["torch_version"] = torch.__version__
        cuda_info["cuda_available"] = torch.cuda.is_available()
        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        cuda_info["error"] = str(e)
        print(f"CUDA check failed: {e}")

    print()
    print("=" * 80)
    print("nvidia-smi")
    print("=" * 80)
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        print(result.stdout if result.stdout else result.stderr)
    except Exception as e:
        print(f"nvidia-smi failed: {e}")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "free_gb": gb(usage.free),
        "image_zip": image_zip_info,
        "mask_zip": mask_zip_info,
        "package_status": package_status,
        "cuda_info": cuda_info,
        "low_storage_mode": gb(usage.free) < 30
    }

    out_path = REPORT_DIR / "00_environment_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 80)
    print("Environment check completed")
    print("=" * 80)
    print(f"Saved summary: {out_path}")

if __name__ == "__main__":
    main()
