from pathlib import Path
from datetime import datetime
import json
import subprocess
import os

PROJECT_ROOT = Path("/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1")

REPORT_DIR = PROJECT_ROOT / "outputs/reports"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript_assets"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

required_files = {
    "main_result_table": MANUSCRIPT_DIR / "tables/deeplabv3_all_models_compact_table.csv",
    "statistical_tests": MANUSCRIPT_DIR / "tables/all_final_models_key_statistical_tests.csv",
    "qualitative_cases": MANUSCRIPT_DIR / "tables/qualitative_selected_cases.csv",
    "xai_summary": MANUSCRIPT_DIR / "tables/xai_case_summary.csv",
    "main_metric_figure": MANUSCRIPT_DIR / "figures_results/fig_deeplabv3_01_main_metric_comparison.pdf",
    "per_class_dice_figure": MANUSCRIPT_DIR / "figures_results/fig_deeplabv3_02_per_class_dice_comparison.pdf",
    "qualitative_montage": MANUSCRIPT_DIR / "figures/fig_qualitative_01_model_prediction_montage.pdf",
    "xai_montage": MANUSCRIPT_DIR / "figures/fig_xai_01_probability_uncertainty_montage.pdf",
    "final_manuscript_results_package": MANUSCRIPT_DIR / "final_manuscript_results_package.md",
}

important_dirs = {
    "scripts": PROJECT_ROOT / "scripts",
    "logs": PROJECT_ROOT / "outputs/logs",
    "checkpoints": PROJECT_ROOT / "outputs/checkpoints",
    "metrics": PROJECT_ROOT / "outputs/metrics",
    "tables": PROJECT_ROOT / "outputs/tables",
    "figures": PROJECT_ROOT / "outputs/figures",
    "manuscript_assets": MANUSCRIPT_DIR,
}

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"ERROR: {e}"

def path_info(path):
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() and path.is_file() else None,
    }

def dir_size(path):
    return run_cmd(f'du -sh "{path}" 2>/dev/null || true')

file_status = {name: path_info(path) for name, path in required_files.items()}
dir_status = {name: dir_size(path) for name, path in important_dirs.items()}

script_files = sorted([str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "scripts").glob("*.py")])
bash_files = sorted([str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "scripts").glob("*.sh")])

checkpoint_files = sorted([str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "outputs/checkpoints").glob("**/best_model.pt")])
metric_files = sorted([str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "outputs/metrics").glob("**/best_metrics.json")])

environment = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "project_root": str(PROJECT_ROOT),
    "python_version": run_cmd("python3 --version"),
    "torch_version": run_cmd('python3 - <<EOF\nimport torch\nprint(torch.__version__)\nEOF'),
    "torchvision_version": run_cmd('python3 - <<EOF\nimport torchvision\nprint(torchvision.__version__)\nEOF'),
    "cuda_available": run_cmd('python3 - <<EOF\nimport torch\nprint(torch.cuda.is_available())\nEOF'),
    "nvidia_smi": run_cmd("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true"),
    "disk": run_cmd(f'df -h "{PROJECT_ROOT}"'),
}

report = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "file_status": file_status,
    "directory_sizes": dir_status,
    "num_python_scripts": len(script_files),
    "num_shell_scripts": len(bash_files),
    "num_best_checkpoints": len(checkpoint_files),
    "num_best_metric_files": len(metric_files),
    "python_scripts": script_files,
    "shell_scripts": bash_files,
    "best_checkpoints": checkpoint_files,
    "best_metric_files": metric_files,
    "environment": environment,
}

json_path = REPORT_DIR / "16_reproducibility_artifact_index.json"
md_path = MANUSCRIPT_DIR / "reproducibility_artifact_index.md"

with open(json_path, "w") as f:
    json.dump(report, f, indent=4)

missing = [name for name, info in file_status.items() if not info["exists"]]

md = []
md.append("# Reproducibility and Artifact Index\n")
md.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
md.append("## Required manuscript assets\n")
for name, info in file_status.items():
    status = "FOUND" if info["exists"] else "MISSING"
    md.append(f"- **{name}**: {status} — `{info['path']}`")

md.append("\n## Directory sizes\n")
for name, size in dir_status.items():
    md.append(f"- **{name}**: `{size}`")

md.append("\n## Environment\n")
for key, value in environment.items():
    md.append(f"- **{key}**: `{value}`")

md.append("\n## Artifact counts\n")
md.append(f"- Python scripts: {len(script_files)}")
md.append(f"- Shell scripts: {len(bash_files)}")
md.append(f"- Best checkpoints: {len(checkpoint_files)}")
md.append(f"- Best metric files: {len(metric_files)}")

if missing:
    md.append("\n## Missing required files\n")
    for name in missing:
        md.append(f"- {name}")
else:
    md.append("\n## Missing required files\n")
    md.append("None. All required manuscript assets are present.")

md_path.write_text("\n".join(md))

print("Saved JSON artifact index:", json_path)
print("Saved markdown artifact index:", md_path)

if missing:
    print("\nMissing files:")
    for name in missing:
        print("-", name)
else:
    print("\nAll required manuscript assets are present.")

print("\nDirectory sizes:")
for name, size in dir_status.items():
    print(f"{name}: {size}")
