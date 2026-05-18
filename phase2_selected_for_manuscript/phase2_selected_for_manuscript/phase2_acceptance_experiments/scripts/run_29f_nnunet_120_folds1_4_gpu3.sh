#!/usr/bin/env bash
set -euo pipefail

cd "/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1"

source phase2_acceptance_experiments/venvs/nnunetv2_erihc/bin/activate
source phase2_acceptance_experiments/nnunet_workspace/activate_nnunet_env.sh

export nnUNet_compile=f
export TORCHDYNAMO_DISABLE=1
export nnUNet_n_proc_DA=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use physical GPU 3
export CUDA_VISIBLE_DEVICES=3

for FOLD in 1 2 3 4
do
  echo "================================================================================"
  echo "Running nnU-Net 120-epoch trainer | Fold ${FOLD} | Physical GPU 3"
  echo "================================================================================"

  nnUNetv2_train 501 2d "${FOLD}" -tr nnUNetTrainer_120epochs

  echo "Completed nnU-Net 120-epoch trainer | Fold ${FOLD}"
done

echo "All nnU-Net 120-epoch folds 1-4 completed."
