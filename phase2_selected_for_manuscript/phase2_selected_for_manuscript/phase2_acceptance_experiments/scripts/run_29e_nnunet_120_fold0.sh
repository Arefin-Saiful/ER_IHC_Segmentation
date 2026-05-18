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
export CUDA_VISIBLE_DEVICES=0

echo "Running nnU-Net 120-epoch trainer | Fold 0"
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_120epochs

echo "nnU-Net 120-epoch fold 0 completed."
