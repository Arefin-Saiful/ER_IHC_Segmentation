#!/usr/bin/env bash
set -euo pipefail

cd "/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1"

for FOLD in 0 1 2 3 4
do
  echo "================================================================================"
  echo "Running Phase 2B U-Net 120 | Fold ${FOLD}"
  echo "================================================================================"

  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python3 phase2_acceptance_experiments/scripts/26_phase2_train_unet_120.py \
    --fold "${FOLD}" \
    --epochs 120 \
    --batch-size 2 \
    --crop-size 320 \
    --base-ch 32 \
    --aug-mode full \
    --num-workers 2

  echo "Completed Phase 2B U-Net 120 | Fold ${FOLD}"
done

echo "All Phase 2B U-Net 120 folds completed."
