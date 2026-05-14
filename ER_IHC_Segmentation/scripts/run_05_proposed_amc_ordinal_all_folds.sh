#!/usr/bin/env bash
set -euo pipefail

cd "/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1"

for FOLD in 0 1 2 3 4
do
  echo "============================================================"
  echo "Running proposed AMC + ordinal model | Fold ${FOLD}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python3 scripts/05_train_proposed_amc_ordinal.py \
    --fold "${FOLD}" \
    --epochs 120 \
    --batch-size 2 \
    --crop-size 320 \
    --base-ch 32 \
    --aug-mode full \
    --aux-weight 0.10 \
    --ft-weight 0.50 \
    --ordinal-weight 0.10 \
    --amc-p0 0.25 \
    --amc-p-min 0.20 \
    --amc-p-max 0.85 \
    --amc-target 0.70 \
    --amc-gain 0.20 \
    --num-workers 2

  echo "Completed fold ${FOLD}"
done

echo "All proposed AMC + ordinal folds completed."
