#!/usr/bin/env bash
set -euo pipefail

cd "/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1"

for FOLD in 0 1 2 3 4
do
  echo "============================================================"
  echo "Running DeepLabV3-ResNet50 baseline | Fold ${FOLD}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python3 scripts/10_train_deeplabv3_resnet50_baseline.py \
    --fold "${FOLD}" \
    --epochs 120 \
    --batch-size 2 \
    --crop-size 320 \
    --aug-mode full \
    --num-workers 2

  echo "Completed DeepLabV3-ResNet50 fold ${FOLD}"
done

echo "All DeepLabV3-ResNet50 folds completed."
