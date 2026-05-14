#!/usr/bin/env bash
set -euo pipefail

cd "/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1"

for FOLD in 3 4
do
  echo "============================================================"
  echo "Running C4-protective proposed model | Fold ${FOLD}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python3 scripts/09_train_proposed_c4_protective.py \
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
    --c2-sample-boost 1.00 \
    --c3-sample-boost 1.00 \
    --c4-sample-boost 1.75 \
    --score-minority-weight 0.40 \
    --score-mean-weight 0.25 \
    --score-c4-weight 0.20 \
    --score-kappa-weight 0.10 \
    --score-ord-mae-weight 0.05 \
    --num-workers 2

  echo "Completed C4-protective fold ${FOLD}"
done

echo "C4-protective pilot completed."
