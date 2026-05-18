#!/usr/bin/env bash
export nnUNet_raw="/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/phase2_acceptance_experiments/nnunet_workspace/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/phase2_acceptance_experiments/nnunet_workspace/nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/storage3/ER Segmentation/ER_IHC_Q1/phase2_acceptance_experiments/nnunet_workspace/nnUNet_results"

echo "nnU-Net environment variables set:"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

# Disable torch.compile to avoid Triton/ptxas path issues caused by spaces in project path.
export nnUNet_compile=f
export TORCHDYNAMO_DISABLE=1
