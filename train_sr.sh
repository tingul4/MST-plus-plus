#!/bin/bash

# Training MST++ with 2x super-resolution
python train_code/train.py \
    --method mst_plus_plus \
    --batch_size 8 \
    --end_epoch 10 \
    --init_lr 4e-4 \
    --data_root /ssd7/ICASSP_2026_Hyper-Object_Challenge/track2/dataset \
    --patch_size 128 \
    --stride 16 \
    --gpu_id 5 \
    --upscale_factor 2 \
    --outf ./exp/mst_plus_plus_sr2x/ \
    --pretrained_model_path /ssd7/ICASSP_2026_Hyper-Object_Challenge/track2/MST-plus-plus/exp/mst_plus_plus_sr2x/2025_10_08_16_56_56/MSTPP_3_35.249081.pth