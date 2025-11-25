#!/bin/bash

# Training MST++ with 2x super-resolution
python3 train_code/train.py \
    --method mst_plus_plus \
    --batch_size 16 \
    --end_epoch 50 \
    --init_lr 4e-4 \
    --data_root /ssd1/daniel/hsi_construct \
    --patch_size 128 \
    --stride 16 \
    --gpu_id 0,1 \
    --upscale_factor 2 \
    --outf ./exp/mst_plus_plus/
    # --pretrained_model_path /home/danielchen/MST-plus-plus/exp/mst_plus_plus_sr2x/2025_10_08_16_56_56/MSTPP_3_35.249081.pth