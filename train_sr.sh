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
    --outf ./exp/mst_plus_plus/ \
    --use_ema \
    --ema_decay 0.999 \
    --lr_restart_epochs 10 \
    --lr_restart_mult 2 \
    --resume /home/danielchen/MST-plus-plus/exp/mst_plus_plus/2025_11_24_17_44_18/MSTPP_26_40.602375.pth