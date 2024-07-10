#!/bin/bash

for checkpoint in "07-07-2024-11:43:57-epoch=34-loss_val=0.26.ckpt" "07-07-2024-14:22:10-epoch=41-loss_val=0.20.ckpt" "07-07-2024-18:04:05-epoch=43-loss_val=0.16.ckpt"  "07-07-2024-21:46:09-epoch=22-loss_val=0.14.ckpt"  "08-07-2024-00:08:52-epoch=48-loss_val=0.12.ckpt"  "04-07-2024-11:38:46-epoch=59-loss_val=0.12.ckpt" "08-07-2024-04:21:35-epoch=122-loss_val=0.10.ckpt" 
do 
    for num in 1 3 5 10 15 20 50 90 100
    do
        python src/test_model.py \
        --tr_npy_path "data/WORD/train_CT_Abd/" \
        --val_npy_path "data/WORD/val_CT_Abd/" \
        --test_npy_path "data/WORD/test_CT_Abd/" \
        --medsam_checkpoint "weights/medsam/medsam_vit_b.pth" \
        --checkpoint $checkpoint \
        --batch_size 24 \
        --num_workers 0 \
        --num_points $num
    done
done
