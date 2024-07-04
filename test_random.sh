#!/bin/bash

for checkpoint in "exp_name=0-epoch=42-val_loss=0.00.ckpt" "exp_name=0-epoch=32-val_loss=0.00.ckpt"  "exp_name=0-epoch=49-val_loss=0.00.ckpt"  "exp_name=0-epoch=40-val_loss=0.00.ckpt"  "exp_name=0-epoch=57-val_loss=0.00.ckpt" "exp_name=0-epoch=89-val_loss=0.00.ckpt" 
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
