#!/bin/bash

# Define the range of values for num_points
for num in 1 3 5 10 15 20
do
    python src/train_point_prompt.py --tr_npy_path "./data/WORD/train_CT_Abd/" --val_npy_path "./data/WORD/val_CT_Abd/" --medsam_checkpoint "./weights/medsam/medsam_vit_b.pth" --max_epochs 200 --batch_size 24 --num_workers 0 --num_points $num
done
