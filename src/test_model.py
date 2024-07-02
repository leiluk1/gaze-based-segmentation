import argparse
import os
import random
from datetime import datetime

import lightning as pl
import numpy as np
import torch
from clearml import Task

from dataset import NpyDataModule
from model import MedSAM


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tr_npy_path',
        type=str,
        help="Path to the train data root directory.",
        required=True
    )
    parser.add_argument(
        '--val_npy_path',
        type=str,
        help="Path to the validation data root directory.",
        required=True
    )
    parser.add_argument(
        '--medsam_checkpoint',
        type=str,
        help="Path to the MedSAM checkpoint.",
        required=True
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help="MedSAM fine-tuned checkpoint file name.",
        required=True
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Batch size."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help="Number of data loader workers."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2023,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        '--disable_aug',
        action='store_true',
        help="Disable data augmentation."
    )
    parser.add_argument(
        '--gt_in_ram',
        default=True, 
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--num_points',
        type=int,
        default=1,
        help="Number of points in prompt to test on."
    )

    return parser


def test(exp_name, args):
    Task.init(
            project_name="medsam_point",
            tags=[
                "testing",
                "1_point",
            ],
            task_name=exp_name,
    )

    medsam_model = MedSAM(
        medsam_checkpoint=args.medsam_checkpoint,
        freeze_image_encoder=True,
        num_points=args.num_points
    )
    checkpoint = torch.load("logs/" + args.checkpoint)
    medsam_model.load_state_dict(checkpoint['state_dict'], strict=False)

    datamodule = NpyDataModule(
        args.tr_npy_path,
        args.val_npy_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_aug=not args.disable_aug,
        gt_in_ram=args.gt_in_ram,
    )
    datamodule.setup()

    trainer = pl.Trainer()

    test_dice = trainer.test(
        medsam_model,
        datamodule.test_dataloader()
    )[0]["dice/test"]

    return test_dice


def main():
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    exp_name = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    test_dice = test(exp_name, args)
    print(test_dice)


if __name__ == "__main__":
    main()