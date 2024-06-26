import argparse
import os
import random
from datetime import datetime

import lightning as pl
import numpy as np
import torch
from clearml import Task
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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
        '--max_epochs',
        type=int,
        default=1000,
        help="Maximum number of epochs."
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
        '--lr',
        type=float,
        default=0.00005,
        help="learning rate (absolute lr)"
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help="Weight decay."
    )
    parser.add_argument(
        '--accumulate_grad_batches',
        type=int,
        default=4,
        help="Accumulate grad batches."
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
        '--num_points',
        type=int,
        default=1,
        help="Number of points in prompt."
    )

    return parser


def train(exp_name, args):
    Task.init(
            project_name="medsam_point",
            tags=[
                "fine_tuning",
            ],
            task_name=exp_name,
    )

    medsam_model = MedSAM(
        medsam_checkpoint=args.medsam_checkpoint,
        freeze_image_encoder=True,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # checkpoint = torch.load(medsam_ckpt_path)
    # medsam_model.load_state_dict(checkpoint, strict=False)
    print(f"MedSAM size: {sum(p.numel() for p in medsam_model.parameters())}")

    datamodule = NpyDataModule(
            args.tr_npy_path,
            args.val_npy_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_points=args.num_points,
            data_aug=not args.disable_aug
    )
    datamodule.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath="logs/",
        filename="{exp_name}-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="loss/val",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="loss/val",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="gpu",
        devices=1
    )
    trainer.fit(
        medsam_model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )

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
    test_dice = train(exp_name, args)
    print(test_dice)


if __name__ == "__main__":
    main()