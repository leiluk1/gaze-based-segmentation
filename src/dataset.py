import os
import glob
import random
import lightning as pl
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2


# Dataset class
class NpyDataset(Dataset):
    def __init__(self, data_root, image_size=1024, num_points=1, data_aug=True, phase="train"):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
        self.image_size = image_size
        self.num_points = num_points
        self.data_aug = data_aug
        self.phase = phase
        # get data in format [[img_path1, mask1], ... [img_pathn, maskn]]
        self.data = self.read_data()

    def __len__(self):
        return len(self.gt_path_files)

    def read_data(self):
        data = []
        for gt_path in self.gt_path_files:
            img_name = os.path.basename(gt_path)
            img_path = join(self.img_path, img_name)
            gt = np.load(gt_path, 'r', allow_pickle=True)  # multiple labels [0,1,4,5...], (256,256)
            label_ids = np.unique(gt)[1:]  # [1,4,5...]
            for label_id in label_ids:
                gt2D = np.uint8(gt == label_id) 
                gt2D = (gt2D * 255).astype(np.uint8)
                thresh = cv2.threshold(
                    gt2D, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]
                cnts = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                if len(cnts) == 1:  # One segment
                    gt_segm = np.uint8(gt2D == 255)
                    data.append([img_path, gt_segm])
                if len(cnts) > 1:
                    for i in range(len(cnts)):
                        mask = np.zeros_like(gt2D)
                        cv2.drawContours(
                            mask, cnts, i, (255, 255, 255), thickness=cv2.FILLED
                        )
                        gt_segm = np.uint8(mask == 255)
                        data.append([img_path, gt_segm])
        return data

    def __getitem__(self, index):
        img_path, gt2D = self.data[index]
        img_name = os.path.basename(img_path)
        img_1024 = np.load(img_path, 'r', allow_pickle=True)  # (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 256, 256)
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0, 'image should be normalized to [0, 1]'

        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
        gt2D = np.uint8(gt2D > 0)
        assert self.num_points > 0, "The number of points in the prompt cannot be less than 1"
        assert self.phase in ["train", "test"], "Uncorrect phase name"
        y_indices, x_indices = np.where(gt2D == 1)
        if self.num_points == 1:
            x_point = np.random.choice(x_indices)
            y_point = np.random.choice(y_indices)
            coords = np.array([x_point, y_point])[None, ...]
        else:
            # if self.phase == "train":
            #     chosen_indices = np.random.choice(len(x_indices), self.num_points, replace=False)
            #     x_points = x_indices[chosen_indices]
            #     y_points = y_indices[chosen_indices]
            #     coords = np.array([x_points, y_points]).T
            # else: 
            
            y_indices_out, x_indices_out = np.where(gt2D == 0)

            num_points_in = int(self.num_points * 0.8)
            num_points_out = self.num_points - num_points_in

            chosen_indices_in = np.random.choice(len(x_indices), num_points_in, replace=False)
            chosen_indices_out = np.random.choice(len(x_indices_out), num_points_out, replace=False)
            x_points_in = x_indices[chosen_indices_in]
            y_points_in = y_indices[chosen_indices_in]

            x_points_out = x_indices_out[chosen_indices_out]
            y_points_out = y_indices_out[chosen_indices_out]

            coords_in = np.array([x_points_in, y_points_in]).T
            coords_out = np.array([x_points_out, y_points_out]).T
            coords = np.concatenate((coords_in, coords_out), axis=0)
            
            # chosen_indices = np.random.choice(len(x_indices), self.num_points, replace=False)
            # x_points = x_indices[chosen_indices]
            # y_points = y_indices[chosen_indices]
            # coords = np.array([x_points, y_points]).T

        # Randomly sample a point from the gt at scale 1024
        gt2D_256 = cv2.resize(
            gt2D,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        return {
            "image": torch.tensor(img_1024).float(),
            "gt2D": torch.tensor(gt2D_256[None, :, :]).long(),
            "coords": torch.tensor(coords).float(),
            "image_name": img_name
        }


class NpyDataModule(pl.LightningDataModule):
    def __init__(
        self, train_data_path, val_data_path, batch_size=8, test_size=0.1, num_workers=0, num_points=1, data_aug=True
    ):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.num_points = num_points
        self.data_aug = data_aug

    def setup(self):
        self.train_dataset = NpyDataset(data_root=self.train_data_path, num_points=self.num_points, data_aug=self.data_aug, phase="train")
        self.valntest_dataset = NpyDataset(data_root=self.val_data_path, num_points=self.num_points, data_aug=False, phase="test")
        self.val_dataset, self.test_dataset = random_split(self.valntest_dataset, [1 - self.test_size, self.test_size])

        print("train size:", len(self.train_dataset))
        print("val size:", len(self.val_dataset))
        print("test size:", len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
