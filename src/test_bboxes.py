import os
import argparse
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry
import torch
import torchvision
from tqdm import tqdm

from dataset import NpyDataModule


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
        '--test_npy_path',
        type=str,
        help="Path to the test data root directory.",
        required=True
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help="Batch size."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
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
        '--num_classes',
        type=int,
        default=16,
        help="Number of classes in the dataset."
        )
    return parser


def load_model(medsam_checkpoint="weights/medsam/medsam_vit_b.pth", device="cuda"):
    medsam_model = sam_model_registry['vit_b'](checkpoint=medsam_checkpoint)
    return medsam_model.to(device)


def dice(pred, true, k=1):
    intersection = np.sum(pred[true == k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


def compute_metrics_per_organ(pred_logits, gt_mask, classes):
    pred_binary = pred_logits > 0.0
    num_classes = torch.max(classes).item()
    metrics = {}
    for organ in range(1, num_classes + 1):
        dice_arr = []
        for i in range(pred_logits.size(0)):
            organ_mask = (classes == organ)
            if organ_mask[i].item():
                dice_score = dice(
                    np.uint8(pred_binary[i].cpu().numpy()),
                    gt_mask[i].cpu().numpy())
                dice_arr.append(dice_score)
        if dice_arr:
            dice_mean = np.mean(dice_arr)
            dice_std = np.std(dice_arr)
            metrics[f"dice_mean/{organ}"] = dice_mean
            metrics[f"dice_std/{organ}"] = dice_std
    return metrics


def generate_bboxes(batch):
    batch_gts = batch["gt2D_orig"]
    batch_bboxes = []
    for i in range(batch_gts.shape[0]):
        gt_segm = batch_gts[i].squeeze().detach().cpu().numpy()
        gt_segm = cv2.normalize(gt_segm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thresh = cv2.threshold(gt_segm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        bboxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append((x, y, w + x, h + y))
        batch_bboxes.append(bboxes)
    return batch_bboxes


def infer_bboxes(batch, medsam_model, device):
    with torch.no_grad():
        img_embed = medsam_model.image_encoder(batch["image"].to(device))
        batch_bboxes = generate_bboxes(batch)
        box_torch = torch.as_tensor(
            batch_bboxes,
            dtype=torch.float,
            device=img_embed.device
        )

        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        pred_binary = low_res_logits > 0.0
        pred_mask = torchvision.transforms.functional.resize(
            pred_binary,
            (1024, 1024),
            interpolation=2
        )
        medsam_seg = pred_mask.squeeze()
    return medsam_seg


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))


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

    medsam_checkpoint = "weights/medsam/medsam_vit_b.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    medsam_model = load_model(medsam_checkpoint, device)
    medsam_model.eval()
    print(f"MedSAM size: {sum(p.numel() for p in medsam_model.parameters())}")

    datamodule = NpyDataModule(
        train_data_path=args.tr_npy_path,
        val_data_path=args.val_npy_path,
        test_npy_path=args.test_npy_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gt_in_ram=args.gt_in_ram,
    )
    datamodule.setup()
    num_classes = args.num_classes

    metrics = {}

    for batch in tqdm(datamodule.test_dataloader()):
        medsam_seg = infer_bboxes(batch, medsam_model, device)
        metrics.update(
            compute_metrics_per_organ(
                medsam_seg,
                batch["gt2D_orig"],
                batch["organ_class"]
            )
        )

    mean_dice = np.mean([metrics[f"dice_mean/{organ}"] for organ in range(1, num_classes + 1)])
    std_dice = np.std([metrics[f"dice_mean/{organ}"] for organ in range(1, num_classes + 1)])

    print(f"Total mean dice: {mean_dice:.4f}")
    print(f"Total std of dice: {std_dice:.4f}")

    for organ in range(1, num_classes + 1):
        print(f"mean_dice/{organ}: {metrics.get(f'dice_mean/{organ}'):.4f}")

    for organ in range(1, num_classes + 1):
        print(f"mean_std/{organ}: {metrics.get(f'dice_std/{organ}'):.4f}")

    print("\nMetrics summary:")
    print(metrics)


if __name__ == "__main__":
    main()
