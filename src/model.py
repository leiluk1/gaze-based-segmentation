import cv2
import lightning as pl
import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
from segment_anything import sam_model_registry
from torch import nn
import torch.nn.functional as F
from torchmetrics import Dice, JaccardIndex
import torchvision


class MedSAM(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "vit_b",
        medsam_checkpoint: str = None,
        freeze_image_encoder: bool = False,
        freeze_prompt_encoder: bool = False,
        lr: float = 0.00005,
        weight_decay: float = 0.01,
        num_points: int = 20,
        is_mask_diff: bool = False,
        is_mask_prompt: bool = False,
        base_medsam_checkpoint: str = None,
        eval_per_organ: bool = False,
        logger=None
    ):
        super().__init__()
        self.sam_model = sam_model_registry[backbone](checkpoint=medsam_checkpoint)

        self.freeze_prompt_encoder = freeze_prompt_encoder
        if self.freeze_prompt_encoder:
            # freeze prompt encoder
            for param in self.sam_model.prompt_encoder.parameters():
                param.requires_grad = False
            print("Prompt encoder is frozen")

        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            for param in self.sam_model.image_encoder.parameters():
                param.requires_grad = False
            print("Image encoder is frozen")

        self.lr = lr
        self.weight_decay = weight_decay

        self.jaccard = JaccardIndex(task="binary")
        self.dice_score = Dice(threshold=0)

        self.seg_loss = monai.losses.DiceLoss(
            sigmoid=True,
            squared_pred=True,
            reduction='mean'
        )
        self.ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

        self.num_points = num_points
        self.is_mask_diff = is_mask_diff
        self.is_mask_prompt = is_mask_prompt
        self.base_medsam_checkpoint = base_medsam_checkpoint

        if self.is_mask_diff and self.base_medsam_checkpoint is not None:
            # load base model
            self.base_sam = sam_model_registry[backbone](checkpoint=medsam_checkpoint)
            base_medsam_checkpoint = torch.load(self.base_medsam_checkpoint)
            self.base_sam.load_state_dict(base_medsam_checkpoint['state_dict'], strict=False)
            # freeze base model
            for param in self.base_sam.parameters():
                param.requires_grad = False

        self.eval_per_organ = eval_per_organ

        self.clearml_logger = logger

    def forward(self, image, point_prompt, mask_prompt=None):
        image_embedding = self.sam_model.image_encoder(image)  # (B, 256, 64, 64)
        # not need to convert box to 1024x1024 grid
        # bbox is already in 1024x1024
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=point_prompt,
            boxes=None,
            masks=mask_prompt,
        )
        low_res_masks, _ = self.sam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        return low_res_masks

    def base_pred(self, image, point_prompt):
        if self.base_medsam_checkpoint is not None:
            base_sam = self.base_sam
        else:
            base_sam = self.sam_model
        base_sam.eval()
        with torch.no_grad():
            image_embedding = base_sam.image_encoder(image)  # (B, 256, 64, 64)
            sparse_embeddings, dense_embeddings = base_sam.prompt_encoder(
                points=point_prompt,
                boxes=None,
                masks=None,
            )
            low_res_masks, _ = base_sam.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=base_sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )  # (B, 1, 256, 256)

        base_sam.train()

        return low_res_masks

    def _shared_step(self, batch, batch_idx,
                     phase: str, calculate_metrics: bool = True):
        image = batch["image"]
        gt2D = batch["gt2D"]  # (B, 256, 256)
        gt2D_orig = batch["gt2D_orig"]  # (B, 1024, 1024)

        if self.is_mask_diff:
            point_prompt, low_base_pred_logits, base_pred_binary = self.generate_prompt_mask_diff(image, gt2D_orig)
            if not self.is_mask_prompt:
                low_base_pred_logits = None
        else:
            point_prompt = self.generate_point_prompt(gt2D_orig, phase)

        medsam_lite_pred = self(image, point_prompt, low_base_pred_logits)
        loss = self.seg_loss(medsam_lite_pred, gt2D) + self.ce_loss(medsam_lite_pred, gt2D.float())

        logs = {
            f"loss_{phase}": loss,
            "step": self.current_epoch + 1
        }

        if calculate_metrics:
            logs.update(self._compute_metrics(medsam_lite_pred, gt2D, phase))
            if self.eval_per_organ:
                classes = batch["organ_class"]
                logs.update(self._compute_metrics_per_organ(medsam_lite_pred, gt2D, classes, phase))

        self.log_dict(logs, prog_bar=True, on_epoch=True, on_step=False)

        if phase == "test":
            if batch_idx < 10:
                pred_binary = medsam_lite_pred[0] > self.sam_model.mask_threshold
                img_name = batch["image_name"][0]
                orig_img = image[0].squeeze().detach().cpu().permute(1, 2, 0)
                pred_mask = torchvision.transforms.functional.resize(
                                pred_binary,
                                (1024, 1024),
                                interpolation=2
                )

                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(orig_img, cmap='gray')
                plt.imshow(pred_mask.squeeze().detach().cpu(), alpha=0.5, cmap='viridis')
                plt.title("Predicted mask")

                plt.subplot(1, 2, 2)
                plt.imshow(orig_img, cmap='gray')
                plt.imshow(gt2D_orig[0].squeeze().detach().cpu(), alpha=0.5, cmap='viridis')
                plt.title("Ground Truth mask")

                self.clearml_logger.report_matplotlib_figure(
                    title=f"Test Prediction: {img_name}",
                    series="Mask pred visualization",
                    iteration=batch_idx,
                    figure=plt
                )

                plt.close()

        if phase == "val" and self.is_mask_diff:
            if batch_idx % 5 == 0:
                orig_img = image[0].squeeze().detach().cpu().permute(1, 2, 0)
                img_name = batch["image_name"][0]
                img = np.load("./data/WORD/val_CT_Abd/imgs/" + img_name, 'r', allow_pickle=True)
                img = (img * 255).astype(np.uint8)

                coords = point_prompt[0][0].cpu().tolist()

                pred_binary = medsam_lite_pred[0] > self.sam_model.mask_threshold
                pred_mask = torchvision.transforms.functional.resize(
                                pred_binary,
                                (1024, 1024),
                                interpolation=2
                )

                fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                axs[0].imshow(orig_img, cmap='gray')
                axs[0].imshow(base_pred_binary[0].squeeze().detach().cpu(), alpha=0.5, cmap='viridis')
                axs[0].set_title("Base prediction mask")

                axs[1].imshow(orig_img, cmap='gray')
                axs[1].imshow(gt2D_orig[0].squeeze().detach().cpu(), alpha=0.5, cmap='viridis')
                axs[1].set_title("Ground Truth mask")

                for x, y in coords:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                axs[2].imshow(img)
                axs[2].set_title("Generated points from mask differences")

                axs[3].imshow(orig_img, cmap='gray')
                axs[3].imshow(pred_mask.squeeze().detach().cpu(), alpha=0.5, cmap='viridis')
                axs[3].set_title("Final predicted mask")

                self.clearml_logger.report_matplotlib_figure(
                    title=f"Validation Prediction: {img_name}",
                    series="Mask pred visualization",
                    iteration=batch_idx,
                    figure=plt
                )

                plt.close()

        return loss, logs

    def _compute_metrics_per_organ(self, pred_logits, gt_mask, classes, phase):
        pred_binary = pred_logits > self.sam_model.mask_threshold
        num_classes = torch.max(classes).item()
        for organ in range(1, num_classes + 1):
            dice_arr = []
            jaccard_arr = []
            organ_mask = (classes == organ)
            for i in range(pred_logits.size(0)):
                if organ_mask[i]:
                    dice = self.dice_score(pred_logits[i], gt_mask[i]).item()
                    jaccard = self.jaccard(pred_binary[i], gt_mask[i]).item()
                    dice_arr.append(dice)
                    jaccard_arr.append(jaccard)
            dice_mean = np.mean(dice_arr)
            dice_std = np.std(dice_arr)

            jaccard_mean = np.mean(jaccard_arr)
            jaccard_std = np.std(jaccard_arr)

            metrics = {
                f"iou_mean/{phase}/{organ}": jaccard_mean,
                f"iou_std/{phase}/{organ}": jaccard_std,
                f"dice_mean/{phase}/{organ}": dice_mean,
                f"dice_std/{phase}/{organ}": dice_std
            }

        return metrics

    def _compute_metrics(self, pred_logits, gt_mask, phase):
        pred_binary = pred_logits > self.sam_model.mask_threshold
        dice_arr = []
        jaccard_arr = []
        for i in range(pred_logits.size(0)):
            dice = self.dice_score(pred_logits[i], gt_mask[i]).item()
            jaccard = self.jaccard(pred_binary[i], gt_mask[i]).item()
            dice_arr.append(dice)
            jaccard_arr.append(jaccard)
        dice_mean = np.mean(dice_arr)
        dice_std = np.std(dice_arr)

        jaccard_mean = np.mean(jaccard_arr)
        jaccard_std = np.std(jaccard_arr)

        metrics = {
            f"iou_mean/{phase}": jaccard_mean,
            f"iou_std/{phase}": jaccard_std,
            f"dice_mean/{phase}": dice_mean,
            f"dice_std/{phase}": dice_std,
        }

        return metrics

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train", False)[0]

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val", True)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test", True)[1]

    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        coords_torch = batch["coords"]  # (B, 2)

        labels_torch = torch.ones(coords_torch.shape[0]).long()  # (B,)

        labels_torch = labels_torch.unsqueeze(1)  # (B, 1)

        point_prompt = (coords_torch, labels_torch)

        medsam_lite_pred = self(image, point_prompt)

        return medsam_lite_pred

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.sam_model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            min_lr=1e-6,
            patience=5,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss_val"
            }
        }

    def generate_point_prompt(self, gt2D_orig, phase):
        assert self.num_points > 0, "The number of points in the prompt cannot be less than 1"
        coords_torch = []
        for i in range(gt2D_orig.shape[0]):  # B
            gt2D = gt2D_orig[i].cpu().numpy()
            y_indices, x_indices = np.where(gt2D == 1)
            if self.num_points == 1:
                x_point = np.random.choice(x_indices)
                y_point = np.random.choice(y_indices)
                coords = np.array([x_point, y_point])[None, ...]
            else:
                # if phase == "train":
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
                coords = np.concatenate((coords_in, coords_out), axis=0)  # (N, 2)

            # chosen_indices = np.random.choice(len(x_indices), self.num_points, replace=False)
            # x_points = x_indices[chosen_indices]
            # y_points = y_indices[chosen_indices]
            # coords = np.array([x_points, y_points]).T

            coords_torch.append(torch.tensor(coords).float())

        coords_torch = torch.stack(coords_torch).cuda()  # (B, N, 2)

        # Fixed label (1)
        labels_torch = torch.ones(coords_torch.shape[0], coords_torch.shape[1]).long()  # (B, N)

        # Padding
        # num_padding = np.random.randint(0, self.num_points)
        # padding_indices = np.random.choice(coords_torch.shape[1], num_padding, replace=False)
        # coords_torch[:, padding_indices, :] = torch.tensor([0, 0], dtype=torch.float, device=coords_torch.device)
        # labels_torch[:, padding_indices] = -1

        # Assign ones as labels for coords_in and zeros for coords_out
        # num_points_in = int(coords_torch.shape[1] * 0.8)
        # num_points_out = coords_torch.shape[1] - num_points_in
        # labels_torch = torch.cat((torch.ones(coords_torch.shape[0], num_points_in),
        #                           torch.zeros(coords_torch.shape[0], num_points_out)), dim=1).long()

        # Random labels (0 or 1)
        # labels_torch = torch.randint(low=0, high=2, size=(coords_torch.shape[0], coords_torch.shape[1])).long()  # (B, N)

        return (coords_torch, labels_torch)

    def generate_prompt_mask_diff(self, image, gt2D_orig):
        coords_torch_base = []
        for i in range(gt2D_orig.shape[0]):  # B
            gt2D = gt2D_orig[i].cpu().numpy()
            y_indices, x_indices = np.where(gt2D == 1)
            chosen_indices = np.random.choice(len(x_indices), self.num_points, replace=False)
            x_points = x_indices[chosen_indices]
            y_points = y_indices[chosen_indices]
            coords_base = np.array([x_points, y_points]).T # (N, 2)
            coords_torch_base.append(torch.tensor(coords_base).float())
        coords_torch_base = torch.stack(coords_torch_base).cuda()  # (B, N, 2)
        labels_torch_base = torch.ones(coords_torch_base.shape[0], coords_torch_base.shape[1]).long()
        point_prompt = (coords_torch_base, labels_torch_base)

        low_base_pred_logits = self.base_pred(image, point_prompt)
        base_pred_logits = F.interpolate(
            low_base_pred_logits,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
            )

        base_pred_binary = (base_pred_logits > self.sam_model.mask_threshold).int()
        gt_mask = gt2D_orig.unsqueeze(1)
        delta = (base_pred_binary - (gt_mask > 0).int()).abs().squeeze(1)

        # num_points = np.random.randint(2, self.num_points)
        gt_mask_for_idx = gt_mask.squeeze(1)

        coords_list = point_prompt[0].tolist()
        labels_list = point_prompt[1].tolist()

        for num_sample in range(delta.size(0)):
            conditions = [
                (delta[num_sample] == 1, 0.7),  # from mask differences
                (gt_mask_for_idx[num_sample] == 1, 0.2),  # from gt mask
                (gt_mask_for_idx[num_sample] == 0, 0.1)  # from outside of gt mask
            ]

            for pos, ratio in conditions:
                y_idx, x_idx = torch.where(pos)
                rnd_idx = np.random.randint(0, len(y_idx), int(ratio*self.num_points))
                coords_list[num_sample].extend([[x_idx[id].item(), y_idx[id].item()] for id in rnd_idx])
                labels_list[num_sample].extend([1] * len(rnd_idx))

        coords_torch = torch.tensor(
            coords_list,
            dtype=torch.float64,
            requires_grad=True,
            device=self.device
        )
        labels_torch = torch.tensor(
            labels_list,
            dtype=torch.float64,
            requires_grad=True,
            device=self.device
        )

        return (coords_torch, labels_torch), low_base_pred_logits, base_pred_binary
