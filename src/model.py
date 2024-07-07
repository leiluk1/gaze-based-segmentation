import lightning as pl
import monai
import numpy as np
import torch
from segment_anything import sam_model_registry
from torch import nn
from torchmetrics import Dice, JaccardIndex


class MedSAM(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "vit_b",
        medsam_checkpoint: str = None,
        freeze_image_encoder: bool = False,
        freeze_prompt_encoder: bool = False,
        lr: float = 0.00005,
        weight_decay: float = 0.01,
        num_points: int = 20
    ):
        super().__init__()
        self.sam_model = sam_model_registry[backbone](checkpoint=medsam_checkpoint)
        self.image_encoder = self.sam_model.image_encoder
        self.mask_decoder = self.sam_model.mask_decoder
        self.prompt_encoder = self.sam_model.prompt_encoder

        self.freeze_prompt_encoder = freeze_prompt_encoder
        if self.freeze_prompt_encoder:
            # freeze prompt encoder
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            print("Prompt encoder is frozen")

        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
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

    def forward(self, image, point_prompt):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # not need to convert box to 1024x1024 grid
        # bbox is already in 1024x1024
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_prompt,
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        return low_res_masks

    def _shared_step(self, batch, batch_idx, 
                     phase: str, calculate_metrics: bool = True):
        image = batch["image"]
        gt2D = batch["gt2D"]  # (B, 256, 256)
        gt2D_orig = batch["gt2D_orig"]  # (B, 1024, 1024)

        point_prompt = self.generate_point_prompt(gt2D_orig, phase)

        medsam_lite_pred = self(image, point_prompt)
        loss = self.seg_loss(medsam_lite_pred, gt2D) + self.ce_loss(medsam_lite_pred, gt2D.float())

        logs = {
            f"loss_{phase}": loss,
            "step": self.current_epoch + 1
        }

        if calculate_metrics:
            logs.update(self._compute_metrics(medsam_lite_pred, gt2D, phase))

        self.log_dict(logs, prog_bar=True, on_epoch=True, on_step=False)

        return loss, logs

    def _compute_metrics(self, pred_logits, gt_mask, phase):
        pred_binary = pred_logits > self.sam_model.mask_threshold
        jaccard = self.jaccard(pred_binary, gt_mask)
        # dice = self.dice_score(pred_logits, gt_mask)
        dice_arr = []
        for i in range(pred_logits.size(0)):
            dice = self.dice_score(pred_logits[i], gt_mask[i]).item()
            dice_arr.append(dice)
        dice = np.mean(dice_arr)

        metrics = {
            f"iou/{phase}": jaccard,
            f"dice/{phase}": dice,
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
