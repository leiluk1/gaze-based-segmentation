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
        lr: float = 0.00005,
        weight_decay: float = 0.01
    ):
        super().__init__()
        self.sam_model = sam_model_registry[backbone](checkpoint=medsam_checkpoint)
        self.image_encoder = self.sam_model.image_encoder
        self.mask_decoder = self.sam_model.mask_decoder
        self.prompt_encoder = self.sam_model.prompt_encoder

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

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
        gt2D = batch["gt2D"]
        coords_torch = batch["coords"]  # (B, N, 2)

        labels_torch = torch.ones(coords_torch.shape[0], coords_torch.shape[1]).long()  # (B, N)
        # for random labels (0 or 1)
        # labels_torch = torch.randint(low=0, high=2, size=(coords_torch.shape[0], coords_torch.shape[1])).long()  # (B, N)

        point_prompt = (coords_torch, labels_torch)

        medsam_lite_pred = self(image, point_prompt)
        loss = self.seg_loss(medsam_lite_pred, gt2D) + self.ce_loss(medsam_lite_pred, gt2D.float())

        logs = {
            f"loss/{phase}": loss,
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
        # for random labels (0 or 1)
        # labels_torch = torch.randint(low=0, high=2, size=(coords_torch.shape[0], 1)).long()  # (B, 1)
        
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
                "monitor": "loss/val"
            }
        }
