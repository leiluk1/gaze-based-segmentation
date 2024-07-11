import torch
from torch.nn import functional as F
import cv2
import numpy as np
from matplotlib import pyplot as plt


class PointPromptDemo:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None

    def show_mask(self, mask, ax, random_color=False, alpha=0.30):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 52/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @torch.no_grad()
    def infer(self, gt, num_points=20, prop_out=0.8):

        point_prompt = self.generate_point_prompt(gt, num_points, prop_out)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = point_prompt,
            boxes = None,
            masks = None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.image_embeddings, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_probs = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_probs,
            size = self.img_size,
            mode = 'bilinear',
            align_corners = False
        )
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()

        seg = np.uint8(low_res_pred > 0.5)

        return seg
    
    def generate_point_prompt(self, gt, num_points=20, prop_out=0.8):
        y_indices, x_indices = np.where(gt > 0)
        y_indices_out, x_indices_out = np.where(gt == 0)

        num_points_in = int(num_points * prop_out)
        num_points_out = num_points - num_points_in

        chosen_indices_in = np.random.choice(len(x_indices), num_points_in, replace=False)
        chosen_indices_out = np.random.choice(len(x_indices_out), num_points_out, replace=False)
        x_points_in = x_indices[chosen_indices_in]
        y_points_in = y_indices[chosen_indices_in]

        x_points_out = x_indices_out[chosen_indices_out]
        y_points_out = y_indices_out[chosen_indices_out]
 
        coords_in = np.array([x_points_in, y_points_in]).T
        coords_out = np.array([x_points_out, y_points_out]).T
        coords = np.concatenate((coords_in, coords_out), axis=0)

        coords_torch = torch.tensor([coords], dtype=torch.float32).to(self.model.device)
        labels_torch = torch.ones(coords_torch.shape[0], coords_torch.shape[1]).long().to(self.model.device)

        # print(coords_torch.shape, labels_torch.shape)

        return (coords_torch, labels_torch)

    def show(self, seg, fig_size=5, alpha=0.95, scatter_size=10):

        assert self.image is not None, "Please set image first."
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))

        plt.tight_layout()
        ax.imshow(self.image)
        ax.axis('off')
        self.show_mask(seg, ax, random_color=False, alpha=alpha)
        plt.show()

    def set_image(self, image):
        self.img_size = image.shape[:2]
        if len(image.shape) == 2:
            image = np.repeat(image[:,:,None], 3, -1)
        self.image = image
        image_preprocess = self.preprocess_image(self.image)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)
    
    def preprocess_image(self, image):
        img_resize = cv2.resize(
            image,
            (1024, 1024),
            interpolation=cv2.INTER_CUBIC
        )
        # Resizing
        # normalize to [0, 1], (H, W, 3)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) 
        # convert the shape to (3, H, W)
        assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)

        return img_tensor
