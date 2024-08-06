from matplotlib import pyplot as plt
import numpy as np
import torch


class PointPromptDemo:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def show_mask(self, mask, ax, random_color=False, alpha=0.30):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 52/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @torch.no_grad()
    def infer(self, batch):
        medsam_lite_pred, coords = self.model.predict_step(batch, 1)
        return medsam_lite_pred, coords

    def show(self, image, seg, fig_size=5, alpha=0.7):
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        plt.tight_layout()
        ax.imshow(image)
        ax.axis('off')
        self.show_mask(seg, ax, random_color=False, alpha=alpha)
        plt.show()
