from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(pred, target):

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    value = ssim(
        pred,
        target,
        data_range=target.max() - target.min()
    )

    return value