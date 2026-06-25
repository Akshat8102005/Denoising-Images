import torch

def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr.item()