import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class DenoiseDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.image_names = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        noisy = np.load(os.path.join(self.noisy_dir, name))
        clean = np.load(os.path.join(self.gt_dir, name))

        noisy = torch.tensor(noisy).float().unsqueeze(0)
        clean = torch.tensor(clean).float().unsqueeze(0)

        # Upsample noisy to match GT size (512x512)
        noisy = F.interpolate(
            noisy.unsqueeze(0),
            size=clean.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return noisy, clean