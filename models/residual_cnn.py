import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.block(x)

        return self.relu(residual + x)
    
    
class ResidualCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_conv = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)

        )

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)

        self.output_conv = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        x = self.input_conv(x)

        x = self.res1(x)

        x = self.res2(x)

        x = self.output_conv(x)

        return x