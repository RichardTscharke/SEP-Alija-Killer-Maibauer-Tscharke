import torch
import torch.nn as nn
from models.utils import ResidualBlock, initialize_weights

class ResNetLightCNN2(nn.Module):
    '''
    Variant of ResNetLight with fully-connected classification head.
    Differences to Version 1:

    -  No SE-Block in the first residual stage
    -> Preserve early low-level features

    -  Adaptive Average Pooling to 4x4 instead of 1x1
    -> Preserve spatial details

    => Goal: Capture fine-grained facial details important for FER.
    '''
    def __init__(self, num_classes=6):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Residual Stages
        # Stage 1 has no SE block to avoid early channel reweighting
        self.stage1 = ResidualBlock(32, 64, stride=1, use_se = False)
        self.stage2 = ResidualBlock(64, 128, stride=2, use_se = True)
        self.stage3 = ResidualBlock(128, 256, stride=2, use_se = True)

        # Adaptive Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)

        # Fully Connected head inspired by our RafCustomCNN
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Explicit weight initialization
        initialize_weights(self)

    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)

        # Residual Stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
