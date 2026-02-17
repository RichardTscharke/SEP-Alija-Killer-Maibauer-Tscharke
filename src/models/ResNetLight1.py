import torch.nn as nn
from .model_utils import ResidualBlock, initialize_weights

class ResNetLightCNN1(nn.Module):
    '''
    Lightweight ResNet-style CNN with SE attention.
    Input:  (Batch, 3, 64, 64)
    Output: (Batch, num_classes) logits
    '''
    def __init__(self, num_classes=6):
        super().__init__()

        # 1. Initial Convolution
        # (B, 3, 64, 64) -> (B, 32, 64, 64)
        self.conv1 = nn.Sequential(

            # 3x3 convolution, spatial size preserved
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),

            # Channel-wise normalization
            nn.BatchNorm2d(32),

            # LeakyReLU allows small gradients for negative values
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # 2. Residual Stages: Increase channels and reduce resolution (check utils.py for details)
        # Three residual stages: 32 -> 64 -> 128 -> 256 filters

        # Stage 1: (B, 32, 64, 64) -> (B, 64, 64, 64)
        # NO downsampling!
        self.stage1 = ResidualBlock(32, 64, stride=1, use_se = True)

        # Stage 2: (B, 64, 64, 64) -> (B, 128, 32, 32)
        self.stage2 = ResidualBlock(64, 128, stride=2, use_se = True)

        # Stage 3: (B, 128, 32, 32) -> (B, 256, 16, 16)
        self.stage3 = ResidualBlock(128, 256, stride=2, use_se = True)

        # 3. Classification head
        # Exclusion of dense layers | Global Average Pooling
        # (B, 256, 16, 16) -> (B, 256, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # (B, 256, 1, 1) -> (B, 256)
        self.flatten = nn.Flatten()

        # Prevent overfitting by randomly switching off neurons
        self.dropout = nn.Dropout(p=0.3)

        # Final linear layer: feature vector -> logits
        self.fc = nn.Linear(256, num_classes)

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