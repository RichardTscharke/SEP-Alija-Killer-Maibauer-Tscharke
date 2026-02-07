import torch
import torch.nn as nn
from utils import ResidualBlock, initialize_weights

class ResNetLightCNN2(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetLightCNN2, self).__init__()

        # Initial Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Residual Stages
        self.stage1 = ResidualBlock(32, 64, stride=1, use_se = False)  # SE not in Stage1
        self.stage2 = ResidualBlock(64, 128, stride=2, use_se = True)
        self.stage3 = ResidualBlock(128, 256, stride=2, use_se = True)

        # Adaptive Pooling auf 4x4 statt 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)

        # Fully Connected wie RafCustomCNN
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
