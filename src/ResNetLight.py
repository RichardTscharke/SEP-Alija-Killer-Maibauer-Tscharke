import torch.nn as nn


class ResNetLightCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetLightCNN, self).__init__()

        # 1. Initial Convolution
        # Input: (64, 64, 3) -> Output: (64, 64, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 2. Residual Stages (The "Eyes")
        # Three residual stages... 32 -> 64 -> 128 filters

        # Stage 1: 32 Channels (Input  32, Output  32)
        self.stage1 = ResidualBlock(32, 32, stride=1)

        # Stage 2: 64 Channels (Downsampling 64x64 -> 32x32)
        self.stage2 = ResidualBlock(32, 64, stride=2)

        # Stage 3: 128 Channels (Downsampling 32x32 -> 16x16)
        self.stage3 = ResidualBlock(64, 128, stride=2)

        # 3. Classifier (The "Brain")
        # Exclusion of dense layers | Global Average Pooling (Batch, 128, 16, 16) -> (Batch, 128, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 128 Input Features -> 6 Emotionen
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        # Residual Stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Classification
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main Path
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut Path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out