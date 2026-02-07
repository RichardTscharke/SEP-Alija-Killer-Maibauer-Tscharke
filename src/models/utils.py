import torch.nn as nn

'''
Utility components shared by both ResNetLight models.
Includes:
- Squeeze and Excitation Block (SEBlock)
- ResidualBlock
- Weight initialization function
'''

class SEBlock(nn.Module):
    '''
    Learn a channel-wise weighting of the feature maps.
    Emphasize informative channels and diminish less informative ones.
    - Squeeze: Reduce every feature map by Global Average Pooling
    - Excitation: Two Fully-Connected layers learn a non-linear reweighting of the channels.
    - Scaling: Original feature maps are scaled by learned channel-wise attention weights.
    '''
    def __init__(self, channel, reduction=16):  
        super(SEBlock, self).__init__()

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully-Connected layer for channel weighting
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),

            # Nonlinearity
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),

            # Outputs within [0, 1] as channel weights
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze: (B, C, H, W) -> (B, C)
        y = self.avg_pool(x).view(b, c)

        # Excitation
        y = self.fc(y).view(b, c, 1, 1)

        # Re-Weight the feature maps
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    '''
    Designed to enable the training of deep CNNs by mitigating vanishing gradients.
    Structure:
    - Two 3x3 convolutions with BatchNorm and LeakyReLU
    - Shortcuts with identity or 1x1-Convolution mapping

    Optional SE-Block in order to optimize the channels' representations.
    '''
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()

        # 1. Convolution Block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 2. Convolution Block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Optional SE-Block
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Shortcut connections
        # Identity mapping if input and output dimensions match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:

            # Adapt the resolution or channel-amount
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Optional channel-wise reweighting
        out = self.se(out) 

        # Residual Additon
        out += self.shortcut(x)

        # Final Activation
        out = self.relu(out)
        return out
    
def initialize_weights(model):
    '''
    Initializes the weights of a model.
    Adapted for ResNet-like architectures with LeakyReLU-Activations.
    Structure:
    - Conv2D: Kaiming Normal (He init.) for LeakyReLU
    - BatchNorm2D: Weight = 1, Bias = 0 
    - Linear: Normal distribution with low variance.
    '''
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming  normal (He Init) for Conv Layer
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm "neutral"
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)