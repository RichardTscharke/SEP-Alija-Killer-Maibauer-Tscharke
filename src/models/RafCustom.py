import torch
import torch.nn as nn


class RafCustomCNN(nn.Module):
    '''
    Simple CNN baseline architecture.
    Input:  (Batch, 3, 64, 64)
    Output: (Batch, num_classes) logits
    '''
    def __init__(self, num_classes=6):
        super().__init__()

        # Standarized convolution blocks
        def conv_block(in_c, out_c):
            return nn.Sequential(
                # 3x3 Convolution => Size remains the same
                nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1),
                # Channel-wise normalization for stabilized training
                nn.BatchNorm2d(out_c),
                # Nonlinearity
                nn.ReLU(),
                # Halfen height and width due to kernel size = 2 = stride size
                nn.MaxPool2d(2),
            )
        
        # Feature extractor: stepwise reduces the spatial resolution
        self.features = nn.Sequential(
            conv_block(3, 32),     # 64 -> 32
            conv_block(32, 64),    # 32 -> 16
            conv_block(64, 128),   # 16 -> 8
            conv_block(128, 256),  # 8 -> 4
        )

        # Classification head: feature maps -> logits
        #                      (B, 256*4*4) -> (B, 6)
        self.classifier = nn.Sequential(
            # Fully connected: feature maps -> neurons
            nn.Linear(256*4*4, 512),
            # Nonlinearity
            nn.ReLU(),
            # Prevent overfitting by randomly switching off neurons
            nn.Dropout(0.5),
            # Final logits per class: neurons -> logits
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Extract visual features 
        x = self.features(x)
        # (B, 256, 4, 4) -> (Batch, 256*4*4)
        x = torch.flatten(x, 1)
        # Return logits
        return self.classifier(x)