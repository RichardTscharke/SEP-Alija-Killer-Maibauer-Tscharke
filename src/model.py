import torch
import torch.nn as nn

class CustomEmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomEmotionCNN, self).__init__()
        
        # --- FEATURE EXTRACTOR (Die "Augen" des Netzes) ---
        # Input: 3 Kanäle (RGB), 64x64 Pixel
        
        # Block 1: 3 -> 32 Kanäle
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Bild wird 32x32
        )
        
        # Block 2: 32 -> 64 Kanäle
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Bild wird 16x16
        )
        
        # Block 3: 64 -> 128 Kanäle
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Bild wird 8x8
        )
        
        # Block 4: 128 -> 256 Kanäle
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Bild wird 4x4
        )
        
        # --- CLASSIFIER (Das "Gehirn") ---
        # Am Ende von Block 4 haben wir 256 Feature-Maps der Größe 4x4.
        # Flatten size = 256 * 4 * 4 = 4096
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(4096, 512), # Erster Fully Connected Layer
            nn.ReLU(),
            nn.Dropout(0.5),      # Gegen Overfitting (wie im Report geplant)
            nn.Linear(512, num_classes) # Output Layer (6 Neuronen für 6 Emotionen)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x