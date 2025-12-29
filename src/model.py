import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetLight(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetLight, self).__init__()
        
        # 1. Start: Input (Batch, 3, 64, 64) -> Output (Batch, 32, 64, 64)
        # Wir nutzen Padding=1, damit die Bildgröße gleich bleibt (64x64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        # 2. Residual Blocks (Hier kannst du später mehr hinzufügen)
        # Beispiel: Ein einfacher Block, der die Kanäle verdoppelt (32 -> 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # MaxPool verkleinert das Bild schrittweise: 64 -> 32 -> 16
        self.pool = nn.MaxPool2d(2, 2) 

        # 3. Global Average Pooling (WICHTIG für Explainable AI & Parameter-Reduktion)
        # Wandelt (Batch, 128, H, W) um in (Batch, 128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Klassifikator: 128 Features -> 6 Emotionen
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x ist am Anfang: (Batch, 3, 64, 64)
        
        # Erster Layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Block 1 + Downsampling (64x64 -> 32x32)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x) 
        
        # Block 2 + Downsampling (32x32 -> 16x16)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Global Average Pooling (macht alles platt zu Vektoren)
        x = self.gap(x) 
        x = x.view(x.size(0), -1) # Flatten: (Batch, 128)
        
        # Klassifikation
        x = self.fc(x)
        
        return x

# Kleiner Test, wenn man die Datei direkt ausführt
if __name__ == "__main__":
    # Simuliere ein Bild (Batch=1, Channels=3, Height=64, Width=64)
    dummy_input = torch.randn(1, 3, 64, 64)
    model = ResNetLight(num_classes=6)
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Erwartet: [1, 6])")
    print("Modell erfolgreich initialisiert!")