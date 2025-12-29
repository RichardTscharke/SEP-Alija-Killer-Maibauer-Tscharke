import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ResNetLight
import os

# --- KONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pfad zu deinen Daten (relativ zum Projektordner)
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'

def get_data_loaders():
    # Transformierungen: Bild laden -> Tensor -> Normalisieren
    # Tipp: Data Augmentation (Spiegeln, Drehen) hilft gegen Overfitting!
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Zufälliges Spiegeln (nur Training)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Werte auf -1 bis 1 bringen
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Wir nutzen ImageFolder - das liest die Ordnerstruktur automatisch!
    # Achtung: Wenn die Ordner leer sind, stürzt das hier ab.
    try:
        train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
        val_data = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
        
        # Zeige Klassen-Mapping an (z.B. 0=Anger, 1=Happy...)
        print(f"Klassen gefunden: {train_data.class_to_idx}")
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader
    except Exception as e:
        print("FEHLER beim Laden der Daten. Hast du Ordner und Bilder in 'data/train'?")
        print(e)
        return None, None

def train():
    print(f"Starte Training auf: {DEVICE}")
    
    train_loader, val_loader = get_data_loaders()
    if train_loader is None: return # Abbruch wenn keine Daten

    model = ResNetLight(num_classes=6).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDIERUNG (Wie gut sind wir wirklich?) ---
        model.eval() # Wichtig: Batchnorm/Dropout ausschalten
        correct = 0
        total = 0
        with torch.no_grad(): # Speicher sparen, keine Gradienten nötig
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_train_loss:.4f} - Val Acc: {accuracy:.2f}%")

        # Speichere nur, wenn wir besser geworden sind
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"   -> Neues bestes Modell gespeichert! ({accuracy:.2f}%)")

if __name__ == "__main__":
    train()