import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# importing the custom model
from model import CustomEmotionCNN


# =========================
#  HARDWARE CHECK
# =========================
def get_device():
    if torch.cuda.is_available():
        print(f"\n🚀 GPU Activated: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("🍏 Apple Silicon GPU (MPS) erkannt.")
        return torch.device("mps")
    else:
        print("\n⚠️  No GPU detected. Training will be performed on CPU, which may be slow.")
        return torch.device("cpu")


DEVICE = get_device()


# =========================
# CONFIG
# =========================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 25

TRAIN_DIR = 'data/RAF_original_processed/train'
VAL_DIR   = 'data/RAF_original_processed/test'
MODEL_DIR = 'models'


# =========================
# MODEL SAVE PATH
# =========================
def get_unique_model_path(base_name="raf_cnn"):
    """
    Find the next available model save path: raf_cnn_v0.pth, raf_cnn_v1.pth, etc.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    counter = 0
    while True:
        filename = f"{base_name}_v{counter}.pth"
        full_path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(full_path):
            return full_path, counter

        counter += 1


# =========================
# VALIDATION
# =========================
def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"    >>> Validation Loss: {val_loss / len(loader):.4f} | Val Acc: {acc:.2f}%")
    return acc


# =========================
# MAIN
# =========================
def main():
    # 1. model save path
    save_path, version_id = get_unique_model_path()
    print("=" * 50)
    print(f"💾 This training will be saved as: {save_path}")
    print("=" * 50)

    # 2. Data Transformations
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    # 3. Datasets & Dataloaders
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    num_classes = len(train_dataset.classes)
    print(f"Klassen ({num_classes}): {train_dataset.classes}")

    # 4. Model, Loss, Optimizer
    model = CustomEmotionCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {avg_loss:.4f} | Train Acc: {epoch_acc:.2f}%"
        )

        # Validation
        val_acc = validate(model, val_loader, criterion)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    🌟 New Record! Model saved to {save_path}")

    print("=" * 50)
    print("Training completed.")
    print(f"The best model achieved {best_val_acc:.2f}% Accuracy on the test data.")
    print(f"Saved as: {save_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
