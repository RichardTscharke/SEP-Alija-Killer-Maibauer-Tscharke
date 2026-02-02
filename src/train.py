import csv
import torch
import shutil
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# importing the custom model
from models.ResNetLight import ResNetLightCNN
from models.ResNetLight2 import ResNetLightCNN2

# OUTPU_DIR for evaluation
OUTPUT_DIR = Path("src/evaluation/outputs")

def prepare_output_dir_evaluation():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#  HARDWARE CHECK
def get_device():
    if torch.cuda.is_available():
        print(f"\n🚀 GPU Activated: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("🍏 Apple Silicon GPU (MPS) erkannt.")
        return torch.device("mps")
    else:
        print(
            "\n⚠️  No GPU detected. Training will be performed on CPU, which may be slow."
        )
        return torch.device("cpu")


DEVICE = get_device()


# CONFIGURATIONS
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 28

# Paths to data directories
TRAIN_DIR = "data/train"
VAL_DIR   = "data/validate"
MODEL_DIR = "models"


# generate unique model save path
def get_unique_model_path(base_name="ResNetLight"):

    # Find the next available model save path: ResNetLight_v0.pth, ResNetLight_v1.pth, etc.
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    counter = 0
    while True:
        filename = f"{base_name}_v{counter}.pth"
        full_path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(
            full_path
        ):  # File does not exist, so we can use this name
            return full_path, counter

        counter += 1


# VALIDATION
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
    print(
        f"    >>> Validation Loss: {val_loss / len(loader):.4f} | Val Acc: {acc:.2f}%"
    )
    return acc, val_loss / len(loader)


# MAIN
def main():

    # 0. prepare evaluation outputs directory
    prepare_output_dir_evaluation()

    # 1. model save path
    save_path, version_id = get_unique_model_path()
    print("=" * 50)
    print(f"💾 This training will be saved as: {save_path}")
    print("=" * 50)

    # 2. Data Transformations & Augmentations
    train_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # 3. Datasets & Dataloaders
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Klassen ({num_classes}): {train_dataset.classes}")

    # 4. Model, Loss, Optimizer
    model = ResNetLightCNN2(num_classes=num_classes).to(DEVICE)
    #criterion = nn.CrossEntropyLoss()

    #[w_ang, w_dis, w_fear, w_happy, w_sad, w_surprise]
    class_weights = torch.tensor([1.2, 1.5, 2.0, 1.0, 1.0, 1.0], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 5.Scheduler
    '''''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5,
    )
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=6,
    factor=0.5,
    min_lr=1e-5
    )

    best_val_acc = 0.0

    best_val_loss = float("inf")

    epoch_log = []

    log_path = "src/evaluation/outputs/epoch_metrics.csv"

    # 6. Training Loop
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

        # 7.Validation
        val_acc, val_loss = validate(model, val_loader, criterion)

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        
        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "train_acc": epoch_acc,
            "val_acc": val_acc,
            "learning_rate": current_lr
        })

        print("LR:", current_lr)

        # 8. Save best model
        '''''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    🌟 New Record! Model saved to {save_path}")
        '''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    🌟 New Record! Model saved to {save_path}")


    print("=" * 50)
    print("Training completed.")
    print(
        f"The best model achieved {best_val_acc:.2f}% Accuracy on the validation data."
        #f"The best model achieved a validation loss of {best_val_loss:.4f} on the validation data."
    )
    print(f"Saved as: {save_path}")
    print("=" * 50)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames = epoch_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_log)


if __name__ == "__main__":
    main()
