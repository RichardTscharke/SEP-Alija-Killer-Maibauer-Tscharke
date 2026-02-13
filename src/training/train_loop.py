import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from .train_utils import (
    TRAIN_DIR,
    VAL_DIR,
    prepare_output_dir_evaluation,
    get_unique_model_path,
    compute_class_weights,
)
from evaluation.scripts.evaluate import run_evaluate

from models.RafCustom import RafCustomCNN
from models.ResNetLight import ResNetLightCNN
from models.ResNetLight2 import ResNetLightCNN2


def trainings_loop(config: dict, device: torch.device):
    """
    Orchestrates the full training pipeline.
    - Dataset preperation and augmentation
    - Model intialization
    - Loss, weights, optimizer and scheduler setup
    - Early stopping and best-model checkpointing
    - Logging of epoch-wise metrics
    - Calls the evaluation script

    All hyperparameters and training options can be manually setup within the train.py interface.
    """
    MODEL = config["model"]
    TRAIN_ON = config["train_on"]

    USE_INV_FREQ_W = config["use_inv_freq_w"]
    CUSTOM_CLASS_WEIGHTS = config["custom_weights"]

    LEARNING_RATE = config["learning_rate"]
    EPOCHS = config["epochs"]
    EARLY_STOP_PATIENCE = config["early_stop_patience"]

    BATCH_SIZE = config["batch_size"]
    NUM_WORKERS = config["num_workers"]

    DEVICE = device
    use_amp = device.type == "cuda"

    # 0. prepare evaluation outputs directory
    output_dir = prepare_output_dir_evaluation()

    # 1. model save path
    save_path, version_id = get_unique_model_path(base_name=config["model"])
    print("=" * 50)
    print(f"💾 This training will be saved as: {save_path}")
    print("=" * 50)

    # 2. Data Transformations & Augmentations
    train_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=4),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.01
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.3, scale=(0.01, 0.1), ratio=(0.3, 3.3)),
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

    """
    # --- WEIGHTED RANDOM SAMPLER IMPLEMENTATION ---
    print("Calculating weights for WeightedRandomSampler...")

    # Get all target labels from the dataset
    targets = train_dataset.targets

    # Calculate the count of each class (0 to num_classes-1)
    class_counts = torch.bincount(torch.tensor(targets))

    # Calculate weight for each class: w = 1 / count
    # (Rare classes get high weights, frequent classes get low weights)
    class_weights = 1.0 / class_counts.float()

    # Assign the corresponding class weight to each individual sample image
    sample_weights = class_weights[targets]

    # Create the sampler
    # replacement=True allows resampling the same image multiple times (oversampling)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    # -----------------------------------------------
    """
    # NOTE: shuffle must be False when using a sampler!
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        # sampler=sampler,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Classes ({num_classes}): {train_dataset.classes}")

    # 4. Model, Weights, Loss, Optimizer
    if MODEL == "ResNetLight2":
        model = ResNetLightCNN2(num_classes=num_classes).to(DEVICE)
    elif MODEL == "ResNetLight1":
        model = ResNetLightCNN(num_classes=num_classes).to(DEVICE)
    elif MODEL == "RafCustom":
        model = RafCustomCNN(num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {MODEL}")

    # Compute class weights
    class_weights = compute_class_weights(
        train_dataset, USE_INV_FREQ_W, CUSTOM_CLASS_WEIGHTS, DEVICE
    )
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.05
    )  # Using class weights to handle imbalance without sampler.
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # If you want to use class weights directly in the loss function instead of WeightedRandomSampler, comment the above line and uncomment this one.

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scaler = GradScaler(enabled=use_amp)

    # 5.Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    epochs_no_improve = 0

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

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Mixed precision
            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(epoch + batch_idx / len(train_loader))

            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(train_loader))
                

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
        val_acc, val_loss = validate(
            model, val_loader, criterion, device=DEVICE, use_amp=use_amp
        )


        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "train_acc": epoch_acc,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }
        )

        print("LR:", current_lr)

        # 8. Save best model

        if TRAIN_ON == "val_loss":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"    🌟 New Record! Model saved to {save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        elif TRAIN_ON == "val_acc":
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"    🌟 New Record! Model saved to {save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"⏹ Early stopping triggered after epoch {epoch + 1}")
            break

    # Save the best val_loss model for evaluation
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    print("=" * 50)
    print("[INFO] Training completed.")

    if TRAIN_ON == "val_loss":
        print(
            f"[INFO] The best model achieved a validation loss of {best_val_loss:.4f} on the validation data."
        )
    elif TRAIN_ON == "val_acc":
        print(
            f"[INFO] The best model achieved a validation accuracy of {best_val_acc:.4f}% on the validation data."
        )

    print(f"[INFO] Saved as: {save_path}")
    print("=" * 50)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_log)

    # Start the evalution based on the training congfigurations and the trained model
    run_evaluate(
        output_dir=output_dir, model_path=save_path, config=config, device=DEVICE
    )


def validate(model, loader, criterion, device, use_amp):
    """
    Evaluates the model on the evaluation dataset.
    Computes:
    - Average validation loss
    - Validation accuracy

    The model is set to evaluation mode and gradients are disabled.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
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
