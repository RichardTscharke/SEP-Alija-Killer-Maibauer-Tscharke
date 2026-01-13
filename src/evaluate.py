import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from model import CustomEmotionCNN
import sys

# Configurations
MODEL_PATH = "models/raf_cnn_v5.pth"  # Make sure this is the latest trained model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64


def evaluate_model(test_dir):
    print("\n" + "=" * 30)
    print(f"🚀 STARTING EVALUATION")
    print(f"🧠 Model: {MODEL_PATH}")
    print(f"📂 Data:  {test_dir}")
    print("=" * 30 + "\n")

    # Load test dataset
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    classes = test_dataset.classes

    # Load the trained model
    model = CustomEmotionCNN(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # CLASSIFICATION REPORT
    print("\n" + "=" * 50)
    print("📊 Classification Report:")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=classes))

    # CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix as a table
    df_table = pd.DataFrame(cm, index=classes, columns=classes)
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX as Table:")
    print("=" * 50)
    print(df_table)
    print("=" * 50 + "\n")

    # create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for Emotion Classification")

    # Save the confusion matrix image (for reports)
    save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\nImage saved as {save_path}")


if __name__ == "__main__":
    # Default path on server
    default_folder = "data/RAF_aligned_processed/test"

    # Allow command line argument for folder path
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = default_folder

    evaluate_model(folder_path)
