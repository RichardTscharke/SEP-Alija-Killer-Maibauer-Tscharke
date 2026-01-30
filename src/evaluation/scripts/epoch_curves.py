import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(exist_ok=True)


def plot_epoch_curves(csv_path):
    data = load_epoch_metrics(csv_path)

    plot_loss_curve(data)
    plot_accuracy_curve(data)
    plot_learning_rate(data)


def load_epoch_metrics(path):
    return pd.read_csv(path)


def plot_loss_curve(data):
    plt.figure()

    plt.plot(data["epoch"], data["train_loss"], label="Train Loss")
    plt.plot(data["epoch"], data["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.savefig(FIGURE_DIR / "loss_curve.png")
    plt.close()


def plot_accuracy_curve(data):
    plt.figure()

    plt.plot(data["epoch"], data["train_acc"], label="Train Accuracy")
    plt.plot(data["epoch"], data["val_acc"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.savefig(FIGURE_DIR / "accuracy_curve.png")
    plt.close()


def plot_learning_rate(data):
    plt.figure()

    plt.plot(data["epoch"], data["learning_rate"])

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    plt.savefig(FIGURE_DIR / "learning_rate_curve.png", dpi=300)
    plt.close()