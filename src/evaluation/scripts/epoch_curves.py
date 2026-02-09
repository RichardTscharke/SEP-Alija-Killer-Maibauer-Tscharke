import pandas as pd
import matplotlib.pyplot as plt


def plot_epoch_curves(figure_dir, csv_path):
    '''
    Orchestrates the plotting for all epoch-based training curves.
    Loads the logged epoch metrics from CSV and generates:
    - loss curve
    - accuracy curve
    - learning rate schedule
    '''
    data = load_epoch_metrics(csv_path)

    plot_loss_curve(figure_dir, data)
    plot_accuracy_curve(figure_dir, data)
    plot_learning_rate(figure_dir, data)

# Load the epoch-wise training metrics from CSV
def load_epoch_metrics(path):
    return pd.read_csv(path)

# Plot training and validation loss over epochs
def plot_loss_curve(figure_dir, data):
    plt.figure()

    plt.plot(data["epoch"], data["train_loss"], label="Train Loss")
    plt.plot(data["epoch"], data["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.savefig(figure_dir / "loss_curve.png", dpi=300)
    plt.close()

# Plot training and validation accuracy over epochs
def plot_accuracy_curve(figure_dir, data):
    plt.figure()

    plt.plot(data["epoch"], data["train_acc"], label="Train Accuracy")
    plt.plot(data["epoch"], data["val_acc"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.savefig(figure_dir / "accuracy_curve.png", dpi=300)
    plt.close()

# Plot the learning rate schedule over epochs
def plot_learning_rate(figure_dir, data):
    plt.figure()

    plt.plot(data["epoch"], data["learning_rate"])

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    plt.savefig(figure_dir / "learning_rate_curve.png", dpi=300)
    plt.close()