import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Define a consistent plotting style for all epoch curves to Match the Latex requirements
PLOT_STYLE = {
    "font.family": "STIXGeneral",
    #"font.serif": ["Times"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
}


def plot_epoch_curves(figure_dir, csv_path):
    """
    Orchestrates the plotting for all epoch-based training curves.
    Loads the logged epoch metrics from CSV and generates:
    - loss curve
    - accuracy curve
    - learning rate schedule
    """
    data = load_epoch_metrics(csv_path)

    plot_loss_curve(figure_dir, data)
    plot_accuracy_curve(figure_dir, data)
    plot_learning_rate(figure_dir, data)


# Load the epoch-wise training metrics from CSV
def load_epoch_metrics(path):
    return pd.read_csv(path)


# Plot training and validation loss over epochs
def plot_loss_curve(figure_dir, data):
    with mpl.rc_context(PLOT_STYLE):

        fig, ax = plt.subplots(figsize=(3.4, 2.6))

        ax.plot(data["epoch"], data["train_loss"], label="Train Loss")
        ax.plot(
            data["epoch"], data["val_loss"], linestyle="--", label="Validation Loss"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()

        plt.savefig(figure_dir / "loss_curve.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)


# Plot training and validation accuracy over epochs
def plot_accuracy_curve(figure_dir, data):
    with mpl.rc_context(PLOT_STYLE):

        fig, ax = plt.subplots(figsize=(3.4, 2.6))

        ax.plot(data["epoch"], data["train_acc"], label="Train Accuracy")
        ax.plot(
            data["epoch"], data["val_acc"], linestyle="--", label="Validation Accuracy"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()

        plt.savefig(
            figure_dir / "accuracy_curve.pdf", format="pdf", bbox_inches="tight"
        )
        plt.close(fig)


# Plot the learning rate schedule over epochs
def plot_learning_rate(figure_dir, data):
    with mpl.rc_context(PLOT_STYLE):

        fig, ax = plt.subplots(figsize=(3.4, 2.6))

        ax.plot(data["epoch"], data["learning_rate"])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        fig.tight_layout()

        plt.savefig(
            figure_dir / "learning_rate_curve.pdf", format="pdf", bbox_inches="tight"
        )
        plt.close(fig)
