import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(output_dir, figure_dir, normalized=True):

    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)

    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": ["Times"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 1.0,
    }):

        y_true = np.load(output_dir / "y_true.npy")
        y_pred = np.load(output_dir / "y_pred.npy")
        class_names = np.load(output_dir / "class_names.npy", allow_pickle=True)

        cm = confusion_matrix(y_true, y_pred)

        if normalized:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(3.4, 3.1))

        im = ax.imshow(cm, interpolation="nearest")

        # colorbar
        cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
        if normalized:
            cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom")
        else:
            cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True"
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Annotate cells
        fmt = ".2f" if normalized else "d"
        thresh = cm.max() / 2.

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i,
                    format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7
                )

        fig.tight_layout()

        save_name = "confusion_matrix_norm.pdf" if normalized else "confusion_matrix.pdf"
        plt.savefig(figure_dir / save_name, format="pdf", bbox_inches="tight")
        plt.close(fig)

