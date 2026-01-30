import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = PROJECT_ROOT / "src" / "evaluation" / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(exist_ok=True)

def plot_confusion_matrix(normalized = False):

    y_true = np.load(OUTPUT_DIR / "y_true.npy")
    y_pred = np.load(OUTPUT_DIR / "y_pred.npy")
    class_names = np.load(OUTPUT_DIR / "class_names.npy", allow_pickle = True)

    cm = confusion_matrix (y_true, y_pred)

    if normalized:
        cm = cm.astype(float) / cm.sum(axis = 1, keepdims = True)

    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = ".2f" if normalized else "d", cmap = "Blues", xticklabels = class_names, yticklabels = class_names,)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (normalized)" if normalized else ""))
    plt.tight_layout()

    save_name = "confusion_matrix_norm.png" if normalized else "confusion_matrix.png"
    plt.savefig(FIGURE_DIR / save_name, dpi=300)