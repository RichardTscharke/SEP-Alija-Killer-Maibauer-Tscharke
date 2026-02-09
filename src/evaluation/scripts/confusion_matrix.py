import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(output_dir, figure_dir, normalized):
    '''
    Plots a confusion matrix based on inference results.
    -> Visualizes how often eacch true class is predicted as each other class

    Two modes:
    - normalized=False : show absolute counts
    - normalized=True  : show percentages
    '''

    # Load inference results
    y_true = np.load(output_dir / "y_true.npy")
    y_pred = np.load(output_dir / "y_pred.npy")
    class_names = np.load(output_dir / "class_names.npy", allow_pickle = True)

    # Compute confusion matrix (rows = ground truth, columns = predictions)
    cm = confusion_matrix (y_true, y_pred)

    # Normalize rows so each row sums to 1
    if normalized:
        cm = cm.astype(float) / cm.sum(axis = 1, keepdims = True)

    # Plotting matrix with heatmap overlay
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = ".2f" if normalized else "d", cmap = "Blues", xticklabels = class_names, yticklabels = class_names,)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (normalized)" if normalized else ""))
    plt.tight_layout()

    # Save confusion matrix
    save_name = "confusion_matrix_norm.png" if normalized else "confusion_matrix.png"
    plt.savefig(figure_dir / save_name, dpi=300)