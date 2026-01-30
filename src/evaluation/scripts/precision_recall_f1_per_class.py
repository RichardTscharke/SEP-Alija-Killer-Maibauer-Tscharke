import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = PROJECT_ROOT / "src" / "evaluation" / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"

def plot_prec_recall_f1_p_class():

    y_true = np.load(OUTPUT_DIR / "y_true.npy")
    y_pred = np.load(OUTPUT_DIR / "y_pred.npy")
    class_names = np.load(OUTPUT_DIR / "class_names.npy", allow_pickle=True)

    precision = precision_score(y_true, y_pred, labels=range(len(class_names)), average=None)
    recall = recall_score(y_true, y_pred, labels=range(len(class_names)), average=None)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    for cls, p, r, f in zip(class_names, precision, recall, f1):
        print(f"{cls}: Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width, precision, width, label='Precision')
    bars2 = ax.bar(x, recall, width, label='Recall')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score')

    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall & F1 per class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    plt.savefig(FIGURE_DIR / "prec_recall_f1_per_class.png", dpi=300)