'''
import cv2
from pathlib import Path
from preprocessing.detectors.retinaface import RetinaFaceDetector
from preprocessing.aligning.detect import detect_and_preprocess

image_path = Path("/Users/richardachtnull/Desktop/IMG_0477.jpg")


def main():
    
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = RetinaFaceDetector(device= "cpu")

    result = detect_and_preprocess(
        image,
        detector,
        vis=True
    )
    
    
    image = cv2.imread(image_path)

    detector = RetinaFaceDetector(device="cpu")

    sample = detect_and_preprocess(image, detector, vis=True)

    print(f"Crop Offset: {sample['meta']['crop_offset']}")
    print(f"M: {sample['meta']['affine_M']}")
    print(f"M^-1: {sample['meta']['affine_M_inv']}")

if __name__ == "__main__":

    main()
'''

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import recall_score, f1_score


def compute_recall_f1(output_dir):
    output_dir = Path(output_dir)

    y_true = np.load(output_dir / "y_true.npy")
    y_pred = np.load(output_dir / "y_pred.npy")
    class_names = np.array(
    ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
    )


    recall = recall_score(
        y_true, y_pred,
        labels=range(len(class_names)),
        average=None
    )

    f1 = f1_score(
        y_true, y_pred,
        labels=range(len(class_names)),
        average=None
    )

    return recall, f1, class_names


def plot_original_vs_aligned(orig_dir, align_dir, figure_dir):

    orig_dir = Path(orig_dir)
    align_dir = Path(align_dir)
    figure_dir = Path(figure_dir)

    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
    }):

        recall_o, f1_o, class_names = compute_recall_f1(orig_dir)
        recall_a, f1_a, _ = compute_recall_f1(align_dir)

        x = np.arange(len(class_names))
        width = 0.35

        fig, axes = plt.subplots(
            1, 2,
            figsize=(6.8, 3.0),
            sharey=True
        )

        # ----- Original -----
        axes[0].bar(x - width/2, recall_o, width, label="Recall")
        axes[0].bar(x + width/2, f1_o, width, label="F1")
        axes[0].set_title("Original Images")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_names, rotation=45, ha="right")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Score")

        # ----- Aligned -----
        axes[1].bar(x - width/2, recall_a, width, label="Recall")
        axes[1].bar(x + width/2, f1_a, width, label="F1")
        axes[1].set_title("Aligned Faces")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[1].set_ylim(0, 1)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,
                   labels,
                   loc="upper center",
                   ncol=2,
                   frameon=False,
                   fontsize=10
        )

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        plt.savefig(
            figure_dir / "original_vs_aligned.pdf",
            format="pdf",
            bbox_inches="tight"
        )

        plt.close(fig)


if __name__ == "__main__":

    plot_original_vs_aligned(
        "/Users/richardachtnull/Desktop/final_report/results/original",
        "/Users/richardachtnull/Desktop/final_report/results/aligned",
        "/Users/richardachtnull/Desktop/final_report/results"
    )
