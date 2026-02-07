from evaluation.scripts.inference import calculate_inference
from evaluation.scripts.epoch_curves import plot_epoch_curves
from evaluation.scripts.precision_recall_f1_per_class import plot_prec_recall_f1_p_class
from evaluation.scripts.confusion_matrix import plot_confusion_matrix

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dir_path = PROJECT_ROOT / "figures"

def main():

    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)

    calculate_inference(MODEL_PATH = "models/ResNetLight_2_v2.pth") # Make sure this is the latest model
    print(f"[INFO] Inference calculated.")

    plot_epoch_curves(csv_path = "src/evaluation/outputs/epoch_metrics.csv")
    print(f"[INFO] Epoch curves drawn.")

    plot_prec_recall_f1_p_class()
    print(f"[INFO] Precision, Recall and F1 per class diagramm created.")

    plot_confusion_matrix(normalized = True) # Flag to True for percentages and False for absolute values
    print(f"[INFO] Confusion Matrix drawn.")


if __name__ == "__main__":
    main()
