import os
import sys
import torch
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(), "src"))
from src.evaluation.scripts.evaluate import run_evaluate

# Evaluation configuration
# Ensure these parameters match the training setup
CONFIG = {
    "model": "ResNetLight2",
    "batch_size": 64,
    "num_workers": 4,
}

# File paths
MODEL_PATH = Path("models/ResNetLight2_v2.pth")
OUTPUT_DIR = Path("src/evaluation/outputs")


def main():
    # Select computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify model existence
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    print(f"[INFO] Starting evaluation for: {MODEL_PATH}")
    print(f"[INFO] Device: {device}")

    # Execute evaluation pipeline
    run_evaluate(OUTPUT_DIR, MODEL_PATH, CONFIG, device)

    print("[INFO] Evaluation finished.")


if __name__ == "__main__":
    main()
