import os
import torch
import shutil
from pathlib import Path

def get_device():
    '''
    Determines the best available computation device.
    Priority:
    1) CUDA-capable GPU
    2) Apple Silicon GPU (MPS)
    3) CPU fallback
    '''
    if torch.cuda.is_available():
        print(f"\n🚀 GPU Activated: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("🍏 Apple Silicon GPU (MPS) erkannt.")
        return torch.device("mps")
    else:
        print(
            "\n⚠️  No GPU detected. Training will be performed on CPU, which may be slow."
        )
        return torch.device("cpu")

# Paths to data directories
TRAIN_DIR = "data/train"
VAL_DIR   = "data/validate"
MODEL_DIR = "model_paths"

# OUTPU_DIR for evaluation
OUTPUT_DIR = Path("src/evaluation/outputs")


def get_unique_model_path(base_name):
    '''
    Generates a unique file path for saving a trained model.
    The function increments a version counter to avoid overwriting existing models.
    '''

    # Find the next available model save path: ResNetLight2_v0.pth, ResNetLight2_v1.pth, etc.
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    counter = 0
    while True:
        filename = f"{base_name}_v{counter}.pth"
        full_path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(
            full_path
        ):  # File does not exist, so we can use this name
            return full_path, counter

        counter += 1

def prepare_output_dir_evaluation():
    '''
    Prepares and returns a clean output directory for evaluation results.
    If the directory already exists, it is deleted and recreated.
    -> Evaluation outputs correspond to the latest training run.
    '''
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def compute_class_weights(train_dataset, use_inv_freq_w: bool, custom_weights, device):
    '''
    Computes class weights for handling class imbalance in the loss function.
    Two modes are supported:
    - Inverse frequency weighting
    - Custom class weights
    The mode and custom weights can be manually setup in the train.py interface.
    '''
    if use_inv_freq_w:
        # Inverse frequency weights approach for tests
        from collections import Counter

        # Count class distribution
        targets = [label for _, label in train_dataset.samples]
        class_counts = Counter(targets)

        num_classes = len(train_dataset.classes)
        total = sum(class_counts.values())

        # inverse frequency
        class_weights = [
            total / (num_classes * class_counts[i]) for i in range(num_classes)
        ]
    
    else:
        # Custom class weights
        class_weights = custom_weights

    return torch.tensor(class_weights, dtype=torch.float, device=device)