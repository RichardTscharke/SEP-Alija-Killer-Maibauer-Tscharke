import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.ResNetLight2 import ResNetLightCNN2
from models.ResNetLight import ResNetLightCNN
from models.RafCustom import RafCustomCNN


TEST_DIR = "data/test"

def calculate_inference(output_dir, model_path, config, device):
    '''
    Runs inference on the test dataset using a trained model.
    Stores the prediction aretefacts fro downstream evaluation and visiualization.
    Saved in outputs folder:
    - y_pred.npy      : predicted class indices
    - y_true.npy      : ground truth class indices
    - class_names.npy : class label names
    '''

    DEVICE = device

    # Image augmentation (same adjustments as for validation)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5],
                             std = [0.5, 0.5, 0.5]),
    ])

    # Test dataset and loader
    test_dataset = datasets.ImageFolder(TEST_DIR, transform = transform)
    test_loader = DataLoader(test_dataset, batch_size = config["batch_size"], shuffle = False)

    class_names = test_dataset.classes
    num_classes = len(class_names)

    # Model selection based on training configurations
    if config["model"] == "ResNetLight2":
        model = ResNetLightCNN2(num_classes=num_classes).to(DEVICE)
    elif config["model"] == "ResNetLight1":
        model = ResNetLightCNN(num_classes=num_classes).to(DEVICE)
    else:
        model = RafCustomCNN(num_classes=num_classes).to(DEVICE)

    # Load trained weights and switch to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location = DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    # Inference loop (Gradients are disabled for efficiency and correctness)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim = 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Save inference results
    np.save(output_dir / "y_pred.npy", np.array(all_preds))
    np.save(output_dir / "y_true.npy", np.array(all_labels))
    np.save(output_dir / "class_names.npy", np.array(class_names))

    print("[INFO] Evaluation data saved:")
    print(f" - {output_dir / 'y_pred.npy'}")
    print(f" - {output_dir / 'y_true.npy'}")
    print(f" - {output_dir / 'class_names.npy'}")