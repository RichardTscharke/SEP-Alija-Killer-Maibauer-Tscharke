import torch

from explaining.explain_utils import load_model
from explaining.explain_gradcam import explain_gradcam
from explaining.visualize_gradcam import visualize
from models.ResNetLight2 import ResNetLightCNN2

# These can be configurated freely:
IMAGE_PATH   = "/Users/richardachtnull/Desktop/disgusted.jpg"
MODEL        = "models/ResNetLight_v2.pth"
TARGET_LAYER = "stage2.conv2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    model = load_model(
        model_class=ResNetLightCNN2,
        weight_path=MODEL,
        device=device
    )

    # Resolve layer dynamically
    target_layer = dict(model.named_modules())[TARGET_LAYER]

    result = explain_gradcam(
        model=model,
        image_path=IMAGE_PATH,
        target_layer=target_layer,
        device=device,
        threshold=0.4
    )

    visualize(
        original_img=result["original_img"],
        aligned_img=result["aligned_img"],
        cam=result["cam"],
        probs=result["probs"],
        threshold=0.4
    )


if __name__ == "__main__":
    main()



