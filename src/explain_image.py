import torch
from pathlib import Path

from explaining.explain_utils import load_model
from explaining.explain_gradcam import explain_gradcam
from explaining.visualize_gradcam import visualize
from models.ResNetLight2 import ResNetLightCNN2

'''
This is the XAI interface (for single image) of our project.
It plots a graphic containing:
- original image
- preprocessed image
- Grad-CAM heatmap for the image
- Heatmap overlayed onto the preprocessed image
- Emotion Class probablities for the image

If you trained your own model please load the model path into the pro
'''
IMAGE_PATH   = Path("/Users/richardachtnull/Desktop/präsi1/richard.JPG")
MODEL_PATH   = "models/ResNetLight_v2.pth"
TARGET_LAYER = "stage2.conv2"
THRESHOLD    = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(image_path, model_path, target_layer_name):
    '''
    Runs a single-image Grad-CAM explanation pipeline:
    - loads trained model
    - resolves target convolutional layer
    - computes Grad-CAM
    - visualizes CAM overlay and class probabilities
    '''
    model = load_model( 
        model_class=ResNetLightCNN2,
        weight_path=MODEL_PATH,
        device=device
    )

    # Resolve layer dynamically
    modules = dict(model.names_modules())
    assert TARGET_LAYER in modules, f"Unknown layer: {TARGET_LAYER}"
    target_layer = modules[TARGET_LAYER]

    result = explain_gradcam(
        model=model,
        image_path=IMAGE_PATH,
        target_layer=target_layer,
        device=device,
        threshold=THRESHOLD
    )

    visualize(
        original_img=result["original_img"],
        aligned_img=result["aligned_img"],
        cam=result["cam"],
        probs=result["probs"],
        threshold=THRESHOLD
    )


if __name__ == "__main__":
    main(IMAGE_PATH, MODEL_PATH, TARGET_LAYER)



