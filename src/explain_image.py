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
- Emotion class probablities for the image

The forward and backward hooks logic can be found within the GradCAM class.
If you trained your own model please load the model path into the 'models' folder lying within the root directory.
You can adjust the following parameters in order to achieve different results:
'''

# By default: An image of the RAF_raw dataset (assuming it is stored within the data folder)
IMAGE_PATH   = Path("/Users/richardachtnull/Desktop/präsi1/richard.JPG")

# By default: Our best trained model
MODEL_PATH   = "models/ResNetLight_v2.pth"

# By default: A stage and layer we often achieved satisfying results with
# Useful options are: "stage2.conv2", "stage3.conv1" and "stage3.conv2" (stage 1 perhaps for edges)
TARGET_LAYER = "stage2.conv2"

# Determines the sufficient strength of a signal to be visible in the heatmap
THRESHOLD    = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(image_path, model_path, target_layer):
    '''
    Runs a single-image Grad-CAM explanation pipeline:
    - loads trained model
    - resolves target convolutional layer
    - computes Grad-CAM
    - visualizes CAM overlay and class probabilities
    '''
    model = load_model( 
        model_class=ResNetLightCNN2,
        weight_path=model_path,
        device=device
    )

    # Resolve layer dynamically based on the parameters at the top
    modules = dict(model.named_modules())
    assert TARGET_LAYER in modules, f"Unknown layer: {TARGET_LAYER}"
    target_layer = modules[target_layer]

    # Computes the Grad-CAM heatmap by running forward and backward pass through the model using the GradCAM clas
    result = explain_gradcam(
        model=model,
        image_path=image_path,
        target_layer=target_layer,
        device=device,
    )
    # Plots a graphic of original image, aligned face, heatmap overlay and class probabilities
    visualize(
        original_img=result["original_img"],
        aligned_img=result["aligned_img"],
        cam=result["cam"],
        probs=result["probs"],
        threshold=THRESHOLD
    )


if __name__ == "__main__":
    main(IMAGE_PATH, MODEL_PATH, TARGET_LAYER)



