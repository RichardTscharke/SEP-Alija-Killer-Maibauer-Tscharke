from explaining.GradCam import GradCAM


def explain_gradcam(
    model,
    input_tensor,
    target_layer,
    class_idx=None,
    ):
    '''
    Computes Grad-CAM for a single image.
    Pipeline:
    - face detection & alignment
    - forward pass
    - Grad-CAM generation for predicted class

    Returns:
    - original image
    - aligned face image
    - CAM heatmap
    - class probabilities
    '''

    # Evaluate model
    model.eval()

    # Forward pass & class probabilities
    grad_cam = GradCAM(target_layer)

    logits = model(input_tensor)

    if class_idx is None:
        class_idx = logits.argmax(dim=1).item()

    # Generate Grad-CAM for predicted class
    cam = grad_cam.generate(logits, class_idx)

    # Remove used forward and backward hooks
    grad_cam.remove()

    return cam, logits
