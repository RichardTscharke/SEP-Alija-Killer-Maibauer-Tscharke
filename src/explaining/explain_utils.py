import cv2
import torch
from models.ResNetLight2 import ResNetLightCNN2
from preprocessing.aligning.detect import detect_and_preprocess


def load_model(model_class, model_path, device, num_classes):

    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def resolve_model_and_layer(model_path, target_layer, device):

    model = load_model(
        model_class=ResNetLightCNN2,
        model_path=model_path,
        device=device,
        num_classes=6
    )

    modules = dict(model.named_modules())
    assert target_layer in modules, f"Unknown layer: {target_layer}"

    return model, modules[target_layer]


def np_to_tensor(np_img, device):
    '''
    Converts an OpenCV BGR image to a normalized torch tensor:
    shape: (1, 3, H, W), range approx [-1, 1]
    '''

    # [0, 255] -> [0, 1]
    np_img = np_img[:, :, ::-1].astype("float32") / 255.0

    # BGR -> RGB
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)

    # [0, 1] -> [-1, 1]
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)


def preprocess_frame(frame, detector, device):
    '''
    Runs face detection and alignment on a single frame.
    Returns a sample dict extended with original image and input tensor.
    Contents:
    - Aligned face image
    - Original image
    - Meta informaton for backprojection
    '''

    # Clip -> Crop -> Align (our preprocessing pipeline)
    sample = detect_and_preprocess(frame, detector)

    if sample is None:
        return None

    sample["original_img"] = frame

    # numpy array -> torch tensor
    sample["input_tensor"] = np_to_tensor(sample["image"], device)

    return sample


def preprocess_image(image_path, detector, device):
    '''
    Wrapper function for an image as input instead of a frame.
    '''
    frame = cv2.imread(str(image_path))
    return preprocess_frame(frame, detector, device)


def run_model(model, input_tensor, require_grad=False):
    '''
    Pure forward pass (no forwatd pass or hooks).
    '''
    model.eval()

    if require_grad:
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    else:
        with torch.no_grad():
            logits = model(input_tensor)
            probs  = torch.softmax(logits, dim=1)

    return logits, probs

