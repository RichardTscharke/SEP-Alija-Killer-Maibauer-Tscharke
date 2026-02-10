import torch
from .cam.cam_runner import run_gradcam
from .cam.cam_projection import cam_to_original

def np_to_tensor(np_img, device):
    '''
    Converts an OpenCV BGR image to a normalized torch tensor:
    shape: (1, 3, H, W), range approx [-1, 1]
    '''
    np_img = np_img[:, :, ::-1].astype("float32") / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)

def explain_sample(sample, model, target_layer, device):
    '''
    Runs Grad-CAM inference on a preprocessed sample.
    Assumes sample contains an aligned face and meta information.
    '''
    if "input_tensor" not in sample:
        sample["input_tensor"] = np_to_tensor(sample["image"], device)

    input_tensor = sample["input_tensor"]

    cam, logits = run_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_layer=target_layer
    )

    probs = torch.softmax(logits, dim=1)[0].cpu().detach().numpy()

    cam_original = cam_to_original(
        cam=cam,
        meta=sample["meta"],
        original_shape=sample["original_img"].shape
    )

    sample.update({
        "cam": cam,
        "cam_original": cam_original,
        "logits": logits,
        "probs": probs
    })

    return sample