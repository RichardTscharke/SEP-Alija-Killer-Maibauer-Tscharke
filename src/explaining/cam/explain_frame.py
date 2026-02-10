import torch
from .cam_runner import run_gradcam
from .cam_projection import cam_to_original


def explain_frame(sample, model, target_layer):
    '''
    Runs Grad-CAM inference on a preprocessed frame.
    Assumes sample contains an aligned face and meta information.
    Mutates and returns sample.
    '''
    input_tensor = sample["input_tensor"]

    cam, logits = run_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_layer=target_layer
    )

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    cam_original = cam_to_original(
        cam=cam,
        meta=sample["meta"],
        original_shape=sample["original_img"].shape
    )

    sample.update({
        "cam": cam,
        "cam_original": cam_original,
        "logits": logits,
        "probs": probs,
        "pred_idx": int(probs.argmax())
    })

    return sample