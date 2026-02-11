import torch
from .cam_runner import run_gradcam
from .cam_projection import cam_to_original


def explain_frame(sample, model, target_layer):
    '''
    Runs Grad-CAM on a single preprocessed frame.
    Assumes sample contains an aligned face and meta information.
    Mutates and returns sample by adding cam, logits, probs and prediction index.
    '''

    # Model input tensor (aligned face, batch size = 1)
    input_tensor = sample["input_tensor"]

    # Run forward + backward pass and compute Grad-CAM on aligned face
    cam, logits = run_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_layer=target_layer
    )

    # Convert logits to class probabilities 
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    # Project CAM from aligned-face coordinates back to original frame
    cam_original = cam_to_original(
        cam=cam,
        meta=sample["meta"],
        original_shape=sample["original_img"].shape
    )

    # Store explanation results and predictions in sample dict
    sample.update({
        "cam": cam,
        "cam_original": cam_original,
        "logits": logits,
        "probs": probs,
        "pred_idx": int(probs.argmax())
    })

    return sample