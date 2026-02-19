from src.explaining.cam.explain_frame import explain_frame

def infer_frame(
        sample,
        model,
        target_layer,
        enable_xai,
        run_model_f
):
    '''
    Performs a single inference step on a preprocessed sample.
    The XAI pipeline is called if asked for by the user.
    If enable_xai=False, only probabilities are returned (no cam).
    '''

    # Grad-CAM explanation enabled
    if enable_xai:
        
        # Run Grad-CAM (forward, backward and projection)
        # explain_frame returns CAM projected into original coordinate system
        explained = explain_frame(
            sample=sample,
            model=model,
            target_layer=target_layer
        )

        # Extract probabilities and cam for the worker thread
        return {
            "probs": explained["probs"],
            "cam": explained["cam_original"]
        }
    
    # Only forward pass, no gradient computation required
    else:
        
        # Extract already prepared model input tensor
        input_tensor = sample["input_tensor"]

        # Perform forward pass
        _, probs = run_model_f(model, input_tensor)

        # Tensor -> numpy array
        probs_np = probs.squeeze(0).cpu().numpy()

        # Extract probabilities and NO cam for the worker thread
        return {
            "probs": probs_np,
            "cam": None
        }