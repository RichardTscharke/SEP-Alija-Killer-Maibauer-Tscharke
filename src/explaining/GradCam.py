import cv2
import torch
import numpy as np

class GradCAM:
    def __init__(self, target_layer):
        self.target_layer = target_layer

        self.feature_maps = None
        self.gradients = None

        self.fh = target_layer.register_forward_hook(self.forward_hook)
        self.bh = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.feature_maps = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, logits, class_idx):

        score = logits[0, class_idx]
        self.target_layer.zero_grad()
        score.backward(retain_graph = True)

        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError(
                "[INFO] GradCAM hooks not triggered. "
                "Make sure forward() was called before generate()."
            )

        weights = self.gradients.mean(dim = (2, 3), keepdim = True)

        cam = (weights * self.feature_maps).sum(dim = 1)
        cam = torch.relu(cam)

        cam = cam.squeeze()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        cam = cam.detach()

        return cam.cpu().numpy()
    
    def remove(self):
        self.fh.remove()
        self.bh.remove()