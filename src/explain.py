import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self.fh = target_layer.register_forward_hook(self.forward_hook)
        self.bh = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.feature_maps = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        logits[0, class_idx].backward()

        pooled_gradients = self.gradients.mean(dim=(0, 2, 3))

        fmap = self.feature_maps.clone()
        for i in range(fmap.shape[1]):
            fmap[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(fmap, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= (heatmap.max() + 1e-8)

        heatmap = heatmap.detach().cpu().numpy()
        return heatmap

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()


def overlay_gradcam(frame, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_color[heatmap < 0.2] = 0
    return cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

