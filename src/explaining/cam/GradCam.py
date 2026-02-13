import torch

class GradCAM:
    '''
    Implements Grad-CAM for a given convolutional layer of a model.
    Pipeline:
    - Instantiate with target layer
    - Run a forward pass through the model
    - Call the generate function
    '''
    def __init__(self, target_layer):
        '''
        Registers forward and backward hooks on the target layer.
        Parameter: convolutional layer whose activations should be explained.
        '''
        self.target_layer = target_layer

        # Updated during forward and backward pass
        self.feature_maps = None
        self.gradients = None

        # Register hooks
        self.fh = target_layer.register_forward_hook(self.forward_hook)
        self.bh = target_layer.register_full_backward_hook(self.backward_hook)

    # Save feature maps produced by the target layer during the forward pass
    def forward_hook(self, module, input, output):
        self.feature_maps = output.detach()

    # Save gradients of the target layer output
    def backward_hook(self, module, grad_input, grad_output):
        # First element of the tuple corresponds to the gradient with regards to thre layer output
        self.gradients = grad_output[0].detach()

    def generate(self, logits, class_idx):
        '''
        Takes the logits and target class index.
        Assumes logits come from the same forward pass that triggered the hooks.
        Generates and returns a Grad-CAM heatmap for a specified class.
        '''

        # Select score of target class
        score = logits[0, class_idx]

        # Zero all model gradients before backward
        score.backward()

        # Make sure the hooks have been triggered
        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError(
                "[INFO] GradCAM hooks not triggered. "
                "Make sure forward() was called before generate()."
            )

        # Global Average Pooling
        weights = self.gradients.mean(dim = (2, 3), keepdim = True)

        # Weighted sum of feature maps
        cam = (weights * self.feature_maps).sum(dim = 1)

        # Nonlinearity
        cam = torch.relu(cam)

        # Normalize heatmap to [0, 1]
        cam = cam.squeeze()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        cam = cam.detach()

        return cam.cpu().numpy()
    
    # Remove existing forward and backward hooks for a fresh start
    def remove(self):
        self.fh.remove()
        self.bh.remove()