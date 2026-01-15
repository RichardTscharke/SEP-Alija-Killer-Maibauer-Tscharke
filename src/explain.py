import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import CustomEmotionCNN

MODEL_PATH = "models/raf_cnn_v1.pth"  # Make sure this is the latest trained model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

def explain_image(folder_path):

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    img = Image.open(folder_path)
    x = transform(img).unsqueeze(0).to(DEVICE)

    model = CustomEmotionCNN(num_classes = 6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    feature_maps = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]


    fh = model.conv3.register_forward_hook(forward_hook)             # change 2 to 3 or vice versa vor different results
    bh = model.conv3.register_full_backward_hook(backward_hook)      # change 2 to 3 or vice versa vor different results

    logits = model(x)
    pred_class = logits.argmax(dim=1).item()

    logits[0, pred_class].backward()

    fh.remove()
    bh.remove()

    pooled_gradients = gradients.mean(dim=(0, 2, 3))

    for i in range(feature_maps.shape[1]):
        feature_maps[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap = heatmap.detach().cpu().numpy()

    heatmap_scaled = cv2.resize(heatmap, (64, 64))

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_scaled),cv2.COLORMAP_JET)

    img_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.uint8(255 * img_np)

    heatmap_color[heatmap_scaled < 0.2] = 0
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    plt.figure(figsize=(4,4))
    plt.imshow(overlay)
    #plt.axis("off")
    plt.show()

if __name__ == "__main__":

    default_folder = "/Users/richardachtnull/Desktop/aligned/test_0003_aligned.jpg"  # change local path to 64x64 picture

    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = default_folder

    explain_image(folder_path)