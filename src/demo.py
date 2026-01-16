import cv2
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from model import CustomEmotionCNN
from explain import GradCAM, overlay_gradcam

# ------------------------------
# constants
# ------------------------------

EMOTIONS = ["suprise", "fear", "disgust", "happiness", "sad", "anger"]
version_raf = "raf_cnn_v5.pth"

# ------------------------------
# utility
# ------------------------------

def load_model(weight_path, device):
    model = CustomEmotionCNN(num_classes=6)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_frame(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (64, 64))
    frame = frame.transpose(2, 0, 1)
    frame = frame / 255.0
    tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

# ------------------------------
# main
# ------------------------------

def main(frame_stride):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(version_raf, device)
    gradcam = GradCAM(model, model.conv3)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened")

    frame_id = 0
    last_emotion = "initializing"
    last_conf = 0.0
    last_heatmap = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_stride == 0:
            input_tensor = preprocess_frame(frame, device)

            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)

            conf, pred = torch.max(probs, dim=1)
            last_conf = conf.item()
            last_emotion = EMOTIONS[pred.item()]

            heatmap = gradcam.generate(input_tensor, pred.item())
            last_heatmap = heatmap

        if last_heatmap is not None:
            frame = overlay_gradcam(frame, last_heatmap)

        label = f"{last_emotion} ({last_conf:.2f})"
        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.imshow("Emotion Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    gradcam.remove_hooks()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    main(args.stride)
