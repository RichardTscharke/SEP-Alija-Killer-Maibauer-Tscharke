import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from model import CustomEmotionCNN

# -------------------------------------------------
# Constants
# -------------------------------------------------

EMOTIONS = ["suprise", "fear", "disgust", "happiness", "sad", "anger"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# -------------------------------------------------
# Model utilities
# -------------------------------------------------

def load_cnn_model(weight_path, device):
    model = CustomEmotionCNN(num_classes=6)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_frame(frame, device):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model input
    frame = cv2.resize(frame, (64, 64))

    frame = frame.transpose(2, 0, 1)

    frame = frame / 255.0

    tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device)

    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
    std = IMAGENET_STD.view(1, 3, 1, 1).to(device)
    tensor = (tensor - mean) / std

    return tensor

def cnn_inference(model, frame, device):
    tensor = preprocess_frame(frame, device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    conf, pred = torch.max(probs, dim=1)
    emotion = EMOTIONS[pred.item()]
    confidence = conf.item()

    return emotion, confidence

# -------------------------------------------------
# Dummy Explainability 
# -------------------------------------------------

def dummy_explainability(frame):
    heatmap = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.circle(
        heatmap,
        (frame.shape[1] // 2, frame.shape[0] // 2),
        30,
        255,
        -1
    )

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def overlay_heatmap(frame, heatmap, alpha=0.4):
    return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

# -------------------------------------------------
# Main demo loop
# -------------------------------------------------

def main(frame_stride):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = load_cnn_model("raf_cnn_v5.pth", device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam konnte nicht geöffnet werden")

    frame_id = 0
    last_emotion = "initializing"
    last_conf = 0.0
    last_heatmap = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze only every n-th frame
        if frame_id % frame_stride == 0:
            last_emotion, last_conf = cnn_inference(cnn_model, frame, device)
            last_heatmap = dummy_explainability(frame)

        if last_heatmap is not None:
            frame = overlay_heatmap(frame, last_heatmap)

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

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------
# Entry point
# -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Analyze every N-th frame"
    )
    args = parser.parse_args()

    main(args.stride)
