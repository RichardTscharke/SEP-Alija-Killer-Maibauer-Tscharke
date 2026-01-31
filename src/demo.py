import cv2
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from models.ResNetLight import ResNetLightCNN
from models.ResNetLight2 import ResNetLightCNN2

from explain import GradCAM, overlay_gradcam

from preprocessing.detectors.retinaface import RetinaFaceDetector
from preprocessing.aligning.detect import detect_and_preprocess

detector = RetinaFaceDetector(device="cpu")

# ------------------------------
# constants
# ------------------------------

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
version = "models/ResNetLight_v2.pth"

# ------------------------------
# utility
# ------------------------------


def load_model(weight_path, device):
    model = ResNetLightCNN2(num_classes=6)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

'''''
def preprocess_frame(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (64, 64))
    frame = frame.transpose(2, 0, 1)
    frame = frame / 255.0
    tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(device)
'''

def preprocess_frame(frame, device):
    preprocessed = detect_and_preprocess(frame, detector)
    np_image = preprocessed["image"]          # HWC, BGR

    np_image = cv2.resize(np_image, (64, 64))
    np_image = np_image[:, :, ::-1]            # BGR → RGB
    np_image = np_image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(np_image)
    tensor = tensor.permute(2, 0, 1)            # CHW
    tensor = (tensor - 0.5) / 0.5               # match training
    tensor = tensor.unsqueeze(0)                # BCHW

    return tensor.to(device)
    

# ------------------------------
# main
# ------------------------------


def main(frame_stride):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(version, device)

    gradcam_conv1 = GradCAM(model, model.conv1)  # fine, local: edges, texture ...
    gradcam_conv2 = GradCAM(model, model.stage1.conv2)  # face features: mouth, nose ...
    gradcam_conv3 = GradCAM(model, model.stage2.conv2)  # whole faceparts: right faceside, mouth area ...
    gradcam_conv4 = GradCAM(model, model.stage3.conv2)  # very global, whole faces ...

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened")

    frame_id = 0
    last_emotion = "initializing"
    last_conf = 0.0
    last_overlay = None
    show_heatmap = True
    active_layer = 3

    cv2.namedWindow("Emotion Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Emotion Demo", 900, 700)

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

            if show_heatmap:
                if active_layer == 1:
                    heatmap = gradcam_conv1.generate(input_tensor, pred.item())
                elif active_layer == 2:
                    heatmap = gradcam_conv2.generate(input_tensor, pred.item())
                elif active_layer == 3:
                    heatmap = gradcam_conv3.generate(input_tensor, pred.item())
                else:
                    heatmap = gradcam_conv4.generate(input_tensor, pred.item())

                last_overlay = overlay_gradcam(frame, heatmap)
            else:
                last_overlay = frame.copy()

        if last_overlay is not None:
            frame = last_overlay

        label = f"{last_emotion} ({last_conf:.2f})"

        cv2.putText(
            frame, label, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            f"Layer: conv{active_layer}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Heatmap: {'ON' if show_heatmap else 'OFF'}",
            (180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Emotion Demo", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("1"):
            active_layer = 1
        elif key == ord("2"):
            active_layer = 2
        elif key == ord("3"):
            active_layer = 3
        elif key == ord("4"):
            active_layer = 4
        elif key == ord("h"):
            show_heatmap = not show_heatmap

        frame_id += 1

    gradcam_conv1.remove_hooks()
    gradcam_conv2.remove_hooks()
    gradcam_conv3.remove_hooks()
    gradcam_conv4.remove_hooks()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()
    main(args.stride)
