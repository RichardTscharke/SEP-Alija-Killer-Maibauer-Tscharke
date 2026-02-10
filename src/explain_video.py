import torch
from pathlib import Path
from tqdm import tqdm

from preprocessing.detectors.retinaface import RetinaFaceDetector
from explaining.explain_utils import resolve_model_and_layer
from explaining.video_utils import open_video, create_video_writer
from explaining.process_frame import process_frame


INPUT_PATH = Path("/Users/richardachtnull/IMG_0522.MOV")

OUTPUT_PATH = Path("/Users/richardachtnull/IMG_0522_explained.MOV")

MODEL_PATH = Path("models/ResNetLight2_v0.pth")

TARGET_LAYER = "stage3"

THRESHOLD = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(input_path):

    model, target_layer = resolve_model_and_layer(MODEL_PATH, TARGET_LAYER, DEVICE)

    print(f"[INFO] Model loaded: {MODEL_PATH}")
    print(f"[INFO] Target Layer: {TARGET_LAYER}")

    detector = RetinaFaceDetector(device="cpu")
    print("[INFO] RetinaFace Detector initialized.")

    print(f"[INFO] Opening video: {input_path}")
    cap, fps, total_frames = open_video(input_path)

    print(f"[INFO] Frames Per Second: {fps}")
    print(f"[INFO] Total Frames: {total_frames}")

    writer = None

    for frame_idx in tqdm(range(total_frames), desc="Explaining video", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        overlayed = process_frame(
            frame,
            detector,
            model,
            target_layer,
            DEVICE,
            THRESHOLD
            )

        if writer is None:
            h, w = overlayed.shape[:2]
            writer = create_video_writer(
                output_path=OUTPUT_PATH,
                fps=fps,
                frame_size=(w, h)
            )

        writer.write(overlayed)

    cap.release()
    if writer is not None:
        writer.release()

    print("[INFO] Finished video explanation")
    print("[INFO] Saved to:", OUTPUT_PATH)



if __name__ == "__main__":
    main(INPUT_PATH)