import torch
import os
import sys
import pandas as pd
from PIL import Image
from torchvision import transforms
from model import CustomEmotionCNN

# Configurations
MODEL_PATH = "models/raf_cnn_v5.pth"  # Make sure this is the latest trained model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output settings
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "predictions.csv"

# Class labels (Must match training order)
CLASSES = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]


def generate_csv(input_folder):

    # 1. Setup output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output folder: {OUTPUT_DIR}")

    output_csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print("\n" + "=" * 30)
    print(f"🚀 STARTING INFERENCE")
    print(f"🧠 Model:  {MODEL_PATH}")
    print(f"📂 Input:  {input_folder}")
    print(f"💾 Output: {output_csv_path}")
    print("=" * 30 + "\n")

    # 2. Load model
    model = CustomEmotionCNN(num_classes=len(CLASSES))
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"❌ Error: Model weights not found at {MODEL_PATH}")
        return

    model.to(DEVICE)
    model.eval()

    # 3. Define transforms (must match training config)
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    results = []
    image_count = 0

    print("Starting inference...")

    # 4. Loop through input directory
    with torch.no_grad():
        for root, _, files in os.walk(input_folder):
            for file in files:
                # Filter for common image formats
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):

                    img_path = os.path.join(root, file)

                    try:
                        # Prepare image
                        image = Image.open(img_path).convert("RGB")
                        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

                        # Forward pass
                        outputs = model(input_tensor)

                        # Get probabilities
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        probs_list = probs.cpu().numpy()[0]

                        # Build CSV row
                        # Structure: filename, anger_score, disgust_score, ...
                        row = {"filename": file}
                        for i, emotion in enumerate(CLASSES):
                            row[emotion] = probs_list[i]

                        # Add final classification
                        predicted_idx = torch.argmax(probs, 1).item()
                        row["prediction"] = CLASSES[predicted_idx]

                        results.append(row)
                        image_count += 1

                        if image_count % 500 == 0:
                            print(f"... processed {image_count} images")

                    except Exception as e:
                        print(f"⚠️ Skipped {file}: {e}")

    # 5. Save to CSV
    if results:
        df = pd.DataFrame(results)

        # Reorder columns: filename first
        cols = ["filename"] + CLASSES + ["prediction"]
        df = df[cols]

        df.to_csv(output_csv_path, index=False)
        print(f"✅ Success! Saved {len(results)} predictions to {output_csv_path}")
    else:
        print("⚠️ No images found in input folder.")


if __name__ == "__main__":
    # Default path on server
    default_folder = "data/RAF_aligned_processed/test"

    # Allow command line argument for folder path
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = default_folder

    generate_csv(folder_path)
