import cv2
import numpy as np

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

def draw_emotion_probs(frame, probs, origin=(10, 30), line_height=40, font_scale=1.0):

    if hasattr(probs, "detach"):
        probs = probs.detach().cpu().numpy()
    probs = np.asarray(probs)

    x, y0 = origin
    max_idx = int(np.argmax(probs))

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    colon_x = x + 160   # feste Spalte für :
    value_x = x + 200   # feste Spalte für Prozent

    for i, (emotion, p) in enumerate(zip(EMOTIONS, probs)):
        y = y0 + i * line_height
        color = (0, 255, 0) if i == max_idx else (0, 50, 255)

        # Emotion
        cv2.putText(frame, emotion, (x, y),
                    font, font_scale, color, thickness, cv2.LINE_AA)

        # Doppelpunkt
        cv2.putText(frame, ":", (colon_x, y),
                    font, font_scale, color, thickness, cv2.LINE_AA)

        # Wert
        value_text = f"{p * 100:5.1f}%"
        cv2.putText(frame, value_text, (value_x, y),
                    font, font_scale, color, thickness, cv2.LINE_AA)
