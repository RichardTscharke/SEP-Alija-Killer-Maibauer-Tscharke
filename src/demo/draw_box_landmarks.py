import cv2

def draw_box_landmarks(frame, original):
    x, y, w, h = original["box"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for _, (lx, ly) in original["landmarks"].items():
        cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
