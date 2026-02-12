import cv2

BOX_COLOR      = (0, 255, 0)
LANDMARK_COLOR = (0, 0, 255)

def set_colors(box_color, landmarks_color):
    global BOX_COLOR, LANDMARK_COLOR
    BOX_COLOR = box_color
    LANDMARK_COLOR = landmarks_color


def draw_box_landmarks(frame, original):
    x, y, w, h = original["box"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)

    for _, (lx, ly) in original["landmarks"].items():
        cv2.circle(frame, (lx, ly), 2, LANDMARK_COLOR, -1)

    return frame
