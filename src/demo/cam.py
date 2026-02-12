import cv2

class Webcam:

    @staticmethod
    def run(process_frame, set_detect_every_n, toggle_xai=None, toggle_landmarks=None):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Webcam can not be opened")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output_frame = process_frame(frame)

            if output_frame is not None:
                cv2.imshow("webcam", output_frame)

            key = cv2.waitKey(10) & 0xFF

            if key in [ord(str(i)) for i in range(1, 10)]:
                n = int(chr(key))
                set_detect_every_n(n)

            if key == ord("h") and toggle_xai is not None:
                toggle_xai()

            if key == ord('k') and toggle_landmarks is not None:
                toggle_landmarks()

            if key == 27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
