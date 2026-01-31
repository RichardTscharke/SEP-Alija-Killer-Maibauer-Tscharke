import cv2

class Webcam:

    @staticmethod
    def run(process_frame):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Webcam can not be opened")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = process_frame(frame)

            if output is not None:
                for name, img in output.items():
                    cv2.imshow(name, img)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
