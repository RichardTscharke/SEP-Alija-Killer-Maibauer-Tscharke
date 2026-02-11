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

            output_frame = process_frame(frame)

            if output_frame is not None:
                cv2.imshow("webcam", output_frame)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
