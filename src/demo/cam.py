import cv2

class Webcam:
    '''
    Demo webcam class for frame per frame processing and key logic.
    '''
    @staticmethod
    def run(process_frame, set_detect_every_n, toggle_xai=None, toggle_landmarks=None):
        '''
        Starts webcam stream and process frames in real time.
        Supports keyboard controls for detection rate and overlays.
        '''

        # Open default webcam
        cap = cv2.VideoCapture(0)

        # Ensure webcam opened successfully
        if not cap.isOpened():
            raise RuntimeError("Webcam can not be opened")

        # Main streaming loop
        while True:

            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply frame processing pipeline injected by fer_controller.py
            output_frame = process_frame(frame)
            
            # Display processed frame is available
            if output_frame is not None:
                cv2.imshow("webcam", output_frame)

            # Read pressed key with 10ms delay
            key = cv2.waitKey(10) & 0xFF

            # Number keys (1-9): adjust detection frequency
            if key in [ord(str(i)) for i in range(1, 10)]:
                n = int(chr(key))
                set_detect_every_n(n)

            # 'h' key: toggle Grad-CAM overlay
            if key == ord("h") and toggle_xai is not None:
                toggle_xai()

            # 'k' key: toggle bund box and landmark overlay
            if key == ord('k') and toggle_landmarks is not None:
                toggle_landmarks()

            # ESC or 'q': exit loop
            if key == 27 or key == ord('q'):
                break
        
        # Release webcam and close window
        cap.release()
        cv2.destroyAllWindows()
