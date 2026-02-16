import cv2

def open_video(input_path):
    '''
    opens video file and returns capture object, FPS and total frame count.
    '''
    # Create OpenCV video capture object
    cap = cv2.VideoCapture(str(input_path))

    # Ensure video was opened successfully
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    
    # Retrieve fps and total number of frames in vide
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, total_frames


def create_video_writer(
        output_path,
        fps,
        frame_size,
        codec="mp4v"
):
    '''
    Creates and returns an OpenCV VideoWriter.
    Takes the target path, frames per second, frame size and fourcc codec
    Returns cv2.VideoWriter
    '''
    # Create four-character codec code
    fourcc = cv2.VideoWriter_fourcc(*codec)

    # Initialize video writer
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size
    )

    # Ensure writer was created successfully
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")
    
    return writer
    