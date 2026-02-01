import sys
import cv2
from preprocessing.detectors.retinaface import RetinaFaceDetector
from preprocessing.aligning.detect import detect_and_preprocess

image_path = "/Users/richardachtnull/Desktop/data2/RAF_raw/Image/original/test_0001.jpg"


def main():
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = RetinaFaceDetector(device= "cpu")

    result = detect_and_preprocess(
        image,
        detector,
        vis=True
    )

if __name__ == "__main__":

    main()
