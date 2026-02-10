import cv2
from pathlib import Path
from preprocessing.detectors.retinaface import RetinaFaceDetector
from preprocessing.aligning.detect import detect_and_preprocess

image_path = Path("/Users/richardachtnull/Desktop/IMG_0477.jpg")


def main():
    '''
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = RetinaFaceDetector(device= "cpu")

    result = detect_and_preprocess(
        image,
        detector,
        vis=True
    )
    '''

    image = cv2.imread(image_path)

    detector = RetinaFaceDetector(device="cpu")

    sample = detect_and_preprocess(image, detector, vis=True)

    print(f"Crop Offset: {sample['meta']['crop_offset']}")
    print(f"M: {sample['meta']['affine_M']}")
    print(f"M^-1: {sample['meta']['affine_M_inv']}")

if __name__ == "__main__":

    main()
