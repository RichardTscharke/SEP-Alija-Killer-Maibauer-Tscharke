from mtcnn import MTCNN
from preprocessing.pipeline import preprocess_image
from preprocessing.fallback import fallback
import numpy as np
from PIL import Image



detector = MTCNN(device="CPU:0")

#image = Image.open("tests/images/ivan.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/Bild.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/richard.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/richard3.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/mama.jpg")
image = Image.open("/Users/richardachtnull/Desktop/RAF_raw/Image/original/test_0001.jpg")

image_array = np.array(image)

#sample = preprocess_image(image_array, detector, debug = True)
fallback(image_array, debug = True)
