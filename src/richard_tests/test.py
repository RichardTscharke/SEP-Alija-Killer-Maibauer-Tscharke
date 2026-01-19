from mtcnn import MTCNN
from preprocessing.pipeline import preprocess_image
from preprocessing.fallback import fallback
import numpy as np
from PIL import Image



detector = MTCNN()

#image = Image.open("tests/images/ivan.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/Bild.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/richard.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/richard3.jpg")
#image = Image.open("/Users/richardachtnull/Desktop/mama.jpg")
image = Image.open("/Users/richardachtnull/Desktop/data/KDEF/Image/KDEF_original_processed/Anger/train_66.jpg")

image_array = np.array(image)

sample = preprocess_image(image_array, detector, vis = True, debug = False)
#fallback(image_array, debug = True)
