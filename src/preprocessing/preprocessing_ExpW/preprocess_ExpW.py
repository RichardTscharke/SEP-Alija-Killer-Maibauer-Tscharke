from .determine_and_filter_labels import determine_and_filter
from preprocessing.sort_data import sort_data
from preprocessing.align_data import align_data

def preprocess_ExpW(EXPW_FILTER):
    """
    Full preprocessing pipeline for ExpW:
    1) dataset-specific determination of single face and filtering
    2) emotion-wise sorting
    3) face alignment
    """

    # ExpW contains images with multiple faces. We extract the face by highest confidence labels within the label file
    determine_and_filter(EXPW_FILTER)

    # Creates emotion classes for the original and aligned directories and fills the original one
    sort_data(data  = "ExpW")

    # Aligns the data and fills the aligned directory
    align_data(data = "ExpW")
