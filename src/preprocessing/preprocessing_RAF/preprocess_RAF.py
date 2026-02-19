from src.preprocessing.preprocessing_RAF.filter_RAF import filter
from src.preprocessing.sort_data import sort_data
from src.preprocessing.align_data import align_data


def preprocess_RAF(RAF_FILTER):
    """
    Full preprocessing pipeline for RAF:
    1) dataset-specific filtering
    2) emotion-wise sorting
    3) face alignment
    """

    # Reduces class imbalance by downsampling selected emotions
    filter(RAF_FILTER)

    # Creates emotion classes for the original and aligned directories and fills the original one
    sort_data(data = "RAF")

    # Aligns the data and fills the aligned directory
    align_data(data = "RAF")
