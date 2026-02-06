from .filter_KDEF import KDEFfilter
from .sort_KDEF import sort_KDEF
from preprocessing.align_data import align_data

# KDEF consists of 35 folders, each containing one person posing the emotions in different positions 
NUM_KDEF_DIRECTORIES = 35

def preprocess_KDEF(KDEF_FILTER):
    """
    Full preprocessing pipeline for KDEF:
    1) dataset-specific filtering
    2) emotion-wise sorting
    3) face alignment
    """

    # Dataset specific filtering since KDEFs emotion/position logic lies within the file names
    kdef_filter = KDEFfilter(KDEF_FILTER, NUM_KDEF_DIRECTORIES)

    # Creates emotion classes for the original and aligned directories and fills the original one
    sort_KDEF(kdef_filter)
    
    # Aligns the data and fills the aligned directory
    align_data(data = "KDEF")
