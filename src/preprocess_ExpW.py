from preprocessing.preparing_ExpW.determine_and_filter_labels import determine_and_filter
from preprocessing.sort_data import sort_data
from preprocessing.align_data import align_data

# Adjust these for custom class configurations
FILTER_RATIOS = {
    "Surprise":  1.0,
    "Fear":      1.0,
    "Disgust":   1.0,
    "Happiness": 0.3,
    "Sadness":   1.0,
    "Anger":     1.0,
}

def main():
    """
    Full preprocessing pipeline for ExpW:
    1) dataset-specific determination of single face and filtering
    2) emotion-wise sorting
    3) face alignment
    """

    # ExpW contains images with multiple faces. We extract the face by highest confidence labels within the label file
    determine_and_filter(FILTER_RATIOS)

    # Creates emotion classes for the original and aligned directories and fills the original one
    sort_data(data  = "ExpW")

    # Aligns the data and fills the aligned directory
    align_data(data = "ExpW")

if __name__ == "__main__":
    main()