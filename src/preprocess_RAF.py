from preprocessing.preparing_RAF.filter_RAF import filter
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
    Full preprocessing pipeline for RAF:
    1) dataset-specific filtering
    2) emotion-wise sorting
    3) face alignment
    """

    # Reduces class imbalance by downsampling selected emotions
    filter(FILTER_RATIOS)

    # Creates emotion classes for the original and aligned directories and fills the original one
    sort_data(data  = "RAF")

    # Aligns the data and fills the aligned directory
    align_data(data = "RAF")


if __name__ == "__main__":
    main()
