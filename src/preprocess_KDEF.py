from preprocessing.preparing_KDEF.filter_KDEF import KDEFfilter
from preprocessing.preparing_KDEF.sort_KDEF import sort_KDEF
from preprocessing.align_data import align_data

# head positions: "S" = Straight, "HL"/"HR" = Half Left/Right, "FL/FR" = Full Left/Right
# parameters:
# emotion   : (allowed head positions, sampling ratio)
KDEF_PARAMS = {
    "Anger"    : (["S", "HL", "HR"], 0.5),
    "Disgust"  : (["S", "HL", "HR"], 1.0),
    "Fear"     : (["S", "HL", "HR"], 1.0),
    "Happiness": (["S", "HL", "HR"], 0),
    "Sadness"  : (["S", "HL", "HR"], 0),
    "Surprise" : (["S", "HL", "HR"], 0),
}

# KDEF consists of 35 folders, each containing one person posing the emotions in different positions 
NUM_KDEF_DIRECTORIES = 35

def main():
    """
    Full preprocessing pipeline for KDEF:
    1) dataset-specific filtering
    2) emotion-wise sorting
    3) face alignment
    """

    # Dataset specific filtering since KDEFs emotion/position logic lies within the file names
    kdef_filter = KDEFfilter(KDEF_PARAMS, NUM_KDEF_DIRECTORIES)

    # Creates emotion classes for the original and aligned directories and fills the original one
    sort_KDEF(kdef_filter)
    
    # Aligns the data and fills the aligned directory
    align_data(data = "KDEF")


if __name__ == "__main__":
    main()