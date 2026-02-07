from preprocessing.preprocessing_RAF.preprocess_RAF import preprocess_RAF
from preprocessing.preprocessing_ExpW.preprocess_ExpW import preprocess_ExpW
from preprocessing.preprocessing_KDEF.preprocess_KDEF import preprocess_KDEF
from preprocessing.merge import merge

'''
This is the preprocessing interface of our project.

Within the filters numbers represent ratios per emotion class per dataset.
For the KDEF dataset there are also head positions to choose from:
-> S = Straight, HL/HR = Half Left/Right, FL/FR = Full Left/Right (we recommend to exclude fullsided in general)
The train/test/validate split is applied onto the filtered datasets and afterwards the splits are merged.
Total number or original images per dataset after filtering (Note that this number is lower after alignment):

RAF:
Surprise: 1619 | Fear: 355 | Disgust: 877 | Happiness: 5957 | Sadness: 2460 | Anger: 867
Final Entries: 12135

ExpW:
Surprise: 6104 | Fear: 962 | Disgust: 3250 | Happiness: 19350 | Sadness: 9090 | Anger: 3278
Final entries: 42034

KDEF:
105 images per emotion (35 persons posing S/HL/HR per emotion)
Final entries: 630

For custom data configurations adjust these:
'''
RAF_FILTER = {
    "Surprise":  0,
    "Fear":      0,
    "Disgust":   0,
    "Happiness": 0,
    "Sadness":   0,
    "Anger":     0,
}
EXPW_FILTER = {
    "Surprise":  1.0,
    "Fear":      1.0,
    "Disgust":   1.0,
    "Happiness": 1.0,
    "Sadness":   1.0,
    "Anger":     1.0,
}
KDEF_FILTER = {
    "Anger"    : (["S", "HL", "HR"], 0),
    "Disgust"  : (["S", "HL", "HR"], 0),
    "Fear"     : (["S", "HL", "HR"], 0),
    "Happiness": (["S", "HL", "HR"], 0),
    "Sadness"  : (["S", "HL", "HR"], 0),
    "Surprise" : (["S", "HL", "HR"], 0),
}
MERGE_SPLIT = {
    "RAF": {
        "train"   : 1.0,
        "test"    : 0.0,
        "validate": 0.0,
    },
    "ExpW": {
        "train"   : 1.0,
        "test"    : 0.0,
        "validate": 1.0,
    },
    "KDEF": {
        "train"   : 1.0,
        "test"    : 1.0,
        "validate": 1.0,
    },
}

# If you want to train on the original images flag this:
USE_ALIGNED = True

def main():

    preprocess_RAF(RAF_FILTER)

    preprocess_ExpW(EXPW_FILTER)

    preprocess_KDEF(KDEF_FILTER)

    merge(MERGE_SPLIT, USE_ALIGNED)

if __name__ == "__main__":
    main()