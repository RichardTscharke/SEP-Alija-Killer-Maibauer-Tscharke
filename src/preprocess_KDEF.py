from preprocessing.preparing_KDEF.filter_KDEF import KDEFfilter
from preprocessing.preparing_KDEF.sort_KDEF import sort_KDEF
from preprocessing.align_data import align_data


# paramater entry: Emotion: ([positions], ratio of used images)

parameters = {"Anger":      (["S", "HL", "HR"], 1.0),
              "Disgust":    (["S", "HL", "HR"], 1.0),
              "Fear":       (["S", "HL", "HR"], 1.0),
              "Happiness":  (["S", "HL", "HR"], 0.5),
              "Sadness":    (["S", "HL", "HR"], 1.0),
              "Surprise":   (["S"], 1.0),
              }

def main():

    kdef_filter = KDEFfilter(parameters, 35)

    sort_KDEF(kdef_filter)

    align_data(data = "KDEF")


if __name__ == "__main__":
    main()