from preprocessing.preparing_KDEF.filter_KDEF import KDEFfilter
from preprocessing.preparing_KDEF.sort_KDEF import sort_KDEF
from preprocessing.align_data import align_data


# entries:    Emotion    : ([positions]   , ratios)
parameters = {"Anger"    : (["S", "HL", "HR"], 0.5),
              "Disgust"  : (["S", "HL", "HR"], 1.0),
              "Fear"     : (["S", "HL", "HR"], 1.0),
              "Happiness": (["S", "HL", "HR"], 0),
              "Sadness"  : (["S", "HL", "HR"], 0.5),
              "Surprise" : (["S"], 0.5),
              }

def main():

    kdef_filter = KDEFfilter(parameters, 35) # 35=num. direc.

    sort_KDEF(kdef_filter)

    align_data(data = "KDEF")


if __name__ == "__main__":
    main()