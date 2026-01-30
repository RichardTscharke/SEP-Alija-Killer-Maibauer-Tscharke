from preprocessing.preparing_ExpW.determine_and_filter_labels import determine_and_filter
#from preprocessing.preparing_ExpW.sort_and_crop_ExpW import sort_and_crop_ExpW
from preprocessing.align_data import align_data

def main():

    determine_and_filter(surprise_ratio = 0.1,
                        fear_ratio      = 0.1,
                        disgust_ratio   = 0.1,
                        happiness_ratio = 0.1,
                        sadness_ratio   = 0.1,
                        anger_ratio     = 0.1)

    #sort_and_crop_KDEF

    #align_data(data = "RAF")


if __name__ == "__main__":
    main()