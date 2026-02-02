from preprocessing.preparing_ExpW.determine_and_filter_labels import determine_and_filter
from preprocessing.sort_data import sort_data
from preprocessing.align_data import align_data

def main():

    determine_and_filter(surprise_ratio = 0.66,
                        fear_ratio      = 1.0,
                        disgust_ratio   = 1.0,
                        happiness_ratio = 0.25,
                        sadness_ratio   = 1.0,
                        anger_ratio     = 1.0)

    sort_data(data  = "ExpW") # Note: ExpW contains multifaced imgs.

    align_data(data = "ExpW")

if __name__ == "__main__":
    main()