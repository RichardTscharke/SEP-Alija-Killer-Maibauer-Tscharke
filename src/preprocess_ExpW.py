from preprocessing.preparing_ExpW.determine_and_filter_labels import determine_and_filter
from preprocessing.sort_data import sort_data
from preprocessing.align_data import align_data

def main():

    determine_and_filter(surprise_ratio = 0.1,
                        fear_ratio      = 0.1,
                        disgust_ratio   = 0.1,
                        happiness_ratio = 0.1,
                        sadness_ratio   = 0.1,
                        anger_ratio     = 0.1)

    sort_data(data = "ExpW") # The data flag creates a crop of the determined face since ExpW contains multifaced images.

    align_data(data = "ExpW")

if __name__ == "__main__":
    main()