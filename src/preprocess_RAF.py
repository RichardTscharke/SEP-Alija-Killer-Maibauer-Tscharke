from preprocessing.preparing_RAF.filter_RAF import filter
from preprocessing.sort_data import sort_data
from preprocessing.align_data import align_data

def main():

    filter(suprise_ratio   = 1,
           fear_ratio      = 1,
           disgust_ratio   = 1,
           happiness_ratio = 0.3,
           sadness_ratio   = 1,
           anger_ratio     = 1)

    sort_data(data  = "RAF")

    align_data(data = "RAF")


if __name__ == "__main__":
    main()
