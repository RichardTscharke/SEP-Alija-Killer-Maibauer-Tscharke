from preprocessing.preparing_RAF.filter_RAF import filter
from preprocessing.preparing_RAF.sort_RAF import sort_RAF
from preprocessing.align_data import align_data

def main():

    filter(happiness_ratio = 0.35)

    sort_RAF()

    align_data(data = "RAF")


if __name__ == "__main__":
    main()
