from .preprocessing.preparing_RAF.filter import filter
from .preprocessing.preparing_RAF.sort_original import sort_original
from .preprocessing.preparing_RAF.sort_aligned import sort_aligned

def main():

    filter(happiness_ratio = 0.5)

    sort_original()

    sort_aligned()


if __name__ == "__main__":
    main()
