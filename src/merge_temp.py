from preprocessing.merge import merge

MERGE_SPLIT = {
    "RAF": {
        "train"   : 0.7,
        "test"    : 0.15,
        "validate": 0.15,
    },
    "ExpW": {
        "train"   : 0.7,
        "test"    : 0.15,
        "validate": 1.15,
    },
    "KDEF": {
        "train"   : 0.7,
        "test"    : 0.15,
        "validate": 0.15,
    },
}

USE_ALIGNED = True

def main():
    merge(MERGE_SPLIT, USE_ALIGNED)

if __name__ == "__main__":
    main()