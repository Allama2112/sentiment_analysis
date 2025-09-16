import pandas as pd

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


def get_specific_subreddit_raw_data(subreddit_name):
    files = list(RAW_DATA_DIR.glob(f"{subreddit_name}_*.csv"))

    if not files:
        raise FileNotFoundError("No CSVs at this directory")
    return max(files, key=lambda f: f.stat().st_mtime)


def preprocess(subreddit_name):
    file_name = get_specific_subreddit_raw_data(subreddit_name)

    df = pd.read_csv(file_name)

    print(df.head())


if __name__ == "__main__":
    preprocess("cowboys")
