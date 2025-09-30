import os
import sys
import pandas as pd

from utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


def get_subreddit_csv(subreddit_name):
    files = list(RAW_DATA_DIR.glob(f"{subreddit_name}_*.csv"))

    if not files:
        raise FileNotFoundError("No CSVs at this directory")
    return max(files, key=lambda f: f.stat().st_mtime)


def get_processed_data(subreddit_name):
    # Getting the list of files under that name
    files = list(PROCESSED_DATA_DIR.glob(f"{subreddit_name}.csv"))

    # If there are no files
    if not files:
        raise FileNotFoundError("No processed data at this directory")
    if len(files) > 1:
        # If there are more than one files for that subreddit, return the first one
        print("There are more than one file for this subreddit at this directory, returning the first one")
        return files[0]
    # Otherwise return the file path
    return files[0]


def summarise_processed_data(subreddit):
    file_name = get_processed_data(subreddit)

    # Reading it in as a csv
    sentiment_df = pd.read_csv(file_name)

    summary = {
        "file_name": subreddit,
        "num_posts": len(sentiment_df),
        "avg_sentiment": sentiment_df[sentiment_df["title_sentiment"] != 0]["title_sentiment"].mean().round(4),
        "top_posts": sentiment_df["title"].head().tolist()
    }

    return sentiment_df, summary
