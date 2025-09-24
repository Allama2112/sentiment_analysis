import pandas as pd
import matplotlib.pyplot as plt
from src.utils.paths import PROCESSED_DATA_DIR


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


def plot_sentiment(subreddit):
    # Getting the file path of the subreddit
    file_name = get_processed_data(subreddit)

    # Reading it in as a csv
    sentiment_df = pd.read_csv(file_name)

    # TODO: Put in a col with numbers 1-100 for easier plotting on x-axis

    # Creating a scatter plot of the last 100 posts
    plt.scatter(x=sentiment_df["id"], y=sentiment_df["title_sentiment"])
    plt.show()


if __name__ == "__main__":
    plot_sentiment("cowboys")
