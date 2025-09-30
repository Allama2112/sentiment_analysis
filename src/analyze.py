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
        # If there are more than one files for that subreddit
        print("There are more than one file for this subreddit at this directory, returning the first one")
    # Return the first file
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


def plot_sentiment(subreddit):
    # Getting the file path of the subreddit
    file_name = get_processed_data(subreddit)

    # Reading it in as a csv
    sentiment_df = pd.read_csv(file_name)

    # Getting the average sentiment of the last 100 posts
    avg_sentiment = sentiment_df[sentiment_df["title_sentiment"] != 0]["title_sentiment"].mean().round(4)

    # Indexing the data frame
    sentiment_df["post_number"] = sentiment_df.index + 1

    # TODO: Find a way to link the posts based on clicking on the points

    # Creating a scatter plot of the last 100 posts
    plt.scatter(x=sentiment_df["post_number"], y=sentiment_df["title_sentiment"])
    plt.xticks(range(1, len(sentiment_df)), rotation=90)
    plt.xlabel("Post Number")
    plt.ylabel("Sentiment")
    plt.suptitle(f"Sentiment of the Past {len(sentiment_df)} Posts on r/{subreddit}")
    plt.title(f"Average Sentiment of Non-Neutral Posts: {avg_sentiment}", fontsize=10)
    plt.show()


if __name__ == "__main__":
    plot_sentiment("cowboys")
