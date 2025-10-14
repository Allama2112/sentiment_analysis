import os
import praw
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from prawcore.exceptions import Forbidden, NotFound, Redirect
from django.core.cache import cache
from django.conf import settings

matplotlib.use('Agg')

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_SECRET_ID")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

LIMIT = 100


def collect_reddit_posts(subreddit_name, limit=LIMIT):
    """
    Fetch latest posts from given subreddit and returns as DataFrame with any error message.

    Args:
        subreddit_name (str): Name of subreddit to get posts from.
        limit (int, optional): Number of posts to analyze. Defaults to 100

    Returns:
        pandas.DataFrame: Data frame containing raw post data such as ID, title, score, URL, number of comments,
        error_msg (str): Specific error messages. If the retrieval is successful, returns None.

    Raises:
        praw.exceptions.Forbidden: If the subreddit is private or restricted.
        praw.exceptions.NotFounc: If the subreddit is not found.
    """

    cache_key = f"sentiment_{subreddit_name.lower()}_{limit}"
    cached_entry = cache.get(cache_key)

    if cached_entry:
        return cached_entry["data"], cached_entry["error"]

    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD")
        )

        subreddit = reddit.subreddit(subreddit_name)

        # Trying hot, then new, then top
        for listing_func in [
            lambda: subreddit.hot(limit=limit),
            lambda: subreddit.new(limit=limit),
            lambda: subreddit.top(time_filter="month", limit=limit),
        ]:
            posts = list(listing_func())
            if posts:
                df = pd.DataFrame([{
                    "id": post.id,
                    "title": post.title,
                    "score": post.score,
                    "url": post.url,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                } for post in posts])

                # Cache for X seconds (default: 10 min)
                cache.set(cache_key, {"data": df, "error": None}, getattr(settings, "CACHE_TTL", 600))
                return df, None

        # No posts found
        error_msg = f"No posts found on r/{subreddit_name}."
        cache.set(cache_key, {"data": pd.DataFrame(), "error": error_msg}, getattr(settings, "CACHE_TTL", 600))
        return pd.DataFrame(), error_msg
    except Forbidden:
        error_msg = f"Access to r/{subreddit_name} is forbidden."
        cache.set(cache_key, {"data": pd.DataFrame(), "error": error_msg}, getattr(settings, "CACHE_TTL", 600))
        return pd.DataFrame(), error_msg
    except (Redirect, NotFound):
        error_msg = f"r/{subreddit_name} does not exist."
        cache.set(cache_key, {"data": pd.DataFrame(), "error": error_msg}, getattr(settings, "CACHE_TTL", 600))
        return pd.DataFrame(), error_msg
    except Exception as e:
        error_msg = "An unknown error occurred. Please try again."
        return pd.DataFrame(), error_msg


def preprocess_text(text):
    """Clean and lemmatize text for sentiment analysis"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    filtered = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemmatized = [lemmatizer.lemmatize(t) for t in filtered]

    return ' '.join(lemmatized)


def analyze_sentiment(text):
    """Compute compound sentiment score (-1 to 1)"""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]


def apply_sentiment_analysis(raw_df):
    # Preprocess and analyze sentiment
    raw_df["processed_title"] = raw_df["title"].apply(preprocess_text)
    raw_df["title_sentiment"] = raw_df["processed_title"].apply(analyze_sentiment)

    return raw_df


def analyze_subreddit_sentiment(subreddit_name, limit=LIMIT):
    """Fetch, preprocess, and analyze sentiment for a subreddit"""

    # Fetch latest posts
    df, error_msg = collect_reddit_posts(subreddit_name, limit=limit)

    if error_msg:
        result = {
            "success": False,
            "error": error_msg,
            "data": None,
            "summary": None
        }
        return result

    analyzed_df = apply_sentiment_analysis(df)

    # Summarize sentiment results
    summary = {
        "subreddit": subreddit_name,
        "num_posts": len(analyzed_df),
        "avg_sentiment": round(df.loc[analyzed_df["title_sentiment"] != 0, "title_sentiment"].mean(), 4),
        "top_positive_posts": analyzed_df.sort_values(by="title_sentiment", ascending=False)["title"].head(5).tolist(),
        "top_negative_posts": analyzed_df.sort_values(by="title_sentiment", ascending=True)["title"].head(5).tolist()
    }

    result = {
        "success": True,
        "error": None,
        "data": analyzed_df,
        "summary": summary
    }

    # Cache the result for 10 minutes
    return result


# TODO: Call this function and display the graph in the dashboard view
def visualize_sentiment(subreddit_name):
    # Collecting and analyzing the data
    raw_df, err_msg = collect_reddit_posts(subreddit_name)

    # If the reddit posts could not be collected
    if err_msg:
        result = {
            "plot": None,
            "error_msg": err_msg
        }
        return err_msg

    # Apply sentiment analysis
    sentiment_df = apply_sentiment_analysis(raw_df)

    # Indexing the data frame
    sentiment_df["post_number"] = sentiment_df.index + 1

    # Creating a scatter plot of the last 100 posts
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x=sentiment_df["post_number"], y=sentiment_df["title_sentiment"])
    ax.set_xticks(range(1, len(sentiment_df)))
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("Post Number")
    ax.set_ylabel("Sentiment")
    ax.set_title(f"Sentiment of the Past {len(sentiment_df)} Posts on r/{subreddit_name}")

    # Setting the buffer
    buf = io.BytesIO()
    plt.tight_layout()

    # Saving the figure
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Storing the plot in base 64
    base64_plot = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    result = {
        "plot": base64_plot,
        "error_msg": None
    }
    return result
