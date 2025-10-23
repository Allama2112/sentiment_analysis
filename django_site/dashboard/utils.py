import os
import praw
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import plotly.express as px

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


# TODO: Figure out why this actually has 100 posts instead of 105
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


def visualize_sentiment(subreddit_name, limit=LIMIT):
    # Collecting and analyzing the data
    raw_df, err_msg = collect_reddit_posts(subreddit_name, limit)

    # If the reddit posts could not be collected
    if err_msg:
        result = {
            "plot": None,
            "error_msg": err_msg
        }
        return result

    # Apply sentiment analysis
    sentiment_df = apply_sentiment_analysis(raw_df)

    # Indexing the data frame
    sentiment_df["post_number"] = sentiment_df.index + 1

    # Information to be displayed on hover
    sentiment_df["hover_text"] = sentiment_df["hover_text"] = (sentiment_df["title"]
                                                               + "<br>Sentiment: " + sentiment_df[
                                                                   "title_sentiment"].round(2).astype(str)
                                                               + "<br>Score: " + sentiment_df["score"].astype(str)
                                                               )

    # Creating the interactable figure
    interactable_fig = px.scatter(
        sentiment_df,
        x="post_number",
        y="title_sentiment",
        hover_name="title",
        hover_data={"score": True, "title_sentiment": True, "hover_text": False},
        text=None,
        color="title_sentiment",
        color_continuous_scale="RdBu"
    )

    # Creating the on click action
    interactable_fig.update_traces(
        customdata=sentiment_df["url"],
        hovertemplate="%{hovertext}<extra></extra>",
    )

    # JavaScript click handler
    interactable_fig.update_layout(
        title="Interactive Sentiment Handler"
    )

    graph_html = interactable_fig.to_html(full_html=False, include_plotlyjs='cdn', div_id="sentiment_plot")
    result = {
        "plot": graph_html,
        "error_msg": err_msg
    }
    return result
