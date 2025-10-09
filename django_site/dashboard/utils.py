import os
import praw
import pandas as pd
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from prawcore.exceptions import Forbidden

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_SECRET_ID")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

LIMIT = 100


def collect_reddit_posts(subreddit_name, limit=LIMIT):
    """Fetch latest posts from subreddit and return as DataFrame"""
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD")
    )

    subreddit = reddit.subreddit(subreddit_name)

    posts = list(subreddit.hot(limit=limit))
    if not posts:
        posts = list(subreddit.new(limit=limit))
    if not posts:
        posts = list(subreddit.top(time_filter='month', limit=limit))
    if not posts:
        return pd.DataFrame()

    # Trying hot, then new, then top
    for listing in [subreddit.hot, subreddit.new, lambda limit: subreddit.top(time_filter="month")]:
        posts = list(listing(limit=limit))
        if posts:
            return pd.DataFrame([{
                "id": post.id,
                "title": post.title,
                "score": post.score,
                "url": post.url,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc
            } for post in posts])

    # No posts found
    return pd.DataFrame()


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


def analyze_subreddit_sentiment(subreddit_name, limit=LIMIT):
    """Fetch, preprocess, and analyze sentiment for a subreddit"""

    # Fetch latest posts
    try:
        df = collect_reddit_posts(subreddit_name, limit=limit)
    except Forbidden:
        return {
            "success": False,
            "error": f"Access to r/{subreddit_name} is restricted or forbidden.",
            "data": None,
            "summary": None
        }

    if df.empty:
        return {
            "success": False,
            "error": f"No posts found on r/{subreddit_name}. The subreddit may be restricted or empty.",
            "data": None,
            "summary": None
        }

    # Preprocess and analyze sentiment
    df["processed_title"] = df["title"].apply(preprocess_text)
    df["title_sentiment"] = df["processed_title"].apply(analyze_sentiment)

    # Summarize sentiment results
    summary = {
        "subreddit": subreddit_name,
        "num_posts": len(df),
        "avg_sentiment": round(df.loc[df["title_sentiment"] != 0, "title_sentiment"].mean(), 4),
        "top_positive_posts": df.sort_values(by="title_sentiment", ascending=False)["title"].head(5).tolist(),
        "top_negative_posts": df.sort_values(by="title_sentiment", ascending=True)["title"].head(5).tolist()
    }

    return {
        "success": True,
        "error": None,
        "data": df,
        "summary": summary
    }
