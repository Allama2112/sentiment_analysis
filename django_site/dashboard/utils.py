import os
import praw
import pandas as pd
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
        user_agent=USER_AGENT
    )

    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.hot(limit=limit):
        posts.append({
            "id": post.id,
            "title": post.title,
            "score": post.score,
            "url": post.url,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc
        })

    df = pd.DataFrame(posts)
    return df


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
    df = collect_reddit_posts(subreddit_name, limit=limit)

    if df.empty:
        raise ValueError(f"No posts found for subreddit '{subreddit_name}'")

    # Preprocess and analyze sentiment
    df["processed_title"] = df["title"].apply(preprocess_text)
    df["title_sentiment"] = df["processed_title"].apply(analyze_sentiment)

    # Summarize sentiment results
    summary = {
        "subreddit": subreddit_name,
        "num_posts": len(df),
        "avg_sentiment": round(df["title_sentiment"].mean(), 4),
        "top_positive_posts": df.sort_values(by="title_sentiment", ascending=False)["title"].head(5).tolist(),
        "top_negative_posts": df.sort_values(by="title_sentiment", ascending=True)["title"].head(5).tolist()
    }

    return df, summary
