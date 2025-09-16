import praw
import os
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Loading the IDs from dotenv
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_SECRET_ID")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Setting up parameters
SUBREDDIT = "cowboys"
LIMIT = 100
SAVE_PATH = "data/raw"


def collect_reddit_posts(subreddit_name=SUBREDDIT, limit=LIMIT):
    """Fetch posts from subreddit -> return as data frame"""
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


def save_posts(df, subreddit_name):
    """Save posts to CSV"""
    os.makedirs(SAVE_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_PATH}/{subreddit_name}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} posts to {filename}")


if __name__ == "__main__":
    df = collect_reddit_posts()
    save_posts(df, SUBREDDIT)