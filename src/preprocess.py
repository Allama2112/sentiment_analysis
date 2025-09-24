import pandas as pd
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


def get_subreddit_csv(subreddit_name):
    files = list(RAW_DATA_DIR.glob(f"{subreddit_name}_*.csv"))

    if not files:
        raise FileNotFoundError("No CSVs at this directory")
    return max(files, key=lambda f: f.stat().st_mtime)


def apply_preprocessing(subreddit_name):
    # Reading in the subreddit data
    file_name = get_subreddit_csv(subreddit_name)
    df = pd.read_csv(file_name)

    # Apply preprocessing
    df["processed_title"] = df["title"].apply(preprocess)
    return df


def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Removing stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join back to string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text


def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(text)
    return scores["compound"]


def save_processed_data(processed_data):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    filename = f"{PROCESSED_DATA_DIR}/cowboys.csv"
    processed_data.to_csv(filename, index=False)
    print(f"Saved processed data to {filename}")


if __name__ == "__main__":
    processed_df = apply_preprocessing("cowboys")
    processed_df["title_sentiment"] = processed_df["processed_title"].apply(get_sentiment)
    save_processed_data(processed_df)
