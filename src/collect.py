from dotenv import load_dotenv
import os

load_dotenv()

# Loading the IDs from dotenv
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_SECRET_ID")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

