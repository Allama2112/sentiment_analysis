from django.shortcuts import render
from .utils import analyze_subreddit_sentiment


def home(request):
    # Home view
    return render(request, "dashboard/home.html")


def dashboard_view(request):
    # Get subreddit from query params
    subreddit = request.GET.get("subreddit", None)
    sentiment_df = None
    summary = None
    error = None

    if subreddit:
        try:
            sentiment_df, summary = analyze_subreddit_sentiment(subreddit)
        except FileNotFoundError:
            error = f"No data found for subreddit '{subreddit}'"

    context = {
        "subreddit": subreddit,
        "summary": summary,
        "error": error
    }

    return render(request, "dashboard/dashboard.html", context)
