from django.shortcuts import render
from .utils import summarise_processed_data


def dashboard_view(request):
    # Get subreddit from query params
    subreddit = request.GET.get("subreddit", None)
    sentiment_df = None
    summary = None
    error = None

    if subreddit:
        try:
            sentiment_df, summary = summarise_processed_data(subreddit)
        except FileNotFoundError:
            error = f"No data found for subreddit '{subreddit}'"

    context = {
        "subreddit": subreddit,
        "summary": summary,
        "error": error
    }

    return render(request, "dashboard/dashboard.html", context)



