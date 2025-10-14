from django.shortcuts import render
from .utils import analyze_subreddit_sentiment, visualize_sentiment
from django.core.cache import cache


def home(request):
    # Home view
    return render(request, "dashboard/home.html")


def dashboard_view(request):
    # Get subreddit from query params
    subreddit_name = request.GET.get("subreddit", "").strip()

    # Will refresh cache on request
    refresh = request.GET.get("refresh") == "true"
    context = {}

    # If the subreddit is found
    if subreddit_name:
        # Get the cache key
        cache_key = f"sentiment_{subreddit_name.lower()}_100"

        # Delete the cache upon user request
        if refresh:
            cache.delete(cache_key)

        # Call the function to get the analysis --> This will check cache for stored data
        result = analyze_subreddit_sentiment(subreddit_name)

        # If we cannot retrieve data for that subreddit
        if not result["success"]:
            context["error"] = result["error"]
        else:
            context["summary"] = result["summary"]

        sentiment_plot_result = visualize_sentiment(subreddit_name)

        if sentiment_plot_result["error_msg"] is not None:
            context["error"] = sentiment_plot_result["error_msg"]
        else:
            context["sentiment_plot"] = sentiment_plot_result["plot"]

    return render(request, "dashboard/dashboard.html", context)
