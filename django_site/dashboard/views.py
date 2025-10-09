from django.shortcuts import render
from .utils import analyze_subreddit_sentiment
from django.core.cache import cache


def home(request):
    # Home view
    return render(request, "dashboard/home.html")


def dashboard_view(request):
    # Get subreddit from query params
    subreddit_name = request.GET.get("subreddit", "").strip()
    refresh = request.GET.get("refresh") == "true"
    context = {}

    if subreddit_name:
        cache_key = f"sentiment_{subreddit_name.lower()}_100"

        if refresh:
            cache.delete(cache_key)

        result = analyze_subreddit_sentiment(subreddit_name)

        # If we cannot retrieve data for that subreddit
        if not result["success"]:
            context["error"] = result["error"]
        else:
            context["summary"] = result["summary"]

    return render(request, "dashboard/dashboard.html", context)
