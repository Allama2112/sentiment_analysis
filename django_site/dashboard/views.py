from django.shortcuts import render
from .utils import analyze_subreddit_sentiment


def home(request):
    # Home view
    return render(request, "dashboard/home.html")


def dashboard_view(request):
    # Get subreddit from query params
    subreddit_name = request.GET.get("subreddit", "").strip()
    context = {}

    if subreddit_name:
        result = analyze_subreddit_sentiment(subreddit_name)

    # If we cannot retrieve data for that subreddit
    if not result["success"]:
        context["error"] = result["error"]

    else:
        context["summary"] = result["summary"]

    return render(request, "dashboard/dashboard.html", context)
