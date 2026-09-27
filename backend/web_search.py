"""
Live web search fallback -- used only when the local knowledge base doesn't
have a confident answer. Uses the free, keyless DuckDuckGo search library,
so this works without you signing up for any search API.
"""

import re
import time

from ddgs import DDGS

import config

# Common filler words that make a question read naturally but hurt search
# engine matching (e.g. "what's a good way to get placed this year" search
# worse than "good way to get placed this year"). Stripped only on retry.
_FILLER_WORDS = {
    "whats", "what's", "what", "is", "are", "a", "an", "the", "to", "for",
    "of", "in", "on", "at", "how", "do", "i", "can", "you", "please",
    "tell", "me", "about",
}


def _simplify_query(query: str) -> str:
    """Strip punctuation and common filler words to make a more search-engine
    -friendly query, used only as a fallback if the raw query returns nothing."""
    words = re.sub(r"[^\w\s]", "", query).split()
    kept = [w for w in words if w.lower() not in _FILLER_WORDS]
    simplified = " ".join(kept) if kept else query
    return simplified


def _try_search(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def search_web(query: str, max_results: int = None) -> list[dict]:
    """
    Returns a list of {title, snippet, url} dicts, or [] if the search
    fails or returns nothing (network issues, rate limiting, etc).
    Never raises -- a failed web search should degrade gracefully, not
    crash the chat request.

    Resilience strategy, in order:
      1. Try the question as typed.
      2. If that returns nothing, retry the same query once after a short
         pause (handles transient rate-limiting).
      3. If still nothing, try a simplified, keyword-only version of the
         query (handles casual/conversational phrasing DuckDuckGo can miss).
    """
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
    attempts = [query]

    simplified = _simplify_query(query)
    if simplified.lower() != query.lower():
        attempts.append(simplified)

    raw_results = []
    for i, q in enumerate(attempts):
        try:
            raw_results = _try_search(q, max_results)
        except Exception as exc:
            print(f"[web_search] search failed for '{q}': {exc}")
            raw_results = []

        if raw_results:
            break

        # Nothing came back -- if this was the first (raw) attempt, give it
        # one quick retry before moving on to the simplified query, in case
        # it was just a transient hiccup.
        if i == 0:
            time.sleep(1.2)
            try:
                raw_results = _try_search(q, max_results)
            except Exception as exc:
                print(f"[web_search] retry failed for '{q}': {exc}")
                raw_results = []
            if raw_results:
                break

    if not raw_results:
        print(f"[web_search] no results for '{query}' (tried: {attempts})")

    results = []
    for r in raw_results:
        results.append({
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        })
    return results