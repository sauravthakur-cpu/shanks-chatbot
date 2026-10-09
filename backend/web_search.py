"""
Live web search fallback -- used only when the local knowledge base doesn't
have a confident answer. Uses the free, keyless DuckDuckGo search library,
so this works without you signing up for any search API.

Resilience: try the question as typed -> retry once after a short pause ->
retry with filler words stripped (keyword-style query). Every attempt has a
timeout so a slow network can never freeze the chat.
"""

import re
import time

from ddgs import DDGS

import config

FILLER_WORDS = {
    "what", "whats", "what's", "how", "hows", "how's", "is", "are", "am", "was", "were",
    "a", "an", "the", "of", "to", "in", "on", "for", "do", "does", "did", "can", "could",
    "should", "would", "i", "me", "my", "we", "you", "please", "tell", "about", "get",
    "good", "way", "this", "that", "there", "any", "some",
}


def _simplify(query: str) -> str:
    """'whats a good way to get placed this year' -> 'placed year'"""
    words = re.findall(r"[a-zA-Z0-9']+", query.lower())
    keep = [w for w in words if w not in FILLER_WORDS]
    return " ".join(keep) if keep else query


def _run_search(query: str, max_results: int) -> list[dict]:
    try:
        try:
            client = DDGS(timeout=config.WEB_SEARCH_TIMEOUT)
        except TypeError:  # very old/new ddgs versions without a timeout argument
            client = DDGS()
        with client as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        print(f"[web_search] attempt failed for '{query}': {exc}")
        return []


def search_web(query: str, max_results: int = None) -> list[dict]:
    """
    Returns a list of {title, snippet, url} dicts, or [] if everything fails.
    Never raises -- a failed web search degrades gracefully instead of
    crashing the chat request.
    """
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS

    raw = _run_search(query, max_results)          # attempt 1: as typed
    if not raw:
        time.sleep(0.5)                            # attempt 2: absorb rate-limiting
        raw = _run_search(query, max_results)
    if not raw:
        simple = _simplify(query)                  # attempt 3: keyword-style query
        if simple != query.lower():
            raw = _run_search(simple, max_results)

    results = []
    for r in raw:
        results.append({
            "title": r.get("title", ""),
            # Trim long snippets: fewer tokens for the LLM to read = faster answer.
            "snippet": (r.get("body", "") or "")[:300],
            "url": r.get("href", ""),
        })
    return results
