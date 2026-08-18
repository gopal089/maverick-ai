"""Web search functionality for Maverick AI supporting Tavily and Brave Search APIs."""

import logging
import os
import requests

logger = logging.getLogger(__name__)

def web_search(query: str, search_provider: str = None) -> str:
    """
    Perform a web search using the configured search provider.

    Args:
        query: The search query string
        search_provider: The search provider to use ('tavily' or 'brave').
                        If None, will check SEARCH_PROVIDER environment variable or default to 'tavily'.

    Returns:
        A concise text summary of search results
    """
    # Determine which search provider to use
    if search_provider is None:
        search_provider = os.environ.get("SEARCH_PROVIDER", "tavily").lower()

    search_provider = search_provider.lower()

    if search_provider == "tavily":
        return _web_search_tavily(query)
    elif search_provider == "brave":
        return _web_search_brave(query)
    else:
        logger.warning(f"Unknown search provider '{search_provider}'. Falling back to Tavily.")
        return _web_search_tavily(query)

def _web_search_tavily(query: str) -> str:
    """Perform a web search using Tavily Search API."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY environment variable not set")
        return "Search is not configured — no API key found. Please set the TAVILY_API_KEY environment variable to enable web search."

    try:
        # Using Tavily Search API
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5,
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract answer and results
        answer = data.get('answer', '')
        results = data.get('results', [])

        # Build a concise summary
        summary_parts = []

        if answer:
            summary_parts.append(answer)

        # Add snippets from results
        for result in results[:3]:  # Use top 3 results
            if isinstance(result, dict):
                content = result.get('content', '')
                if content:
                    summary_parts.append(content)
                title = result.get('title', '')
                if title and title not in summary_parts[-1]:
                    # Prepend title if not already included
                    summary_parts[-1] = f"{title}: {summary_parts[-1]}"

        # Join and limit length
        summary = " ".join(summary_parts)

        # If no useful results found
        if not summary or len(summary.strip()) < 10:
            return f"I searched for '{query}' but couldn't find any useful information."

        # Truncate to reasonable length for LLM consumption
        if len(summary) > 1500:
            summary = summary[:1500] + "..."

        logger.info(f"Web search for '{query}' returned: {summary[:100]}...")
        logger.debug(f"Web search full summary for '{query}': {summary}")
        return summary

    except requests.exceptions.RequestException as e:
        logger.error(f"Error performing web search: {e}")
        return f"I encountered an error while searching the web for '{query}': {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in web search: {e}")
        return f"I encountered an unexpected error while searching for '{query}': {str(e)}"

def _web_search_brave(query: str) -> str:
    """Perform a web search using Brave Search API."""
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        logger.warning("BRAVE_API_KEY environment variable not set")
        return "Search is not configured — no API key found. Please set the BRAVE_API_KEY environment variable to enable web search."

    try:
        # Using Brave Search API
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        params = {
            "q": query,
            "count": 5,  # Number of results
            "offset": 0,
            "safesearch": "moderate"
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract results
        web_results = data.get('web', {})
        results = web_results.get('results', [])

        # Build a concise summary
        summary_parts = []

        # Add snippets from results
        for result in results[:3]:  # Use top 3 results
            if isinstance(result, dict):
                # Brave API provides description and title
                description = result.get('description', '')
                title = result.get('title', '')

                if description:
                    if title and title not in description:
                        summary_parts.append(f"{title}: {description}")
                    else:
                        summary_parts.append(description)
                elif title:
                    summary_parts.append(title)

        # Join and limit length
        summary = " ".join(summary_parts)

        # If no useful results found
        if not summary or len(summary.strip()) < 10:
            return f"I searched for '{query}' but couldn't find any useful information."

        # Truncate to reasonable length for LLM consumption
        if len(summary) > 1500:
            summary = summary[:1500] + "..."

        logger.info(f"Web search for '{query}' returned: {summary[:100]}...")
        logger.debug(f"Web search full summary for '{query}': {summary}")
        return summary

    except requests.exceptions.RequestException as e:
        logger.error(f"Error performing web search: {e}")
        return f"I encountered an error while searching the web for '{query}': {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in web search: {e}")
        return f"I encountered an unexpected error while searching for '{query}': {str(e)}"