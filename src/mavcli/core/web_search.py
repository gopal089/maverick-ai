"""Web search functionality for Maverick AI using Tavily Search API."""

import logging
import os
import requests

logger = logging.getLogger(__name__)

def web_search(query: str) -> str:
    """
    Perform a web search using Tavily Search API.

    Args:
        query: The search query string

    Returns:
        A concise text summary of search results
    """
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