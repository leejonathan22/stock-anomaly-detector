"""
explain.py — Uses Anthropic Claude with web search to explain stock anomalies.
"""

import os
import anthropic
import pandas as pd
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from .env file (falls back to system environment)
load_dotenv()

# claude-sonnet-4-6 is the current Sonnet model and supports the web search tool
MODEL_ID = "claude-sonnet-4-6"

# System prompt that sets the analyst persona and strict output format
SYSTEM_PROMPT = (
    "You are a senior equity analyst assistant. When given a stock ticker, "
    "a specific date, and anomaly metrics, you search the web for news and "
    "events from that date and explain what likely caused the unusual trading "
    "activity. Your explanation must be 2-3 sentences in plain professional "
    "English, lead with the most likely cause, reference specific facts you "
    "found, and never speculate without evidence. If no clear cause is found, "
    "say so honestly. Tone: factual, confident, concise."
)


def explain_anomaly(ticker: str, date: str, row: pd.Series) -> str:
    """
    Ask Claude (with web search) to explain why a specific trading day was anomalous.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL"
        date:   ISO date string, e.g. "2024-02-08"
        row:    Pandas Series containing at least: anomaly_score, daily_return,
                rel_volume, intraday_range

    Returns:
        2-3 sentence explanation string, or "Explanation unavailable" on any failure.
    """

    # Build the user prompt with the anomaly metrics filled in
    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Date: {date}\n"
        f"Anomaly score: {row['anomaly_score']:.3f}\n"
        f"Daily return: {row['daily_return']:+.2f}%\n"
        f"Volume vs 20-day avg: {row['rel_volume']:.1f}x\n"
        f"Intraday range: {row['intraday_range']:.2f}%\n\n"
        f"Search the web for news about {ticker} on or around {date} and explain "
        "what likely caused this unusual trading day."
    )

    try:
        # Instantiate the Anthropic client; it reads ANTHROPIC_API_KEY from the environment
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # web_search_20250305 allows Claude to fetch live search results
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract all text blocks from the response (tool_use blocks are skipped)
        explanation_parts = [
            block.text
            for block in response.content
            if hasattr(block, "text") and block.text.strip()
        ]

        explanation = " ".join(explanation_parts).strip()

        # Return a safe fallback if Claude returned an empty response
        return explanation if explanation else "Explanation unavailable."

    except anthropic.APIError as e:
        # API-level errors (auth, rate limits, etc.) — log briefly and return fallback
        print(f"[explain_anomaly] Anthropic API error for {ticker} on {date}: {e}")
        return "Explanation unavailable."

    except Exception as e:
        # Catch-all for unexpected errors (network, parsing, etc.)
        print(f"[explain_anomaly] Unexpected error for {ticker} on {date}: {e}")
        return "Explanation unavailable."
