"""
api.py — FastAPI backend that orchestrates the full anomaly detection pipeline.

Endpoints:
    GET /health              — liveness check
    GET /analyze             — runs fetch → features → model → explain, returns JSON
"""

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env before importing modules that read env vars (e.g. explain.py)
load_dotenv()

from data import fetch_stock_data
from features import engineer_features
from model import detect_anomalies
from explain import explain_anomaly

# --- App setup ---
app = FastAPI(
    title="Stock Anomaly Detector API",
    description="Detects and explains anomalous trading days using Isolation Forest + Claude.",
    version="1.0.0",
)

# Allow Streamlit (running on localhost:8501) to call this API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health endpoint ---
@app.get("/health", summary="Liveness check")
def health():
    """Simple health check so monitoring tools and the UI can verify the server is up."""
    return {"status": "ok"}


# --- Main analysis endpoint ---
@app.get("/analyze", summary="Detect and explain stock anomalies")
def analyze(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. AAPL"),
    top_n: int = Query(10, ge=1, le=50, description="Number of top anomalies to return"),
    contamination: float = Query(
        0.05, ge=0.01, le=0.10,
        description="Expected fraction of anomalies (0.01–0.10)"
    ),
):
    """
    Full pipeline:
      1. Fetch 2 years of OHLCV data via yfinance
      2. Engineer 5 trading features
      3. Train Isolation Forest and score every trading day
      4. For each of the top_n anomalies, call Claude + web search for an explanation
      5. Return structured JSON
    """

    # --- Step 1: Fetch raw data ---
    try:
        df_raw = fetch_stock_data(ticker, period="2y")
    except ConnectionError as e:
        # Rate limit hit — ask the client to retry after a short wait
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError as e:
        # Raised by fetch_stock_data when the ticker is invalid or delisted
        raise HTTPException(status_code=404, detail=str(e))

    # --- Step 2: Engineer features ---
    df_features = engineer_features(df_raw)

    # --- Step 3: Detect anomalies ---
    df_anomalies = detect_anomalies(df_features, contamination=contamination, top_n=top_n)

    # --- Step 4: Explain each anomaly ---
    results = []
    for date_idx, row in df_anomalies.iterrows():
        # Convert DatetimeIndex entry to ISO date string (YYYY-MM-DD)
        date_str = str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx)

        # Call Claude with web search; handles its own errors gracefully
        explanation = explain_anomaly(ticker.upper(), date_str, row)

        results.append({
            "date": date_str,
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "daily_return": round(float(row["daily_return"]), 4),
            "rel_volume": round(float(row["rel_volume"]), 4),
            "intraday_range": round(float(row["intraday_range"]), 4),
            "explanation": explanation,
        })

    # --- Step 5: Return structured response ---
    return {
        "ticker": ticker.upper(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "anomalies": results,
    }
