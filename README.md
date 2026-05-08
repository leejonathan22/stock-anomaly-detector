[README.md](https://github.com/user-attachments/files/27511482/README.md)
# 📈 Stock Anomaly Detector

A full-stack tool that automatically detects and explains anomalous trading days for any stock ticker. It combines **Isolation Forest** machine learning with **Claude AI + web search** to surface unusual market behavior and provide context for why each anomaly occurred.

---

## How It Works

The pipeline runs in four stages:

1. **Fetch** — pulls 2 years of OHLCV (Open, High, Low, Close, Volume) data via `yfinance`
2. **Feature Engineering** — computes 5 technical indicators per trading day
3. **Anomaly Detection** — trains an Isolation Forest to score every day; higher score = more anomalous
4. **Explanation** — calls Claude (with live web search) to explain *why* each flagged day was unusual

---

## Project Structure

```
stock-anomaly-detector/
├── api.py           # FastAPI backend — orchestrates the full pipeline, exposes /analyze endpoint
├── data.py          # Fetches raw OHLCV stock data via yfinance
├── features.py      # Engineers 5 trading features from raw data
├── model.py         # Trains Isolation Forest and scores anomalies
├── explain.py       # Calls Claude + web search to explain each anomaly
├── ui.py            # Streamlit frontend — user interface
├── requirements.txt # Python dependencies
└── .env.example     # Template for required environment variables
```

---

## Features Engineered

| Feature | Description |
|---|---|
| `daily_return` | % change in closing price vs. previous day |
| `volume_zscore` | Rolling 20-day z-score of volume (how unusual today's volume is) |
| `intraday_range` | `(High - Low) / Open * 100` — daily price swing as % of open |
| `volatility` | Rolling 20-day standard deviation of daily returns |
| `rel_volume` | Today's volume divided by its rolling 20-day mean |

---

## API Endpoints

The FastAPI backend (`api.py`) exposes two endpoints:

### `GET /health`
Liveness check — confirms the server is running.

```json
{ "status": "ok" }
```

### `GET /analyze`
Runs the full detection pipeline and returns structured results.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ticker` | string | required | Stock ticker symbol (e.g. `AAPL`) |
| `top_n` | int | `10` | Number of top anomalies to return (1–50) |
| `contamination` | float | `0.05` | Expected fraction of anomalies (0.01–0.10) |

**Example Response:**
```json
{
  "ticker": "AAPL",
  "generated_at": "2024-10-15T12:00:00+00:00",
  "anomalies": [
    {
      "date": "2024-08-05",
      "anomaly_score": 0.9821,
      "daily_return": -4.82,
      "rel_volume": 3.14,
      "intraday_range": 6.73,
      "explanation": "..."
    }
  ]
}
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/leejonathan22/stock-anomaly-detector.git
cd stock-anomaly-detector
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your_api_key_here
```

### Running the App

Start the FastAPI backend:

```bash
uvicorn api:app --reload
```

In a separate terminal, launch the Streamlit UI:

```bash
streamlit run ui.py
```

The UI will be available at `http://localhost:8501` and will communicate with the API at `http://localhost:8000`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | [yfinance](https://github.com/ranaroussi/yfinance) |
| ML | [scikit-learn](https://scikit-learn.org/) — Isolation Forest |
| AI Explanation | [Anthropic Claude](https://www.anthropic.com/) + web search |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Frontend | [Streamlit](https://streamlit.io/) |
| Charting | [Plotly](https://plotly.com/python/) |

---

## Disclaimer

This project is for educational and research purposes only. Anomaly detection results and AI-generated explanations should **not** be used as the basis for financial or investment decisions.
