"""
ui.py — Streamlit front-end for the Stock Anomaly Detector.

Run with:  streamlit run ui.py
Requires the FastAPI backend to be running at http://localhost:8000
"""

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# URL of the FastAPI backend
API_BASE = "http://localhost:8000"


# --- Page config ---
st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Anomaly Detector")
st.caption(
    "Detects unusual trading days using Isolation Forest and explains them with Claude AI + web search."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Ticker input — default to AAPL as a familiar example
    ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).strip().upper()

    # Number of top anomalies to surface
    top_n = st.slider(
        "Top N Anomalies",
        min_value=3,
        max_value=20,
        value=10,
        help="How many of the most anomalous days to display and explain.",
    )

    # Contamination: expected fraction of anomalous days in the dataset
    contamination = st.slider(
        "Contamination",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help=(
            "Expected proportion of anomalies. "
            "Lower = only flag extreme outliers; higher = flag more days."
        ),
    )

    # Trigger button — analysis only runs when clicked to avoid redundant API calls
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True)

# ── Main content area ─────────────────────────────────────────────────────────
if not analyze_clicked:
    # Show instructions when the app first loads
    st.info("Configure settings in the sidebar and click **Analyze** to get started.")
    st.stop()

# Check that the backend is reachable before attempting the full analysis
try:
    health = requests.get(f"{API_BASE}/health", timeout=3)
    health.raise_for_status()
except Exception:
    st.error(
        "❌ Cannot reach the FastAPI backend at `http://localhost:8000`. "
        "Make sure it's running with: `uvicorn api:app --reload`"
    )
    st.stop()

# --- Call the /analyze endpoint with a loading spinner ---
with st.spinner(f"Fetching data and analyzing {ticker}… this may take a minute."):
    try:
        resp = requests.get(
            f"{API_BASE}/analyze",
            params={"ticker": ticker, "top_n": top_n, "contamination": contamination},
            timeout=300,  # Claude web-search calls can be slow; give plenty of time
        )

        # Handle specific HTTP error codes with friendly messages
        if resp.status_code == 404:
            st.error(f"❌ Ticker **{ticker}** not found. Please check the symbol and try again.")
            st.stop()
        if resp.status_code == 429:
            st.warning("⏳ yfinance rate limit hit. Wait 30–60 seconds and click Analyze again.")
            st.stop()

        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError:
        st.error("❌ Lost connection to the backend. Is `uvicorn api:app --reload` still running?")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

# Parse anomalies into a DataFrame for display
anomalies = data.get("anomalies", [])
if not anomalies:
    st.warning("No anomalies were returned for this ticker and settings combination.")
    st.stop()

df_anomalies = pd.DataFrame(anomalies)
df_anomalies["date"] = pd.to_datetime(df_anomalies["date"])

# ── Candlestick chart ─────────────────────────────────────────────────────────
st.subheader(f"🕯️ {ticker} — Last 2 Years with Anomaly Highlights")

# Fetch full OHLCV data directly from yfinance for the chart
# (The API only returns the top-N anomalies, not the full price series)
try:
    import yfinance as yf
    raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    fig = go.Figure()

    # Candlestick trace for the full price history
    fig.add_trace(
        go.Candlestick(
            x=raw.index,
            open=raw["Open"],
            high=raw["High"],
            low=raw["Low"],
            close=raw["Close"],
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Add a red vertical line + annotation for each anomaly date
    for _, anom in df_anomalies.iterrows():
        anom_date = anom["date"]
        score = anom["anomaly_score"]

        # Vertical line via shape
        fig.add_vline(
            x=anom_date,
            line_width=1.5,
            line_dash="dot",
            line_color="red",
        )

        # Score label above the candle
        fig.add_annotation(
            x=anom_date,
            y=1.01,
            yref="paper",          # position relative to chart height, not price
            text=f"{score:.2f}",
            showarrow=False,
            font=dict(size=9, color="red"),
            textangle=-60,
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Could not render candlestick chart: {e}")

# ── Anomaly table ─────────────────────────────────────────────────────────────
st.subheader("🔎 Top Anomalous Trading Days")

# Build a display-friendly table
display_df = df_anomalies.copy()
display_df["Date"] = display_df["date"].dt.strftime("%Y-%m-%d")
display_df["Return %"] = display_df["daily_return"].map("{:+.2f}%".format)
display_df["Volume vs Avg"] = display_df["rel_volume"].map("{:.1f}x".format)
display_df["Anomaly Score"] = display_df["anomaly_score"].map("{:.4f}".format)
display_df["Explanation"] = display_df["explanation"]

# Only show the human-friendly columns
table_cols = ["Date", "Return %", "Volume vs Avg", "Anomaly Score", "Explanation"]
st.dataframe(
    display_df[table_cols].set_index("Date"),
    use_container_width=True,
    height=400,
)

# Timestamp footer
st.caption(f"Analysis generated at: {data.get('generated_at', 'unknown')}")
