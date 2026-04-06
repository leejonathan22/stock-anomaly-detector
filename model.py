"""
model.py — Trains an Isolation Forest to detect anomalous trading days.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


# The five features that the Isolation Forest is trained on
FEATURE_COLS = ["daily_return", "volume_zscore", "intraday_range", "volatility", "rel_volume"]


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Train an Isolation Forest on the engineered features and label anomalous rows.

    Isolation Forest works by randomly partitioning the feature space;
    samples that are isolated quickly (short average path length) are anomalies.

    Args:
        df:            DataFrame produced by engineer_features() — must contain FEATURE_COLS
        contamination: Expected proportion of anomalies in the dataset (0.01–0.50)
        top_n:         Number of most anomalous rows to return

    Returns:
        DataFrame of the top_n most anomalous rows, sorted by anomaly_score descending,
        with two additional columns:
            anomaly_score — float in [0, 1], higher means more anomalous
            is_anomaly    — bool, True for rows classified as anomalies
    """

    # Extract only the feature matrix; drop rows with any remaining NaNs
    X = df[FEATURE_COLS].dropna()

    # Align the main DataFrame to the cleaned feature index
    df = df.loc[X.index].copy()

    # --- Train Isolation Forest ---
    # random_state ensures reproducibility across runs
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,      # more trees → more stable scores
    )
    iso_forest.fit(X)

    # decision_function returns raw anomaly scores:
    #   more negative  → more anomalous
    #   near 0 / positive → normal
    raw_scores = iso_forest.decision_function(X)

    # Invert and normalize to [0, 1] so that higher score = more anomalous
    # MinMaxScaler maps the most anomalous score → 1.0, least → 0.0
    inverted = -raw_scores.reshape(-1, 1)        # flip sign: higher is worse
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(inverted).flatten()

    df["anomaly_score"] = normalized

    # Label rows that Isolation Forest explicitly flags as anomalies (-1 = anomaly)
    predictions = iso_forest.predict(X)
    df["is_anomaly"] = predictions == -1

    # Return the top_n rows sorted by anomaly score (highest first)
    top = df.nlargest(top_n, "anomaly_score")

    return top
