"""Build a local exploratory dashboard dataset from public market-data symbols.

This runner is for demonstrating the dashboard. Replace its exploratory inputs
with official NSE/RBI files before reporting academic results.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs
from src.features import future_stress_label, log_returns, market_features, rolling_connectedness

SYMBOLS = {
    "nifty_close": "^NSEI",
    "india_vix": "^INDIAVIX",
    "usd_inr": "INR=X",
    "gold": "GC=F",
    "brent": "BZ=F",
    "sp500": "^GSPC",
    "us_vix": "^VIX",
}


def main() -> None:
    ensure_project_dirs()
    raw_prices_path = RAW_DATA_DIR / "exploratory_market_prices.csv"
    downloaded = yf.download(
        list(SYMBOLS.values()),
        start="2010-01-01",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if downloaded.empty or "Close" not in downloaded:
        if raw_prices_path.exists():
            close = pd.read_csv(raw_prices_path, parse_dates=["date"]).set_index("date")
        else:
            raise RuntimeError("Market-data download returned no close prices and no local raw fallback exists.")
    else:
        close = downloaded["Close"].rename(columns={symbol: name for name, symbol in SYMBOLS.items()})
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close.index.name = "date"
        close = close.dropna(axis=1, how="all")
        close = close.loc[close["nifty_close"].notna()].ffill(limit=3)
        close.to_csv(raw_prices_path)
    if len(close) < 800:
        raise RuntimeError("Insufficient NIFTY observations to fit the prototype.")

    features = market_features(close["nifty_close"])
    for source, target in (
        ("india_vix", "india_vix_change"),
        ("usd_inr", "usd_inr_return"),
        ("gold", "gold_return"),
    ):
        if source in close and close[source].notna().any():
            features[target] = close[source].pct_change()

    features["stress_next_10d"] = future_stress_label(close["nifty_close"], horizon=10)
    asset_returns = log_returns(close.dropna(axis=1, how="all"))
    connectedness = rolling_connectedness(asset_returns, window=60)
    features = features.join(connectedness, how="left")
    if "mean_abs_correlation_60d" in features:
        features["connectedness_60d"] = features["mean_abs_correlation_60d"]
    asset_returns.tail(60).corr().to_csv(PROCESSED_DATA_DIR / "latest_correlation_matrix.csv")

    features["regime"] = pd.Series(pd.NA, index=features.index, dtype="Int64")
    regime_columns = [
        column
        for column in ("nifty_log_return", "volatility_20d", "drawdown_60d", "india_vix_change", "usd_inr_return")
        if column in features and features[column].notna().any()
    ]
    regime_input = features[regime_columns].dropna()
    if len(regime_input) >= 250:
        scaled = (regime_input - regime_input.mean()) / regime_input.std()
        hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=300, random_state=42)
        hmm.fit(scaled)
        features.loc[regime_input.index, "regime"] = hmm.predict(scaled)

    target = "stress_next_10d"
    feature_columns = [
        column
        for column in features.select_dtypes("number").columns
        if column not in {target, "nifty_close"} and features[column].notna().any()
    ]
    labeled = features.dropna(subset=[target])
    if labeled[target].nunique() < 2:
        raise RuntimeError("The selected stress target contains only one class.")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(labeled[feature_columns], labeled[target].astype(int))
    features["stress_probability"] = model.predict_proba(features[feature_columns])[:, 1]
    features["warning_level"] = pd.cut(
        features["stress_probability"],
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Normal", "Elevated", "Severe"],
    )

    importances = pd.Series(model.named_steps["model"].feature_importances_, index=feature_columns).sort_values(ascending=False)
    for rank, (name, value) in enumerate(importances.head(8).items(), start=1):
        features[f"driver_{rank}_name"] = name
        features[f"driver_{rank}_importance"] = value

    dashboard_columns = [
        column
        for column in [
            "nifty_close",
            "nifty_log_return",
            "volatility_5d",
            "volatility_20d",
            "volatility_60d",
            "momentum_20d",
            "drawdown_60d",
            "india_vix_change",
            "usd_inr_return",
            "gold_return",
            "connectedness_60d",
            "regime",
            "stress_next_10d",
            "stress_probability",
            "warning_level",
            *[f"driver_{rank}_name" for rank in range(1, 9)],
            *[f"driver_{rank}_importance" for rank in range(1, 9)],
        ]
        if column in features
    ]
    dashboard = features[dashboard_columns].reset_index()
    dashboard.to_csv(PROCESSED_DATA_DIR / "dashboard_data.csv", index=False)
    features.to_csv(PROCESSED_DATA_DIR / "local_prototype_features.csv")
    close.reset_index().to_csv(PROCESSED_DATA_DIR / "master_market_data.csv", index=False)
    state_columns = [
        column
        for column in features.columns
        if column
        in {
            "nifty_close",
            "nifty_log_return",
            "volatility_5d",
            "momentum_5d",
            "drawdown_5d",
            "volatility_20d",
            "momentum_20d",
            "drawdown_20d",
            "volatility_60d",
            "momentum_60d",
            "drawdown_60d",
            "india_vix_change",
            "usd_inr_return",
            "gold_return",
            "stress_next_10d",
            "regime",
        }
    ]
    features[state_columns].reset_index().to_csv(PROCESSED_DATA_DIR / "market_state_features.csv", index=False)
    features[state_columns].reset_index().to_csv(PROCESSED_DATA_DIR / "regime_features.csv", index=False)
    connectedness_columns = [column for column in ("mean_abs_correlation_60d", "connectedness_60d") if column in features]
    features[connectedness_columns].reset_index().to_csv(PROCESSED_DATA_DIR / "connectedness_features.csv", index=False)
    (PROCESSED_DATA_DIR / "local_prototype_metadata.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "source": "Exploratory public market-data symbols via yfinance",
                "warning": "Replace with validated official NSE/RBI data before academic reporting.",
                "rows": len(dashboard),
                "latest_date": str(dashboard["date"].iloc[-1].date()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(dashboard):,} dashboard rows through {dashboard['date'].iloc[-1].date()}.")


if __name__ == "__main__":
    main()
