"""Leakage-safe daily feature construction for NIFTY-Sentinel."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns using only the current and previous observed close."""
    return np.log(prices).diff()


def market_features(close: pd.Series, windows: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """Build trailing market-state features without forward information."""
    close = close.rename("nifty_close").sort_index()
    returns = np.log(close).diff().rename("nifty_log_return")
    result = pd.concat([close, returns], axis=1)

    for window in windows:
        result[f"volatility_{window}d"] = returns.rolling(window).std() * np.sqrt(252)
        result[f"momentum_{window}d"] = close.pct_change(window)
        peak = close.rolling(window, min_periods=1).max()
        result[f"drawdown_{window}d"] = close.div(peak).sub(1)

    return result


def rolling_connectedness(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate mean absolute off-diagonal correlation from trailing windows."""
    output: list[tuple[pd.Timestamp, float]] = []
    cleaned = returns.dropna(how="all")
    for end in range(window, len(cleaned) + 1):
        matrix = cleaned.iloc[end - window : end].corr().to_numpy()
        upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
        output.append((cleaned.index[end - 1], float(np.nanmean(np.abs(upper_triangle)))))
    return pd.DataFrame(output, columns=["date", f"mean_abs_correlation_{window}d"]).set_index("date")


def future_stress_label(
    close: pd.Series,
    horizon: int = 10,
    drawdown_threshold: float = -0.05,
    volatility_quantile: float = 0.80,
) -> pd.Series:
    """Create a documented forward label for supervised experiments.

    Thresholds are research assumptions. Tune them only through a training-period
    protocol and keep final test periods untouched.
    """
    future_min = pd.concat([close.shift(-step) for step in range(1, horizon + 1)], axis=1).min(axis=1)
    forward_drawdown = future_min.div(close).sub(1)
    returns = np.log(close).diff()
    future_volatility = returns.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    observed_volatility = returns.rolling(horizon).std() * np.sqrt(252)
    cutoff = observed_volatility.expanding(min_periods=252).quantile(volatility_quantile).shift(1)
    valid_forward_window = close.shift(-horizon).notna() & future_volatility.notna()
    stress = ((forward_drawdown <= drawdown_threshold) | (future_volatility >= cutoff)).where(valid_forward_window).astype("Int64")
    stress.name = f"stress_next_{horizon}d"
    return stress
