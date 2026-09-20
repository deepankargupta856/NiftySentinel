"""Streamlit decision-support dashboard for NIFTY-Sentinel."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_PATH = Path("data/processed/dashboard_data.csv")
CORRELATION_PATH = Path("data/processed/latest_correlation_matrix.csv")
METADATA_PATH = Path("data/processed/local_prototype_metadata.json")
CASE_STUDIES = {
    "2013 taper-related stress": ("2013-05-01", "2013-10-31"),
    "2018 IL&FS/NBFC stress": ("2018-07-01", "2018-12-31"),
    "2020 COVID-19 shock": ("2020-01-01", "2020-06-30"),
    "2022 inflation and tightening": ("2022-01-01", "2022-12-31"),
}
WARNING_COLORS = {"Normal": "#2ca25f", "Elevated": "#f0ad4e", "Severe": "#d9534f"}


def load_dashboard_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    required = {"date", "stress_probability", "warning_level", "nifty_close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dashboard data is missing required columns: {', '.join(sorted(missing))}")
    return data


def add_thresholds(figure: go.Figure) -> go.Figure:
    figure.add_hline(y=0.30, line_dash="dot", line_color="#f0ad4e", annotation_text="Elevated")
    figure.add_hline(y=0.60, line_dash="dash", line_color="#d9534f", annotation_text="Severe")
    return figure


def compact_number(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.2f}{suffix}"


def render_header(data: pd.DataFrame) -> None:
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    st.title("NIFTY-Sentinel Market Stress Intelligence")
    st.caption(
        "Explainable early-warning dashboard for Indian market stress, built from the project report's "
        "market state, risk appetite, volatility, regime, and connectedness layers."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Stress probability", f"{latest['stress_probability']:.1%}", f"{latest['stress_probability'] - previous['stress_probability']:+.1%}")
    col2.metric("Warning", str(latest["warning_level"]))
    col3.metric("NIFTY close", compact_number(latest["nifty_close"]), compact_number(latest["nifty_close"] - previous["nifty_close"]))
    col4.metric("Regime", str(latest.get("regime", "n/a")))
    col5.metric("Connectedness", compact_number(latest.get("connectedness_60d"), ""))


def render_overview(data: pd.DataFrame) -> None:
    probability = px.area(
        data,
        x="date",
        y="stress_probability",
        color="warning_level",
        color_discrete_map=WARNING_COLORS,
        title="Stress probability with warning zones",
        labels={"stress_probability": "Probability", "date": "Date", "warning_level": "Warning"},
    )
    add_thresholds(probability)
    st.plotly_chart(probability, width="stretch")

    regime_data = data.dropna(subset=["regime"]) if "regime" in data else pd.DataFrame()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(x=data["date"], y=data["nifty_close"], name="NIFTY close", line={"color": "#1f77b4"}), secondary_y=False)
    figure.add_trace(go.Scatter(x=data["date"], y=data["stress_probability"], name="Stress probability", line={"color": "#d62728"}), secondary_y=True)
    if not regime_data.empty:
        figure.add_trace(
            go.Scatter(
                x=regime_data["date"],
                y=regime_data["nifty_close"],
                mode="markers",
                marker={"size": 5, "color": regime_data["regime"], "colorscale": "Viridis", "showscale": True},
                name="HMM regime",
            ),
            secondary_y=False,
        )
    figure.update_layout(title="NIFTY price, stress probability, and inferred regimes", height=520)
    figure.update_yaxes(title_text="NIFTY close", secondary_y=False)
    figure.update_yaxes(title_text="Stress probability", tickformat=".0%", secondary_y=True, range=[0, 1])
    st.plotly_chart(figure, width="stretch")


def render_market_layers(data: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        available = [column for column in ["volatility_5d", "volatility_20d", "volatility_60d"] if column in data]
        if available:
            st.plotly_chart(
                px.line(data, x="date", y=available, title="Rolling volatility stack", labels={"value": "Volatility", "variable": "Window"}),
                width="stretch",
            )
        if "drawdown_60d" in data:
            st.plotly_chart(
                px.area(data, x="date", y="drawdown_60d", title="60-day drawdown depth", labels={"drawdown_60d": "Drawdown"}),
                width="stretch",
            )
    with col2:
        available = [column for column in ["momentum_20d", "nifty_log_return"] if column in data]
        if available:
            st.plotly_chart(
                px.line(data, x="date", y=available, title="Return and momentum signals", labels={"value": "Signal", "variable": "Feature"}),
                width="stretch",
            )
        if "connectedness_60d" in data:
            st.plotly_chart(
                px.line(data, x="date", y="connectedness_60d", title="Rolling cross-asset connectedness", labels={"connectedness_60d": "Mean absolute correlation"}),
                width="stretch",
            )


def render_risk_appetite(data: pd.DataFrame) -> None:
    available = [column for column in ["india_vix_change", "usd_inr_return", "gold_return"] if column in data]
    if not available:
        st.info("Risk-appetite proxies will appear after market inputs are available.")
        return
    melted = data.melt(id_vars="date", value_vars=available, var_name="proxy", value_name="return")
    st.plotly_chart(
        px.line(melted, x="date", y="return", color="proxy", title="Risk appetite and macro proxy moves"),
        width="stretch",
    )

    recent = data.tail(252)
    matrix_columns = [column for column in available + ["stress_probability"] if column in recent]
    if len(matrix_columns) >= 2:
        st.plotly_chart(
            px.scatter_matrix(
                recent,
                dimensions=matrix_columns,
                color="warning_level",
                color_discrete_map=WARNING_COLORS,
                title="Recent proxy relationship matrix",
            ).update_traces(diagonal_visible=False),
            width="stretch",
        )


def render_connectedness() -> None:
    if not CORRELATION_PATH.exists():
        st.info("Run the regimes and connectedness notebook to publish the latest correlation matrix.")
        return
    correlation = pd.read_csv(CORRELATION_PATH, index_col=0)
    st.plotly_chart(
        px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Latest 60-day cross-asset return correlation",
            labels={"color": "Correlation"},
        ),
        width="stretch",
    )
    edge_data = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack().reset_index()
    edge_data.columns = ["asset_1", "asset_2", "correlation"]
    edge_data["absolute_correlation"] = edge_data["correlation"].abs()
    edge_data = edge_data.sort_values("absolute_correlation", ascending=False).head(12)
    if not edge_data.empty:
        edge_data["pair"] = edge_data["asset_1"] + " / " + edge_data["asset_2"]
        st.plotly_chart(
            px.bar(edge_data, x="absolute_correlation", y="pair", orientation="h", title="Strongest current cross-asset links"),
            width="stretch",
        )


def render_model_explainability(data: pd.DataFrame) -> None:
    latest = data.iloc[-1]
    name_columns = sorted([column for column in data.columns if column.startswith("driver_") and column.endswith("_name")])
    rows = []
    for name_column in name_columns:
        rank = name_column.split("_")[1]
        importance_column = f"driver_{rank}_importance"
        if importance_column in data and pd.notna(latest[name_column]):
            rows.append({"feature": str(latest[name_column]).replace("_", " "), "importance": latest[importance_column]})
    if rows:
        drivers = pd.DataFrame(rows).sort_values("importance", ascending=True)
        st.plotly_chart(
            px.bar(drivers, x="importance", y="feature", orientation="h", title="Random Forest feature importance from actual historical market data"),
            width="stretch",
        )

    if "stress_next_10d" in data:
        labeled = data.dropna(subset=["stress_next_10d"])
        if not labeled.empty:
            validation = pd.crosstab(labeled["warning_level"], labeled["stress_next_10d"], normalize="index").reset_index()
            validation = validation.melt(id_vars="warning_level", var_name="Observed future stress", value_name="Share")
            st.plotly_chart(
                px.bar(validation, x="warning_level", y="Share", color="Observed future stress", barmode="stack", title="Warning bands versus forward stress labels"),
                width="stretch",
            )


def render_case_studies(data: pd.DataFrame) -> None:
    choice = st.selectbox("Historical episode", list(CASE_STUDIES))
    start, end = CASE_STUDIES[choice]
    subset = data[data["date"].between(start, end)]
    if subset.empty:
        st.warning("This dataset does not yet cover the selected episode.")
        return
    col1, col2 = st.columns(2)
    with col1:
        chart = px.line(subset, x="date", y="stress_probability", title=f"{choice}: stress probability")
        add_thresholds(chart)
        st.plotly_chart(chart, width="stretch")
    with col2:
        st.plotly_chart(px.line(subset, x="date", y="nifty_close", title=f"{choice}: NIFTY close"), width="stretch")
    feature_columns = [column for column in ["volatility_20d", "drawdown_60d", "connectedness_60d", "usd_inr_return", "gold_return"] if column in subset]
    if feature_columns:
        st.plotly_chart(px.line(subset, x="date", y=feature_columns, title=f"{choice}: stress-layer indicators"), width="stretch")


def render_method(data: pd.DataFrame) -> None:
    st.subheader("Project alignment")
    st.write(
        "This dashboard follows the initial NIFTY-Sentinel documentation: it is an explainable early-warning "
        "system for stress conditions, not a NIFTY price predictor or trading system."
    )
    st.subheader("Live market data status")
    st.write(f"Rows loaded: {len(data):,}. Date range: {data['date'].min().date()} to {data['date'].max().date()}.")
    if METADATA_PATH.exists():
        st.code(METADATA_PATH.read_text(encoding="utf-8"), language="json")
    st.subheader("Validation reminder")
    st.write(
        "The dashboard uses actual historical public-market data committed with the repository. "
        "Official NSE/RBI validation is still recommended before final academic performance claims."
    )


st.set_page_config(page_title="NIFTY-Sentinel", page_icon="N", layout="wide")

if not DATA_PATH.exists():
    st.info("No dashboard data is available yet. Run the notebooks or scripts/run_local_prototype.py to create dashboard_data.csv.")
    st.stop()

try:
    dashboard_data = load_dashboard_data()
except ValueError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("NIFTY-Sentinel")
    page = st.radio(
        "Dashboard section",
        [
            "Executive monitor",
            "Market layers",
            "Risk appetite",
            "Connectedness",
            "Model explainability",
            "Historical case studies",
            "Method",
        ],
    )
    window = st.slider("Visible history in trading days", min_value=126, max_value=len(dashboard_data), value=min(1260, len(dashboard_data)), step=63)

visible_data = dashboard_data.tail(window)
render_header(visible_data)

if page == "Executive monitor":
    render_overview(visible_data)
elif page == "Market layers":
    render_market_layers(visible_data)
elif page == "Risk appetite":
    render_risk_appetite(visible_data)
elif page == "Connectedness":
    render_connectedness()
elif page == "Model explainability":
    render_model_explainability(visible_data)
elif page == "Historical case studies":
    render_case_studies(dashboard_data)
else:
    render_method(dashboard_data)

st.caption("For academic use only. Feature attribution explains model behaviour; it does not prove causality or provide trading advice.")
