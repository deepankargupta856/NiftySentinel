"""Build a static GitHub Pages dashboard from the latest NIFTY-Sentinel outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dashboard_data.csv"
CORRELATION_PATH = ROOT / "data" / "processed" / "latest_correlation_matrix.csv"
METADATA_PATH = ROOT / "data" / "processed" / "local_prototype_metadata.json"
OUT_DIR = ROOT / "site_pages"
WARNING_COLORS = {"Normal": "#2ca25f", "Elevated": "#f0ad4e", "Severe": "#d9534f"}


def chart_div(figure: go.Figure) -> str:
    return figure.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})


def add_thresholds(figure: go.Figure) -> go.Figure:
    figure.add_hline(y=0.30, line_dash="dot", line_color="#f0ad4e", annotation_text="Elevated")
    figure.add_hline(y=0.60, line_dash="dash", line_color="#d9534f", annotation_text="Severe")
    return figure


def metric_card(label: str, value: str, detail: str = "") -> str:
    detail_html = f"<span>{detail}</span>" if detail else ""
    return f"<article class='metric'><p>{label}</p><strong>{value}</strong>{detail_html}</article>"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    visible = data.tail(min(1260, len(data)))
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

    probability = px.area(
        visible,
        x="date",
        y="stress_probability",
        color="warning_level",
        color_discrete_map=WARNING_COLORS,
        title="Stress Probability With Warning Zones",
    )
    add_thresholds(probability)

    regime_data = visible.dropna(subset=["regime"]) if "regime" in visible else pd.DataFrame()
    regime_price = make_subplots(specs=[[{"secondary_y": True}]])
    regime_price.add_trace(go.Scatter(x=visible["date"], y=visible["nifty_close"], name="NIFTY close"), secondary_y=False)
    regime_price.add_trace(
        go.Scatter(x=visible["date"], y=visible["stress_probability"], name="Stress probability"),
        secondary_y=True,
    )
    if not regime_data.empty:
        regime_price.add_trace(
            go.Scatter(
                x=regime_data["date"],
                y=regime_data["nifty_close"],
                mode="markers",
                marker={"size": 5, "color": regime_data["regime"], "colorscale": "Viridis", "showscale": True},
                name="HMM regime",
            ),
            secondary_y=False,
        )
    regime_price.update_layout(title="NIFTY Price, Stress Probability, and HMM Regimes", height=520)
    regime_price.update_yaxes(title_text="NIFTY close", secondary_y=False)
    regime_price.update_yaxes(title_text="Stress probability", tickformat=".0%", secondary_y=True, range=[0, 1])

    volatility_columns = [column for column in ["volatility_5d", "volatility_20d", "volatility_60d"] if column in visible]
    volatility = px.line(visible, x="date", y=volatility_columns, title="Rolling Volatility Stack")

    layers = [column for column in ["momentum_20d", "drawdown_60d", "connectedness_60d"] if column in visible]
    stress_layers = px.line(visible, x="date", y=layers, title="Momentum, Drawdown, and Connectedness")

    risk_columns = [column for column in ["india_vix_change", "usd_inr_return", "gold_return"] if column in visible]
    risk = px.line(
        visible.melt(id_vars="date", value_vars=risk_columns, var_name="proxy", value_name="move"),
        x="date",
        y="move",
        color="proxy",
        title="Risk Appetite Proxy Moves",
    )

    correlation_html = ""
    if CORRELATION_PATH.exists():
        correlation = pd.read_csv(CORRELATION_PATH, index_col=0)
        correlation_html = chart_div(
            px.imshow(
                correlation,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Latest 60-Day Cross-Asset Return Correlation",
            )
        )

    driver_rows = []
    for rank in range(1, 9):
        name_col = f"driver_{rank}_name"
        importance_col = f"driver_{rank}_importance"
        if name_col in data and importance_col in data and pd.notna(latest[name_col]):
            driver_rows.append({"feature": str(latest[name_col]).replace("_", " "), "importance": latest[importance_col]})
    drivers = px.bar(
        pd.DataFrame(driver_rows).sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Model Feature Importance",
    )

    cards = "\n".join(
        [
            metric_card("Stress probability", f"{latest['stress_probability']:.1%}", f"{latest['stress_probability'] - previous['stress_probability']:+.1%} vs prior row"),
            metric_card("Warning level", str(latest["warning_level"])),
            metric_card("NIFTY close", f"{latest['nifty_close']:,.2f}", f"{latest['nifty_close'] - previous['nifty_close']:+,.2f}"),
            metric_card("Regime", str(latest.get("regime", "n/a"))),
            metric_card("Connectedness", f"{latest.get('connectedness_60d', 0):.3f}"),
        ]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NIFTY-Sentinel Market Stress Intelligence</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6d7e;
      --line: #d7dbdd;
      --panel: #ffffff;
      --bg: #f5f7fa;
      --accent: #1f618d;
      --stress: #b03a2e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      background: #111827;
      color: white;
      padding: 28px clamp(18px, 4vw, 56px);
      border-bottom: 4px solid var(--accent);
    }}
    header h1 {{ margin: 0 0 8px; font-size: clamp(26px, 4vw, 42px); letter-spacing: 0; }}
    header p {{ max-width: 960px; margin: 0; color: #d5d8dc; line-height: 1.55; }}
    main {{ width: min(1440px, 100%); margin: 0 auto; padding: 24px clamp(12px, 3vw, 32px) 48px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .metric p {{ margin: 0 0 6px; color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; font-size: 24px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; margin-top: 4px; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .note {{ color: var(--muted); line-height: 1.55; }}
    footer {{ color: var(--muted); font-size: 13px; padding: 18px 0 0; }}
    @media (max-width: 980px) {{
      .metrics, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>NIFTY-Sentinel Market Stress Intelligence</h1>
    <p>Static GitHub Pages edition of the explainable early-warning dashboard for Indian market stress. The full Streamlit app remains in the repository for interactive local or Streamlit Cloud deployment.</p>
  </header>
  <main>
    <div class="metrics">{cards}</div>
    <section>{chart_div(probability)}</section>
    <section>{chart_div(regime_price)}</section>
    <div class="grid">
      <section>{chart_div(volatility)}</section>
      <section>{chart_div(stress_layers)}</section>
      <section>{chart_div(risk)}</section>
      <section>{chart_div(drivers)}</section>
    </div>
    <section>{correlation_html}</section>
    <section>
      <h2>Prototype Data Note</h2>
      <p class="note">This page uses exploratory public-market data generated for project demonstration. Replace with validated NSE/RBI files before making final academic performance claims.</p>
      <pre>{json.dumps(metadata, indent=2)}</pre>
    </section>
    <footer>Generated from NIFTY-Sentinel project outputs. For academic use only; not trading advice.</footer>
  </main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
