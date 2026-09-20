# NIFTY-Sentinel

NIFTY-Sentinel is an explainable early-warning system for Indian financial-market stress. It estimates the probability of elevated market stress over a forward horizon; it does not predict an exact NIFTY price or execute trades.

## Project structure

```text
notebooks/      Google Colab notebooks, run in numerical order
src/            Reusable data and feature-engineering code
data/raw/       Downloaded source data (not committed)
data/processed/ Clean daily data and model-ready feature sets
models/         Saved fitted models and metadata
figures/        Charts exported for the report and presentation
website/        Streamlit dashboard assets
```

## Colab quick start

1. Upload this folder to GitHub or Google Drive.
2. Open `notebooks/00_colab_setup.ipynb` in Google Colab.
3. Mount Drive, set `PROJECT_ROOT`, then run the cells in order.
4. Run `01_data_collection.ipynb` for exploratory data, then `01b_official_data_import.ipynb` for official exports.
5. Run `02a_build_canonical_dataset.ipynb`, then continue in numerical order. Run `04b_ablation_and_shap.ipynb` after the baseline model and before the dashboard. Each notebook writes durable outputs to Drive.

## Data policy

Use official NSE historical reports for NIFTY indices and India VIX, and RBI DBIE for Indian government-security yields where possible. Supplementary sources can speed up exploratory work but must be documented in `data/source_register.csv` before final analysis. Every feature must use information available on or before its date.

## Dashboard

The Streamlit app reads `data/processed/dashboard_data.csv`. Until the modeling notebooks create that file, it intentionally shows an empty-state message rather than simulated market warnings.

```bash
pip install -r requirements-colab.txt
streamlit run app.py
```
