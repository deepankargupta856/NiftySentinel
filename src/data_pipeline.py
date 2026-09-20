"""Create a canonical, traceable daily market dataset from approved inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def _read_dated_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    if "date" not in frame.columns:
        raise ValueError(f"{path} must contain a date column.")
    return frame.drop_duplicates("date").set_index("date").sort_index()


def combine_preferred_sources(
    official_path: str | Path | None = None,
    exploratory_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer official values over exploratory values and expose field-level provenance.

    The returned provenance table records the chosen source for every populated
    field and date. It does not impute missing market observations.
    """
    official_path = Path(official_path or PROCESSED_DATA_DIR / "official_master_market_data.csv")
    exploratory_path = Path(exploratory_path or RAW_DATA_DIR / "exploratory_market_prices.csv")

    official = _read_dated_csv(official_path) if official_path.exists() else pd.DataFrame()
    exploratory = _read_dated_csv(exploratory_path) if exploratory_path.exists() else pd.DataFrame()
    if official.empty and exploratory.empty:
        raise FileNotFoundError("No official or exploratory market-data file was found.")

    all_columns = sorted(set(official.columns).union(exploratory.columns))
    full_index = official.index.union(exploratory.index).sort_values()
    canonical = pd.DataFrame(index=full_index)
    provenance = pd.DataFrame(index=full_index)

    for column in all_columns:
        official_values = official[column].reindex(full_index) if column in official else pd.Series(index=full_index, dtype="float64")
        exploratory_values = exploratory[column].reindex(full_index) if column in exploratory else pd.Series(index=full_index, dtype="float64")
        canonical[column] = official_values.combine_first(exploratory_values)
        provenance[column] = pd.Series(pd.NA, index=full_index, dtype="string")
        provenance.loc[exploratory_values.notna(), column] = "exploratory"
        provenance.loc[official_values.notna(), column] = "official"

    canonical.index.name = "date"
    provenance.index.name = "date"
    return canonical, provenance


def save_canonical_data(canonical: pd.DataFrame, provenance: pd.DataFrame) -> tuple[Path, Path]:
    """Save the analysis dataset and a companion field-level source ledger."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = PROCESSED_DATA_DIR / "canonical_market_data.csv"
    source_path = PROCESSED_DATA_DIR / "canonical_data_provenance.csv"
    canonical.reset_index().to_csv(data_path, index=False)
    provenance.reset_index().to_csv(source_path, index=False)
    return data_path, source_path
