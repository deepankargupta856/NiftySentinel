"""Import and audit manually downloaded official market-data files.

The project deliberately keeps official-data retrieval separate from data cleaning:
the user downloads NSE/RBI exports, preserves the original file in data/raw/, and
uses these functions to make the column mapping reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

DATE_CANDIDATES = ("date", "timestamp", "trade date", "trading date", "index date")
CLOSE_CANDIDATES = ("close", "closing index value", "closing price", "close price")
VOLUME_CANDIDATES = ("volume", "shares traded", "total traded quantity")


def _canonical_column_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    canonical = {_canonical_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in canonical:
            return canonical[candidate]
    raise ValueError(f"Could not find any of {candidates}. Available columns: {list(frame.columns)}")


def read_official_csv(path: str | Path, series_name: str, include_volume: bool = False) -> pd.DataFrame:
    """Normalize a typical NSE/RBI CSV export into date, series, and volume fields.

    Inspect the source columns before running this function if a provider changes
    its export format. It fails loudly rather than guessing a potentially wrong
    financial field.
    """
    source_path = Path(path)
    frame = pd.read_csv(source_path)
    date_column = _find_column(frame, DATE_CANDIDATES)
    close_column = _find_column(frame, CLOSE_CANDIDATES)

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], dayfirst=True, errors="coerce"),
            series_name: pd.to_numeric(
                frame[close_column].astype(str).str.replace(",", "", regex=False), errors="coerce"
            ),
        }
    )
    if include_volume:
        volume_column = _find_column(frame, VOLUME_CANDIDATES)
        output[f"{series_name}_volume"] = pd.to_numeric(
            frame[volume_column].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )

    return output.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")


def merge_series(series_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join normalized official series and retain data gaps for review."""
    if not series_frames:
        raise ValueError("Provide at least one normalized series frame.")
    master = series_frames[0]
    for frame in series_frames[1:]:
        master = master.merge(frame, on="date", how="outer", validate="one_to_one")
    return master.sort_values("date").reset_index(drop=True)


def quality_report(master: pd.DataFrame) -> pd.DataFrame:
    """Return coverage and missingness metrics for each non-date series."""
    records = []
    for column in master.columns:
        if column == "date":
            continue
        non_null = master.loc[master[column].notna(), "date"]
        records.append(
            {
                "series": column,
                "first_available": non_null.min(),
                "last_available": non_null.max(),
                "missing_pct": round(master[column].isna().mean() * 100, 2),
                "observations": int(master[column].notna().sum()),
            }
        )
    return pd.DataFrame(records).sort_values("series")


def write_official_master(master: pd.DataFrame) -> tuple[Path, Path]:
    """Persist a clean official master file and its quality report."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PROCESSED_DATA_DIR / "official_master_market_data.csv"
    report_path = PROCESSED_DATA_DIR / "official_data_quality.csv"
    master.to_csv(master_path, index=False)
    quality_report(master).to_csv(report_path, index=False)
    return master_path, report_path


def expected_raw_path(filename: str) -> Path:
    """Provide a stable documented location for each manually downloaded file."""
    return RAW_DATA_DIR / filename
