"""Dataset utilities for the movie recommendation project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["title", "genres", "keywords", "overview"]


def load_movies(csv_path: str | Path) -> pd.DataFrame:
    """Load and normalize a movie dataset from a CSV file."""
    frame = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    normalized = frame.copy()
    normalized["title"] = normalized["title"].fillna("").astype(str).str.strip()

    for column in ("genres", "keywords", "overview"):
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    normalized = normalized.loc[normalized["title"] != ""].reset_index(drop=True)
    return normalized
