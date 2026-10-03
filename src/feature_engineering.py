"""Feature construction for content similarity and cluster interpretation."""

from __future__ import annotations
import re
import pandas as pd

TEXT_CANDIDATES = ["title", "director", "cast", "country", "listed_in", "description"]


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Combine only available descriptive fields; retain a row per title."""
    cols = [column for column in TEXT_CANDIDATES if column in df.columns]
    if not cols:
        raise ValueError("No supported text columns found in the dataset")
    return df[cols].fillna("").astype("string").agg(" ".join, axis=1)


def clean_text(text: str) -> str:
    """Normalize case and spacing while retaining Unicode letters and digits."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def make_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined_text and normalized_text without dropping source metadata."""
    result = df.copy()
    result["combined_text"] = build_combined_text(result)
    result["normalized_text"] = result["combined_text"].map(clean_text)
    return result


def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable content metadata features when source fields exist."""
    result = df.copy()
    if "date_added" in result:
        result["date_added_parsed"] = pd.to_datetime(result["date_added"], errors="coerce")
        result["year_added"] = result["date_added_parsed"].dt.year.astype("Int64")
    for field, output in [("country", "country_count"), ("listed_in", "category_count"),
                          ("cast", "cast_member_count"), ("director", "director_count")]:
        if field in result:
            result[output] = result[field].fillna("").map(
                lambda value: sum(bool(item.strip()) for item in str(value).split(","))
            )
    if "duration" in result:
        duration = result["duration"].astype("string")
        result["duration_value"] = pd.to_numeric(duration.str.extract(r"(\d+)")[0], errors="coerce")
        result["duration_unit"] = duration.str.extract(r"(min|Season)s?\s*$", expand=False)
    return result
