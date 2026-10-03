"""Conservative cleaning and schema inspection for Netflix catalog records."""

from __future__ import annotations
import pandas as pd


def inspect_schema(df: pd.DataFrame) -> dict:
    """Return actual-schema diagnostics, including data dictionary candidates."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_count": df.isna().sum().to_dict(),
        "missing_percent": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_count": df.nunique(dropna=False).to_dict(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "text_columns": df.select_dtypes(include=["str", "object"]).columns.tolist(),
    }


def clean_titles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply conservative, documented cleaning and return impact diagnostics.

    Missing descriptive metadata is preserved; removing those rows would discard
    useful content. Only exact duplicate records are removed. Text is stripped
    and whitespace-normalized without changing meaningful case or punctuation.
    """
    cleaned = df.copy()
    before_rows = len(cleaned)
    before_duplicates = int(cleaned.duplicated().sum())
    before_missing = cleaned.isna().sum()
    normalized_cells = {}

    # Preserve the source schema; normalize whitespace in string-like fields.
    for col in cleaned.select_dtypes(include=["str", "object"]).columns:
        original = cleaned[col].copy()
        cleaned[col] = cleaned[col].map(
            lambda value: " ".join(value.split()) if isinstance(value, str) else value
        )
        changed = original.notna() & cleaned[col].notna() & cleaned[col].ne(original)
        normalized_cells[col] = int(changed.fillna(False).sum())

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    after_missing = cleaned.isna().sum()
    impact = pd.DataFrame({
        "missing_before": before_missing,
        "missing_after": after_missing,
        "missing_pct_before": (before_missing / max(before_rows, 1) * 100).round(2),
        "missing_pct_after": (after_missing / max(len(cleaned), 1) * 100).round(2),
        "whitespace_normalized_cells": pd.Series(normalized_cells),
    })
    return cleaned, impact
