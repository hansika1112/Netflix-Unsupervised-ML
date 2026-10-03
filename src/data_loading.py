"""Dataset loading helpers for the Netflix clustering project."""

from pathlib import Path
import pandas as pd


def load_titles(path: str | Path) -> pd.DataFrame:
    """Load the supplied Netflix titles CSV without assuming a fixed schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)
