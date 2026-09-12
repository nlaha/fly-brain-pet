"""
Prints columns, dtypes, and a few sample rows for each downloaded
feather file, so you can fix COLUMN_MAP in brain/connectome.py to
match reality.

Usage: uv run inspect_data.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "raw"


def show(path: Path, n_rows: int = 3):
    if not path.exists():
        print(f"-- {path.name}: not found, skipping --\n")
        return
    df = pd.read_feather(path)
    print(f"-- {path.name} --")
    print(f"shape: {df.shape}")
    print("columns:", list(df.columns))
    print(df.head(n_rows).to_string())
    print()


if __name__ == "__main__":
    for fname in sorted(DATA_DIR.glob("*.feather")):
        show(fname)
