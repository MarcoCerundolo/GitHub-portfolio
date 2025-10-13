"""Utility to convert Stata .dta files in the data directory to CSV files."""

from pathlib import Path

import pandas as pd


def convert_dta_to_csv(data_dir: Path) -> None:
    """Convert every .dta file in data_dir into a CSV with the same stem."""
    dta_files = sorted(data_dir.glob("*.dta"))
    if not dta_files:
        print(f"No .dta files found in {data_dir}")
        return

    for dta_path in dta_files:
        csv_path = dta_path.with_suffix(".csv")
        print(f"Converting {dta_path.name} -> {csv_path.name}")
        df = pd.read_stata(dta_path)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    convert_dta_to_csv(data_dir)
