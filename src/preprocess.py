from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = PROJECT_ROOT / "data" / "raw" / "master_data.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_tourism.csv"


def preprocess(source_csv: Path, output_csv: Path) -> pd.DataFrame:
    if not source_csv.exists():
        raise FileNotFoundError(
            f"Source dataset not found at: {source_csv}\n"
            "Place the file at data/raw/master_data.csv or pass --source-csv <path>."
        )
    df = pd.read_csv(source_csv)

    df["CityName"] = df["CityName"].fillna("Unknown")
    df["Country"] = df["Country"].fillna("Unknown")
    df["Region"] = df["Region"].fillna("Unknown")
    df["Continent"] = df["Continent"].fillna("Unknown")

    numeric_id_cols = [
        "TransactionId",
        "UserId",
        "VisitYear",
        "VisitMonth",
        "VisitModeId",
        "AttractionId",
        "ContinentId",
        "RegionId",
        "CountryId",
        "CityId",
        "AttractionCityId",
        "AttractionTypeId",
    ]

    for col in numeric_id_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(df["Rating"].median())
    df["Rating"] = df["Rating"].clip(lower=1, upper=5)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess tourism master dataset")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    df = preprocess(args.source_csv, args.output_csv)
    print(f"Saved cleaned dataset: {args.output_csv}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
