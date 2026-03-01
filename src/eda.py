from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_tourism.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def run_eda(input_csv: Path, fig_dir: Path) -> dict:
    df = pd.read_csv(input_csv)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Rating", color="#1f77b4")
    plt.title("Rating Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "rating_distribution.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    order = df["VisitMode"].value_counts().index
    sns.countplot(data=df, x="VisitMode", order=order, color="#2ca02c")
    plt.title("Visit Mode Distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fig_dir / "visit_mode_distribution.png", dpi=140)
    plt.close()

    top_countries = df["Country"].value_counts().head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top_countries.values, y=top_countries.index, color="#ff7f0e")
    plt.title("Top 10 Countries by Transactions")
    plt.xlabel("Transactions")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_countries.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="VisitMode", y="Rating", color="#9467bd")
    plt.title("Rating by Visit Mode")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fig_dir / "ratings_by_visit_mode.png", dpi=140)
    plt.close()

    insights = {
        "rows": int(df.shape[0]),
        "users": int(df["UserId"].nunique()),
        "attractions": int(df["AttractionId"].nunique()),
        "avg_rating": float(df["Rating"].mean()),
        "most_common_visit_mode": str(df["VisitMode"].mode().iat[0]),
        "top_country": str(df["Country"].mode().iat[0]),
    }
    return insights


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA for tourism dataset")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    insights = run_eda(args.input_csv, args.fig_dir)
    print("EDA insights:")
    for key, value in insights.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
