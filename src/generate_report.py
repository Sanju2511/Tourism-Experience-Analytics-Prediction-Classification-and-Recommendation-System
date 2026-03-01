from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_tourism.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "metrics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "PROJECT_REPORT.md"


def generate_report() -> None:
    df = pd.read_csv(CLEANED_CSV)
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    report = f"""# Tourism Experience Analytics Report

## 1. Dataset Summary
- Total records: {len(df):,}
- Unique users: {df['UserId'].nunique():,}
- Unique attractions: {df['AttractionId'].nunique():,}
- Average rating: {df['Rating'].mean():.3f}
- Most common visit mode: {df['VisitMode'].mode().iat[0]}

## 2. Data Preparation
- Filled missing values in geographical columns (`CityName`, `Country`, `Region`, `Continent`) with `Unknown`.
- Enforced integer types on ID and time fields.
- Clipped ratings into the valid 1-5 range.
- Saved cleaned data to `data/processed/cleaned_tourism.csv`.

## 3. EDA Outputs
Generated the following charts under `reports/figures/`:
- `rating_distribution.png`
- `visit_mode_distribution.png`
- `top_countries.png`
- `ratings_by_visit_mode.png`

## 4. Model Comparison
### Regression (Target: Rating)
Best model: **{metrics['regression']['best_model']}**
- MAE: {metrics['regression']['best_metrics']['mae']:.4f}
- RMSE: {metrics['regression']['best_metrics']['rmse']:.4f}
- R2: {metrics['regression']['best_metrics']['r2']:.4f}

### Classification (Target: VisitModeId)
Best model: **{metrics['classification']['best_model']}**
- Accuracy: {metrics['classification']['best_metrics']['accuracy']:.4f}
- Precision (weighted): {metrics['classification']['best_metrics']['precision_weighted']:.4f}
- Recall (weighted): {metrics['classification']['best_metrics']['recall_weighted']:.4f}
- F1 (weighted): {metrics['classification']['best_metrics']['f1_weighted']:.4f}

### Recommendation
Collaborative filtering with cosine KNN:
- Users in matrix: {metrics['recommendation']['users_in_matrix']:,}
- Items in matrix: {metrics['recommendation']['items_in_matrix']:,}
- HitRate@5: {metrics['recommendation']['hitrate_at_5']:.4f}

## 5. Business Insights
- The model stack supports three required use-cases: rating prediction, visit mode classification, and attraction recommendation.
- Most transactions are high-rated, indicating generally positive tourism experiences.
- Visit mode segmentation can support targeted campaign strategies.

## 6. Streamlit App
The app (`app/streamlit_app.py`) provides:
- Dashboard for trends
- Rating prediction
- Visit mode prediction
- Personalized recommendations
"""

    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Report generated at: {REPORT_MD}")


if __name__ == "__main__":
    generate_report()
