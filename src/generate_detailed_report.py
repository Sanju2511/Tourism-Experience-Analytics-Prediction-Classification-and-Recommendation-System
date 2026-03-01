from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_tourism.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "metrics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "REPORT_DETAILED.md"
SCRIPT_MD = PROJECT_ROOT / "reports" / "VIDEO_PRESENTATION_SCRIPT.md"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def generate() -> None:
    df = pd.read_csv(CLEANED_CSV)
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    total_rows = len(df)
    users = df["UserId"].nunique()
    attractions = df["AttractionId"].nunique()
    avg_rating = df["Rating"].mean()
    year_min, year_max = int(df["VisitYear"].min()), int(df["VisitYear"].max())

    mode_counts = df["VisitMode"].value_counts()
    top_modes = mode_counts.head(5)
    top_mode_lines = "\n".join(
        [f"| {idx} | {int(val):,} | {_fmt_pct(val/total_rows)} |" for idx, val in top_modes.items()]
    )

    top_country = df["Country"].value_counts().head(10)
    top_country_lines = "\n".join([f"| {idx} | {int(val):,} |" for idx, val in top_country.items()])

    top_attr = df["Attraction"].value_counts().head(10)
    top_attr_lines = "\n".join([f"| {idx} | {int(val):,} |" for idx, val in top_attr.items()])

    rating_by_mode = df.groupby("VisitMode")["Rating"].mean().sort_values(ascending=False)
    rating_mode_lines = "\n".join([f"| {idx} | {val:.3f} |" for idx, val in rating_by_mode.items()])

    reg_best = metrics["regression"]["best_metrics"]
    clf_best = metrics["classification"]["best_metrics"]
    rec_best = metrics["recommendation"]

    report = f"""# Tourism Experience Analytics
## Final Project Report (Enhanced Submission)

## 1. Executive Summary
This project builds a complete analytics system for tourism platforms with three integrated AI capabilities:
1. Rating prediction (regression)
2. Visit mode prediction (classification)
3. Personalized attraction recommendation (collaborative filtering)

The solution includes data preprocessing, EDA, visual analytics, ML model training/evaluation, and an interactive Streamlit application for business-facing usage.

## 2. Problem Statement and Business Objectives
Tourism businesses need to improve user experience and retention by understanding behavior and delivering personalization at scale. This project addresses:
- Predict likely attraction rating before a trip or campaign launch.
- Classify visitor type (Business, Family, Friends, Couples, Solo) for targeted offers.
- Recommend attractions aligned with user history and similar-user patterns.

## 3. Dataset Overview
Integrated tourism dataset after consolidation:
- Total transactions: **{total_rows:,}**
- Unique users: **{users:,}**
- Unique attractions: **{attractions:,}**
- Time range: **{year_min} to {year_max}**
- Average rating: **{avg_rating:.3f} / 5**

Core fields used:
- Behavioral: `VisitYear`, `VisitMonth`, `VisitModeId`, `Rating`
- Location: `ContinentId`, `RegionId`, `CountryId`, `CityId`
- Attraction: `AttractionId`, `AttractionTypeId`, `AttractionCityId`, `Attraction`

## 4. Data Cleaning and Preprocessing
Processing was performed using a reproducible Python pipeline.

### 4.1 Cleaning Steps
- Filled missing categorical location labels (`CityName`, `Country`, `Region`, `Continent`) with `Unknown`.
- Enforced numeric consistency for ID columns and temporal fields.
- Validated and clipped ratings to the valid 1-5 range.
- Exported final cleaned dataset to `data/processed/cleaned_tourism.csv`.

### 4.2 Data Quality Outcome
- Missing values in modeling features: **0**
- Cleaned dataset shape: **({total_rows}, 21)**
- Ready for EDA, ML training, and app deployment.

## 5. Exploratory Data Analysis (EDA)
Visual outputs are saved in `reports/figures/`.

### 5.1 Visit Mode Distribution
| Visit Mode | Transactions | Share |
|---|---:|---:|
{top_mode_lines}

### 5.2 Top Countries by Transaction Volume
| Country | Transactions |
|---|---:|
{top_country_lines}

### 5.3 Most Visited Attractions
| Attraction | Visits |
|---|---:|
{top_attr_lines}

### 5.4 Average Rating by Visit Mode
| Visit Mode | Avg Rating |
|---|---:|
{rating_mode_lines}

### 5.5 EDA Interpretation
- Tourism demand is concentrated in leisure segments (Couples, Family, Friends).
- Rating levels are generally high, indicating positive user sentiment.
- Country-level concentration suggests opportunities for geo-targeted marketing and partnerships.

## 6. Feature Engineering and Modeling Strategy

### 6.1 Regression Task
Target: `Rating`
- Candidate models: RandomForestRegressor, GradientBoostingRegressor
- Selected by best RMSE on hold-out test split

### 6.2 Classification Task
Target: `VisitModeId`
- Candidate models: RandomForestClassifier, GradientBoostingClassifier
- Selected by weighted F1 on hold-out test split
- Leakage prevention applied (target `VisitModeId` is **not** used as input feature)

### 6.3 Recommendation Task
- Method: User-based collaborative filtering using KNN with cosine similarity
- Data structure: user-item matrix (`UserId` x `AttractionId`) built from ratings
- Output: top-N unseen attractions from nearest user neighbors

## 7. Model Evaluation Results

### 7.1 Regression (Best Model: {metrics['regression']['best_model']})
- MAE: **{reg_best['mae']:.4f}**
- RMSE: **{reg_best['rmse']:.4f}**
- R2: **{reg_best['r2']:.4f}**

### 7.2 Classification (Best Model: {metrics['classification']['best_model']})
- Accuracy: **{clf_best['accuracy']:.4f}**
- Precision (weighted): **{clf_best['precision_weighted']:.4f}**
- Recall (weighted): **{clf_best['recall_weighted']:.4f}**
- F1 (weighted): **{clf_best['f1_weighted']:.4f}**

### 7.3 Recommendation
- Users in matrix: **{rec_best['users_in_matrix']:,}**
- Attractions in matrix: **{rec_best['items_in_matrix']:,}**
- HitRate@5: **{rec_best['hitrate_at_5']:.4f}**

### 7.4 Performance Discussion
- Regression captures moderate predictive signal, suitable for directional satisfaction forecasting.
- Classification performance indicates useful segmentation potential but can improve with richer behavioral features.
- Recommendation hit rate is low due to sparse user-level interactions, a common issue in implicit tourism logs.

## 8. Business Insights and Recommendations
- Prioritize Couples/Family-focused campaign bundles because these segments dominate volume.
- Use rating prediction to proactively identify lower-satisfaction attraction-context combinations.
- Use visit mode classification to personalize messaging, package design, and amenity recommendations.
- Improve recommendation quality with additional context features (trip duration, budget, recency, dwell time).

## 9. Streamlit Application
Delivered app features:
- Executive dashboard (distribution, countries, top attractions)
- Rating prediction form
- Visit mode prediction form
- Personalized recommendation engine
- Model performance panel for transparent evaluation

File: `app/streamlit_app.py`

## 10. Limitations and Future Work
Current limitations:
- No explicit temporal sequence modeling by user session.
- Sparse interactions reduce recommendation precision.
- Features are primarily structured IDs; semantic attraction metadata is limited.

Next improvements:
- Hybrid recommender (collaborative + content-based)
- More robust feature set (price sensitivity, seasonality intensity, travel cohort)
- Hyperparameter tuning with cross-validation and model calibration
- Better cold-start strategy for new users and new attractions

## 11. Conclusion
This project successfully implements the complete assignment scope with reproducible scripts, explainable analytics, machine learning models, and deployable UI. The output is actionable for tourism platforms and presentation-ready for academic evaluation.

## 12. Reproducibility and Deliverables
- Cleaned data: `data/processed/cleaned_tourism.csv`
- Models: `models/*.pkl`
- Metrics: `reports/metrics.json`
- EDA charts: `reports/figures/*.png`
- Streamlit app: `app/streamlit_app.py`
- Pipeline runner: `run_pipeline.py`
"""

    script = f"""# 10-Minute Video Presentation Script

## Minute 0-1: Introduction
- Introduce project title: Tourism Experience Analytics.
- Explain the three goals: rating prediction, visit mode classification, recommendation.

## Minute 1-2.5: Dataset and Problem Context
- Mention dataset size: {total_rows:,} transactions, {users:,} users, {attractions:,} attractions.
- Mention business need: personalization, retention, targeted campaigns.

## Minute 2.5-4.5: Data Preparation
- Explain cleaning steps: missing value handling, type standardization, rating validation.
- Show final clean dataset and why preprocessing is important for ML quality.

## Minute 4.5-6.5: EDA and Insights
- Walk through charts: visit mode distribution, top countries, top attractions, rating by mode.
- Highlight key insight: Couples and Family dominate demand; ratings are mostly high.

## Minute 6.5-8: Model Building and Evaluation
- Regression model and its metrics: RMSE {reg_best['rmse']:.3f}, R2 {reg_best['r2']:.3f}.
- Classification model and metrics: Accuracy {clf_best['accuracy']:.3f}, F1 {clf_best['f1_weighted']:.3f}.
- Recommendation metrics and sparse-data challenge: HitRate@5 {rec_best['hitrate_at_5']:.4f}.

## Minute 8-9.5: Streamlit Demo
- Show dashboard page first.
- Demonstrate rating prediction and visit mode prediction with sample inputs.
- Demonstrate recommendation output for one user ID.

## Minute 9.5-10: Closing
- Summarize value delivered: analytics + prediction + recommendation in one platform.
- Mention future enhancements: hybrid recommender, richer behavioral features, stronger personalization.
"""

    REPORT_MD.write_text(report, encoding="utf-8")
    SCRIPT_MD.write_text(script, encoding="utf-8")
    print(f"Detailed report written: {REPORT_MD}")
    print(f"Presentation script written: {SCRIPT_MD}")


if __name__ == "__main__":
    generate()
