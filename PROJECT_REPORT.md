# Tourism Experience Analytics Report

## 1. Dataset Summary
- Total records: 52,930
- Unique users: 33,530
- Unique attractions: 30
- Average rating: 4.158
- Most common visit mode: Couples

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
Best model: **GradientBoostingRegressor**
- MAE: 0.7165
- RMSE: 0.9136
- R2: 0.1137

### Classification (Target: VisitModeId)
Best model: **RandomForestClassifier**
- Accuracy: 0.4930
- Precision (weighted): 0.4942
- Recall (weighted): 0.4930
- F1 (weighted): 0.4411

### Recommendation
Collaborative filtering with cosine KNN:
- Users in matrix: 33,530
- Items in matrix: 30
- HitRate@5: 0.0032

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
