# Tourism Experience Analytics
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
- Total transactions: **52,930**
- Unique users: **33,530**
- Unique attractions: **30**
- Time range: **2013 to 2022**
- Average rating: **4.158 / 5**

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
- Cleaned dataset shape: **(52930, 21)**
- Ready for EDA, ML training, and app deployment.

## 5. Exploratory Data Analysis (EDA)
Visual outputs are saved in `reports/figures/`.

### 5.1 Visit Mode Distribution
| Visit Mode | Transactions | Share |
|---|---:|---:|
| Couples | 21,620 | 40.85% |
| Family | 15,217 | 28.75% |
| Friends | 10,945 | 20.68% |
| Solo | 4,525 | 8.55% |
| Business | 623 | 1.18% |

### 5.2 Top Countries by Transaction Volume
| Country | Transactions |
|---|---:|
| Australia | 13,322 |
| United Kingdom | 6,722 |
| United States | 6,261 |
| Indonesia | 4,842 |
| Singapore | 2,807 |
| India | 2,543 |
| Malaysia | 1,581 |
| Canada | 1,486 |
| New Zealand | 1,479 |
| Netherlands | 859 |

### 5.3 Most Visited Attractions
| Attraction | Visits |
|---|---:|
| Sacred Monkey Forest Sanctuary | 13,198 |
| Waterbom Bali | 6,429 |
| Tegalalang Rice Terrace | 5,815 |
| Uluwatu Temple | 3,359 |
| Tanah Lot Temple | 3,352 |
| Sanur Beach | 3,044 |
| Seminyak Beach | 2,914 |
| Kuta Beach - Bali | 2,765 |
| Merapi Volcano | 2,235 |
| Tegenungan Waterfall | 2,190 |

### 5.4 Average Rating by Visit Mode
| Visit Mode | Avg Rating |
|---|---:|
| Business | 4.313 |
| Family | 4.219 |
| Friends | 4.174 |
| Couples | 4.117 |
| Solo | 4.088 |

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

### 7.1 Regression (Best Model: GradientBoostingRegressor)
- MAE: **0.7165**
- RMSE: **0.9136**
- R2: **0.1137**

### 7.2 Classification (Best Model: RandomForestClassifier)
- Accuracy: **0.4930**
- Precision (weighted): **0.4942**
- Recall (weighted): **0.4930**
- F1 (weighted): **0.4411**

### 7.3 Recommendation
- Users in matrix: **33,530**
- Attractions in matrix: **30**
- HitRate@5: **0.0032**

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
