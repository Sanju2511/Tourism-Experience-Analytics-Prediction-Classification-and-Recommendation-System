from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_tourism.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_recommendation_hitrate(df: pd.DataFrame, k: int = 5) -> float:
    user_counts = df["UserId"].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    df_valid = df[df["UserId"].isin(valid_users)].copy()

    heldout = df_valid.groupby("UserId", as_index=False).tail(1)
    train_df = df_valid.drop(index=heldout.index)

    user_item = train_df.pivot_table(index="UserId", columns="AttractionId", values="Rating", aggfunc="mean", fill_value=0)
    if user_item.shape[0] < 2:
        return 0.0

    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(6, len(user_item)))
    knn.fit(user_item.values)

    hits = 0
    total = 0

    for _, row in heldout.iterrows():
        user_id = row["UserId"]
        true_item = row["AttractionId"]
        if user_id not in user_item.index:
            continue

        user_vec = user_item.loc[user_id].values.reshape(1, -1)
        _, idxs = knn.kneighbors(user_vec, n_neighbors=min(6, len(user_item)))
        neighbor_users = user_item.index[idxs[0][1:]]

        seen_items = set(train_df.loc[train_df["UserId"] == user_id, "AttractionId"])
        neighbor_scores = (
            train_df[train_df["UserId"].isin(neighbor_users)]
            .groupby("AttractionId")["Rating"]
            .mean()
            .sort_values(ascending=False)
        )
        recs = [item for item in neighbor_scores.index if item not in seen_items][:k]

        total += 1
        if true_item in recs:
            hits += 1

    return float(hits / total) if total else 0.0


def train_models(input_csv: Path, models_dir: Path, reports_dir: Path) -> dict:
    df = pd.read_csv(input_csv)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    reg_feature_cols = [
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

    clf_feature_cols = [
        "VisitYear",
        "VisitMonth",
        "AttractionId",
        "ContinentId",
        "RegionId",
        "CountryId",
        "CityId",
        "AttractionCityId",
        "AttractionTypeId",
    ]

    X_reg = df[reg_feature_cols]
    X_clf = df[clf_feature_cols]

    # Regression: predict rating
    y_reg = df["Rating"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=120, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }

    reg_results = {}
    best_reg_name = None
    best_reg_rmse = np.inf
    best_reg_model = None

    for name, model in reg_models.items():
        model.fit(Xr_train, yr_train)
        pred = model.predict(Xr_test)
        rmse = mean_squared_error(yr_test, pred) ** 0.5
        result = {
            "mae": float(mean_absolute_error(yr_test, pred)),
            "rmse": float(rmse),
            "r2": float(r2_score(yr_test, pred)),
        }
        reg_results[name] = result
        if rmse < best_reg_rmse:
            best_reg_rmse = rmse
            best_reg_name = name
            best_reg_model = model

    # Classification: predict visit mode
    y_clf = df["VisitModeId"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    clf_models = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=60, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    }

    clf_results = {}
    best_clf_name = None
    best_clf_f1 = -1.0
    best_clf_model = None

    for name, model in clf_models.items():
        model.fit(Xc_train, yc_train)
        pred = model.predict(Xc_test)
        result = {
            "accuracy": float(accuracy_score(yc_test, pred)),
            "precision_weighted": float(precision_score(yc_test, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(yc_test, pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(yc_test, pred, average="weighted", zero_division=0)),
        }
        clf_results[name] = result
        if result["f1_weighted"] > best_clf_f1:
            best_clf_f1 = result["f1_weighted"]
            best_clf_name = name
            best_clf_model = model

    # Recommendation model
    user_item = df.pivot_table(index="UserId", columns="AttractionId", values="Rating", aggfunc="mean", fill_value=0)
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(6, len(user_item)))
    knn.fit(user_item.values)

    hitrate_at_5 = evaluate_recommendation_hitrate(df, k=5)

    # Save artifacts
    joblib.dump(best_reg_model, models_dir / "best_regression_model.pkl")
    joblib.dump(best_clf_model, models_dir / "best_classification_model.pkl")
    joblib.dump(knn, models_dir / "recommendation_knn.pkl")
    joblib.dump(user_item, models_dir / "user_item_matrix.pkl")
    joblib.dump(
        {"regression": reg_feature_cols, "classification": clf_feature_cols},
        models_dir / "model_features.pkl",
    )

    metrics = {
        "regression": {
            "all_models": reg_results,
            "best_model": best_reg_name,
            "best_metrics": reg_results[best_reg_name],
        },
        "classification": {
            "all_models": clf_results,
            "best_model": best_clf_name,
            "best_metrics": clf_results[best_clf_name],
        },
        "recommendation": {
            "users_in_matrix": int(user_item.shape[0]),
            "items_in_matrix": int(user_item.shape[1]),
            "hitrate_at_5": float(hitrate_at_5),
        },
    }

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tourism models")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    metrics = train_models(args.input_csv, args.models_dir, args.reports_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
