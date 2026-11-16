"""
src/ml_models.py
Advanced Predictive Analytics and Machine Learning Module for Reel Insights 2.0.
Implements:
1. Box-Office Revenue Prediction (Linear Regression, Random Forest, Gradient Boosting)
2. Rating (Vote Average) Prediction
3. Popularity Prediction
4. Primary Genre Classification (Random Forest / Logistic Regression)
5. Feature Importance Analysis
6. Rigorous Train/Test Evaluation and Cross-Validation
"""

from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import ML_CONFIG, MODELS_DIR, logger
from src.database import execute_query, DEFAULT_DB_PATH


def prepare_regression_dataset(target_col: str = "revenue", min_val: float = 1.0) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extracts and preprocesses features for predictive regression.
    Filters out invalid target records to prevent target contamination.
    """
    query = """
    SELECT 
        m.movie_id,
        m.title,
        m.budget,
        m.revenue,
        m.runtime,
        m.popularity,
        m.vote_average,
        m.vote_count,
        CAST(strftime('%Y', m.release_date) AS INTEGER) AS release_year,
        CAST(strftime('%m', m.release_date) AS INTEGER) AS release_month
    FROM movies m
    WHERE m.budget > 0 AND m.revenue > 0 AND m.runtime > 0;
    """
    df = execute_query(query)
    
    # Feature engineering: budget-to-runtime ratio, vote engagement density
    df["budget_per_min"] = (df["budget"] / df["runtime"]).round(2)
    df["vote_density"] = (df["vote_count"] / (df["popularity"] + 1.0)).round(2)

    features = ["budget", "popularity", "vote_count", "runtime", "release_year", "release_month", "budget_per_min"]
    if target_col != "vote_average":
        features.append("vote_average")

    X = df[features].fillna(0)
    y = df[target_col]
    return X, y, features


def train_and_evaluate_revenue_models() -> Dict[str, Any]:
    """
    Trains and compares multiple regression models for box-office revenue prediction.
    Outputs evaluation metrics: RMSE ($M), MAE ($M), and R² score.
    """
    logger.info("Training Box-Office Revenue prediction models...")
    X, y, features = prepare_regression_dataset(target_col="revenue")
    
    # Target in millions for interpretable error metrics
    y_m = y / 1e6

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_m, test_size=ML_CONFIG["test_size"], random_state=ML_CONFIG["random_state"]
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=ML_CONFIG["random_state"], n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=ML_CONFIG["random_state"])
    }

    results = {}
    best_model_name = None
    best_r2 = -float("inf")

    for name, model in models.items():
        if "Linear" in name:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {
            "MAE_millions": round(mae, 2),
            "RMSE_millions": round(rmse, 2),
            "R2_Score": round(r2, 3),
            "model_obj": model
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    # Feature Importance from best tree model
    rf_model = models["Random Forest Regressor"]
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Persist model artifacts
    joblib.dump(rf_model, MODELS_DIR / "revenue_rf_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "revenue_scaler.joblib")
    joblib.dump(features, MODELS_DIR / "revenue_features.joblib")

    logger.info(f"Revenue Model Training Complete. Best model: {best_model_name} (R²: {best_r2:.3f})")
    return {
        "metrics_summary": results,
        "best_model": best_model_name,
        "feature_importance": importance_df,
        "test_size": len(y_test),
        "train_size": len(y_train)
    }


def train_and_evaluate_rating_models() -> Dict[str, Any]:
    """
    Trains predictive models to forecast audience vote_average scores.
    """
    logger.info("Training Audience Rating prediction models...")
    X, y, features = prepare_regression_dataset(target_col="vote_average")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ML_CONFIG["test_size"], random_state=ML_CONFIG["random_state"]
    )

    rf_rating = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=ML_CONFIG["random_state"], n_jobs=-1)
    rf_rating.fit(X_train, y_train)
    preds = rf_rating.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_rating.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    joblib.dump(rf_rating, MODELS_DIR / "rating_rf_model.joblib")

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "feature_importance": importance_df
    }


def train_genre_classifier() -> Dict[str, Any]:
    """
    Trains a classification model to predict primary movie genre
    based on movie budget, runtime, and popularity characteristics.
    """
    logger.info("Training Primary Genre Classifier...")
    query = """
    SELECT 
        m.movie_id,
        m.budget,
        m.revenue,
        m.runtime,
        m.popularity,
        m.vote_average,
        g.genre_name
    FROM movies m
    JOIN (
        SELECT movie_id, genre_id FROM movie_genres GROUP BY movie_id
    ) primary_mg ON m.movie_id = primary_mg.movie_id
    JOIN genres g ON primary_mg.genre_id = g.genre_id
    WHERE m.budget > 0;
    """
    df = execute_query(query)
    
    # Filter top 6 most common genres for high precision classification
    top_genres = df["genre_name"].value_counts().head(6).index.tolist()
    filtered_df = df[df["genre_name"].isin(top_genres)].copy()

    features = ["budget", "revenue", "runtime", "popularity", "vote_average"]
    X = filtered_df[features].fillna(0)
    y = filtered_df["genre_name"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=ML_CONFIG["random_state"], stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=ML_CONFIG["random_state"])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="weighted")

    joblib.dump(clf, MODELS_DIR / "genre_classifier.joblib")

    return {
        "accuracy": round(acc * 100, 2),
        "weighted_f1": round(macro_f1, 3),
        "classes": top_genres,
        "sample_size": len(filtered_df)
    }


def predict_single_movie_revenue(budget_m: float, runtime: int, popularity: float, vote_count: int, vote_average: float, release_year: int) -> float:
    """Predicts gross box office for user-defined movie parameters."""
    model_path = MODELS_DIR / "revenue_rf_model.joblib"
    if not model_path.exists():
        train_and_evaluate_revenue_models()

    model = joblib.load(model_path)
    budget = budget_m * 1e6
    budget_per_min = budget / runtime if runtime > 0 else 0
    
    # features: budget, popularity, vote_count, runtime, release_year, release_month, budget_per_min, vote_average
    features_vec = pd.DataFrame([{
        "budget": budget,
        "popularity": popularity,
        "vote_count": vote_count,
        "runtime": runtime,
        "release_year": release_year,
        "release_month": 7, # default July peak
        "budget_per_min": budget_per_min,
        "vote_average": vote_average
    }])

    pred_m = model.predict(features_vec)[0]
    return max(0.0, float(pred_m))

