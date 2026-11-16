"""
tests/test_ml_models.py
Tests for machine learning models and recommendation engine.
"""

import pytest
import numpy as np
import pandas as pd
from src.ml_models import (
    prepare_regression_dataset,
    train_and_evaluate_revenue_models,
    train_and_evaluate_rating_models,
    predict_single_movie_revenue
)
from src.recommendations import get_recommendation_engine


def test_prepare_regression_dataset_no_leakage():
    X, y, features = prepare_regression_dataset(target_col="revenue")
    assert len(X) == len(y)
    assert "revenue" not in features, "Target leakage detected: revenue present in features!"
    assert (X["budget"] > 0).all()


def test_revenue_prediction_models():
    res = train_and_evaluate_revenue_models()
    assert "best_model" in res
    assert "metrics_summary" in res
    assert res["metrics_summary"][res["best_model"]]["R2_Score"] > 0.40
    assert len(res["feature_importance"]) > 0


def test_single_movie_prediction():
    # Avatar parameters: Budget 237M, runtime 162, pop 185, votes 29000, rating 7.6, 2009
    pred_rev = predict_single_movie_revenue(
        budget_m=237.0, runtime=162, popularity=185.0, vote_count=29000, vote_average=7.6, release_year=2009
    )
    assert pred_rev > 500.0, f"Expected high blockbuster prediction, got: {pred_rev}"


def test_recommendation_engine():
    engine = get_recommendation_engine()
    recs = engine.recommend("Interstellar", top_n=5)
    assert len(recs) == 5
    for r in recs:
        assert "title" in r
        assert "similarity_score" in r
        assert r["similarity_score"] > 0.0

