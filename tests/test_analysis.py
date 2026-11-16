"""
tests/test_analysis.py
Tests for all 10 Pandas analytical questions and core metrics.
"""

import pytest
import pandas as pd
from src.cleaning import execute_full_cleaning_pipeline
from src.analysis import (
    pandas_q1_entity_counts,
    pandas_q2_missing_financials,
    pandas_q3_profit_and_roi,
    pandas_q4_top_15_roi,
    pandas_q5_releases_per_year,
    pandas_q6_runtime_outliers,
    pandas_q7_primary_genre_popularity,
    pandas_q8_prolific_actors,
    pandas_q9_duplicate_titles_check,
    pandas_q10_popularity_vote_divergence,
    compute_comprehensive_kpis
)


@pytest.fixture(scope="module")
def cleaned_tables():
    return execute_full_cleaning_pipeline()


def test_pandas_q1_entity_counts(cleaned_tables):
    counts = pandas_q1_entity_counts(cleaned_tables)
    assert counts["unique_movies"] == 2503
    assert counts["unique_genres"] == 19
    assert counts["unique_cast_members"] > 10000
    assert counts["unique_crew_members"] > 2000


def test_pandas_q2_missing_financials(cleaned_tables):
    df_missing, summary = pandas_q2_missing_financials(cleaned_tables["movies"])
    assert summary["zero_budget_count"] == 90
    assert summary["zero_revenue_count"] == 102
    assert summary["both_zero_count"] == 59
    assert len(df_missing) == 133


def test_pandas_q3_profit_and_roi(cleaned_tables):
    roi_df = pandas_q3_profit_and_roi(cleaned_tables["movies"])
    assert len(roi_df) == 2370
    assert (roi_df["budget"] > 0).all()
    assert (roi_df["revenue"] > 0).all()
    assert (roi_df["profit"] == roi_df["revenue"] - roi_df["budget"]).all()


def test_pandas_q4_top_15_roi(cleaned_tables):
    top15 = pandas_q4_top_15_roi(cleaned_tables["movies"], min_budget=1_000_000)
    assert len(top15) == 15
    assert (top15["budget"] >= 1_000_000).all()
    assert top15["roi_percentage"].is_monotonic_decreasing


def test_pandas_q5_releases_per_year(cleaned_tables):
    yearly, summary = pandas_q5_releases_per_year(cleaned_tables["movies"])
    assert summary["peak_year"] == 2016
    assert summary["peak_movie_count"] == 121
    assert len(yearly) > 50


def test_pandas_q6_runtime_outliers(cleaned_tables):
    outliers, bounds = pandas_q6_runtime_outliers(cleaned_tables["movies"])
    assert bounds["outlier_count"] == 60
    assert bounds["lower_bound"] == 58.5
    assert bounds["upper_bound"] == 166.5


def test_pandas_q7_genre_popularity(cleaned_tables):
    genre_pop = pandas_q7_primary_genre_popularity(
        cleaned_tables["movies"], cleaned_tables["movie_genres"], cleaned_tables["genres"]
    )
    assert len(genre_pop) == 19
    assert "avg_popularity" in genre_pop.columns


def test_pandas_q8_prolific_actors(cleaned_tables):
    top_actors = pandas_q8_prolific_actors(cleaned_tables["cast"], cleaned_tables["movies"], top_n=10)
    assert len(top_actors) == 10
    assert top_actors.iloc[0]["actor_name"] == "Samuel L. Jackson"


def test_pandas_q9_duplicate_titles(cleaned_tables):
    dups = pandas_q9_duplicate_titles_check(cleaned_tables["movies"])
    assert len(dups) == 100


def test_pandas_q10_divergence(cleaned_tables):
    div_df = pandas_q10_popularity_vote_divergence(cleaned_tables["movies"], top_n=15)
    assert len(div_df) == 15
    assert "divergence" in div_df.columns

