"""
tests/test_cleaning.py
Unit tests for data cleaning and financial handling.
"""

import pytest
import numpy as np
import pandas as pd
from src.cleaning import (
    clean_movies,
    clean_genres,
    clean_movie_genres,
    clean_cast,
    clean_crew,
    clean_movie_keywords,
    compute_financial_metrics
)


def test_clean_movies_deduplication():
    # Test movie deduplication on movie_id
    raw_df = pd.DataFrame({
        "movie_id": [1, 2, 1],
        "title": ["Movie A", "Movie B", "Movie A Duplicate"],
        "release_date": ["2020-01-01", "2021-05-15", "2020-01-01"],
        "budget": [1000000.0, 0.0, 1000000.0],
        "revenue": [5000000.0, 0.0, 5000000.0],
        "runtime": [120, 95, 120],
        "popularity": [15.2, 5.0, 15.2],
        "vote_average": [7.5, 6.2, 7.5],
        "vote_count": [100, 20, 100],
        "original_language": ["en", "fr", "en"],
        "status": ["Released", "Released", "Released"],
        "overview": ["Test 1", "Test 2", "Test 1"]
    })
    cleaned = clean_movies(raw_df)
    assert len(cleaned) == 2
    assert cleaned["movie_id"].tolist() == [1, 2]
    assert cleaned["release_year"].tolist() == [2020, 2021]


def test_financial_zero_handling_and_roi():
    # Test that 0 budget and 0 revenue are preserved and ROI handles division by zero safely
    df = pd.DataFrame({
        "movie_id": [10, 20, 30],
        "title": ["Normal", "Zero Budget", "Zero Revenue"],
        "budget": [10_000_000.0, 0.0, 5_000_000.0],
        "revenue": [30_000_000.0, 10_000_000.0, 0.0]
    })
    fin_df = compute_financial_metrics(df)
    
    # Movie 10: 30M rev, 10M bud -> profit 20M, ROI 200%
    assert fin_df.loc[0, "profit"] == 20_000_000.0
    assert fin_df.loc[0, "roi_percentage"] == 200.0
    assert fin_df.loc[0, "is_profitable"] == True

    # Movie 20 (zero budget): ROI must be NaN, NOT infinite or 0
    assert np.isnan(fin_df.loc[1, "profit"])
    assert np.isnan(fin_df.loc[1, "roi_percentage"])

    # Movie 30 (zero revenue): ROI must be NaN, NOT -100%
    assert np.isnan(fin_df.loc[2, "profit"])
    assert np.isnan(fin_df.loc[2, "roi_percentage"])


def test_orphan_filtering():
    # Test that child records referencing non-existent movie_id are dropped
    valid_movie_ids = {100, 200}
    valid_genre_ids = {1, 2}

    raw_mg = pd.DataFrame({
        "movie_id": [100, 200, 999], # 999 is orphan
        "genre_id": [1, 2, 1]
    })
    cleaned_mg = clean_movie_genres(raw_mg, valid_movie_ids, valid_genre_ids)
    assert len(cleaned_mg) == 2
    assert 999 not in cleaned_mg["movie_id"].values

