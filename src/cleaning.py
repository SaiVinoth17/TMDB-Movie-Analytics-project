"""
src/cleaning.py
Data cleaning, anomaly detection, type casting, and relational integrity enforcement
for the Reel Insights 2.0 TMDB Movie Intelligence Platform.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import DATA_DIR, RAW_DATA_PATHS, logger


def load_all_raw_datasets(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load all six authentic TMDB raw CSV datasets with validation."""
    logger.info(f"Loading raw datasets from directory: {data_dir.resolve()}")
    dfs = {}
    for name, path in RAW_DATA_PATHS.items():
        resolved_path = data_dir / path.name
        if not resolved_path.exists():
            err_msg = f"Missing required dataset file: {resolved_path}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        dfs[name] = pd.read_csv(resolved_path)
        logger.info(f"Loaded '{name}': {dfs[name].shape[0]:,} rows x {dfs[name].shape[1]} columns")
    return dfs


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans movies dataset idempotently:
    - Removes exact duplicate records (subset=['movie_id'])
    - Enforces correct data types
    - Cleans whitespace and string anomalies
    - Preserves 0 budget/revenue values as unknown placeholders (does not invent figures)
    - Adds derived analytical columns (release_year, release_month, release_decade)
    """
    logger.info(f"Cleaning movies: {len(df):,} raw records input")
    cleaned = df.copy()

    # Drop exact duplicates or duplicate movie_ids keeping first occurrence
    cleaned = cleaned.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)

    # Clean text columns
    text_cols = ["title", "original_language", "status", "overview"]
    for col in text_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace(["nan", "None", ""], "Unknown")

    # Parse release date with coerced errors
    cleaned["release_date"] = pd.to_datetime(cleaned["release_date"], errors="coerce")
    
    # Extract temporal features
    cleaned["release_year"] = cleaned["release_date"].dt.year
    cleaned["release_month"] = cleaned["release_date"].dt.month
    cleaned["release_decade"] = (cleaned["release_year"] // 10 * 10).fillna(0).astype(int).astype(str) + "s"

    # Enforce numeric types
    cleaned["movie_id"] = cleaned["movie_id"].astype(int)
    cleaned["runtime"] = pd.to_numeric(cleaned["runtime"], errors="coerce").fillna(0).astype(int)
    cleaned["budget"] = pd.to_numeric(cleaned["budget"], errors="coerce").fillna(0).astype(float)
    cleaned["revenue"] = pd.to_numeric(cleaned["revenue"], errors="coerce").fillna(0).astype(float)
    cleaned["popularity"] = pd.to_numeric(cleaned["popularity"], errors="coerce").fillna(0.0).round(3)
    cleaned["vote_average"] = pd.to_numeric(cleaned["vote_average"], errors="coerce").fillna(0.0).round(2)
    cleaned["vote_count"] = pd.to_numeric(cleaned["vote_count"], errors="coerce").fillna(0).astype(int)

    # Format release_date back to ISO string for SQLite storage
    cleaned["release_date_str"] = cleaned["release_date"].dt.strftime("%Y-%m-%d")

    logger.info(f"Movies cleaning completed: {len(cleaned):,} unique records retained")
    return cleaned


def clean_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans genres lookup table."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["genre_id"]).reset_index(drop=True)
    cleaned["genre_id"] = cleaned["genre_id"].astype(int)
    cleaned["genre_name"] = cleaned["genre_name"].astype(str).str.strip()
    return cleaned


def clean_movie_genres(df: pd.DataFrame, valid_movie_ids: set, valid_genre_ids: set) -> pd.DataFrame:
    """Cleans movie_genres junction table and removes duplicates & orphans."""
    cleaned = df.copy()
    cleaned["movie_id"] = pd.to_numeric(cleaned["movie_id"], errors="coerce").astype(int)
    cleaned["genre_id"] = pd.to_numeric(cleaned["genre_id"], errors="coerce").astype(int)

    # Remove exact business duplicates
    cleaned = cleaned.drop_duplicates(subset=["movie_id", "genre_id"]).reset_index(drop=True)

    # Remove orphaned references
    cleaned = cleaned[cleaned["movie_id"].isin(valid_movie_ids)]
    cleaned = cleaned[cleaned["genre_id"].isin(valid_genre_ids)]
    return cleaned.reset_index(drop=True)


def clean_cast(df: pd.DataFrame, valid_movie_ids: set) -> pd.DataFrame:
    """Cleans cast dataset and handles missing character names."""
    cleaned = df.copy()
    
    # Drop duplicates on business key
    cleaned = cleaned.drop_duplicates(subset=["movie_id", "person_id", "cast_order"]).reset_index(drop=True)

    # Types
    cleaned["cast_row_id"] = pd.to_numeric(cleaned["cast_row_id"], errors="coerce").astype(int)
    cleaned["movie_id"] = pd.to_numeric(cleaned["movie_id"], errors="coerce").astype(int)
    cleaned["person_id"] = pd.to_numeric(cleaned["person_id"], errors="coerce").astype(int)
    cleaned["cast_order"] = pd.to_numeric(cleaned["cast_order"], errors="coerce").fillna(0).astype(int)

    cleaned["actor_name"] = cleaned["actor_name"].astype(str).str.strip()
    cleaned["character_name"] = cleaned["character_name"].fillna("Unknown").astype(str).str.strip()
    cleaned["character_name"] = cleaned["character_name"].replace(["nan", "None", ""], "Unknown")

    # Filter orphans
    cleaned = cleaned[cleaned["movie_id"].isin(valid_movie_ids)]
    return cleaned.reset_index(drop=True)


def clean_crew(df: pd.DataFrame, valid_movie_ids: set) -> pd.DataFrame:
    """Cleans crew dataset."""
    cleaned = df.copy()

    # Drop duplicates on business key
    cleaned = cleaned.drop_duplicates(subset=["movie_id", "person_id", "job"]).reset_index(drop=True)

    cleaned["crew_row_id"] = pd.to_numeric(cleaned["crew_row_id"], errors="coerce").astype(int)
    cleaned["movie_id"] = pd.to_numeric(cleaned["movie_id"], errors="coerce").astype(int)
    cleaned["person_id"] = pd.to_numeric(cleaned["person_id"], errors="coerce").astype(int)

    cleaned["person_name"] = cleaned["person_name"].astype(str).str.strip()
    cleaned["job"] = cleaned["job"].astype(str).str.strip()
    cleaned["department"] = cleaned["department"].astype(str).str.strip()

    # Filter orphans
    cleaned = cleaned[cleaned["movie_id"].isin(valid_movie_ids)]
    return cleaned.reset_index(drop=True)


def clean_movie_keywords(df: pd.DataFrame, valid_movie_ids: set) -> pd.DataFrame:
    """Cleans movie_keywords dataset."""
    cleaned = df.copy()

    # Drop duplicates on composite key
    cleaned = cleaned.drop_duplicates(subset=["movie_id", "keyword_id"]).reset_index(drop=True)

    cleaned["movie_id"] = pd.to_numeric(cleaned["movie_id"], errors="coerce").astype(int)
    cleaned["keyword_id"] = pd.to_numeric(cleaned["keyword_id"], errors="coerce").astype(int)
    cleaned["keyword_name"] = cleaned["keyword_name"].astype(str).str.strip()

    # Filter orphans
    cleaned = cleaned[cleaned["movie_id"].isin(valid_movie_ids)]
    return cleaned.reset_index(drop=True)


def execute_full_cleaning_pipeline(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Runs complete end-to-end cleaning on all six tables with validation."""
    raw_tables = load_all_raw_datasets(data_dir)
    logger.info("Executing comprehensive relational cleaning pipeline...")

    # 1. Movies & Genres
    cleaned_movies = clean_movies(raw_tables["movies"])
    cleaned_genres = clean_genres(raw_tables["genres"])

    valid_movie_ids = set(cleaned_movies["movie_id"])
    valid_genre_ids = set(cleaned_genres["genre_id"])

    # 2. Relational child tables
    cleaned_movie_genres = clean_movie_genres(raw_tables["movie_genres"], valid_movie_ids, valid_genre_ids)
    cleaned_cast = clean_cast(raw_tables["cast"], valid_movie_ids)
    cleaned_crew = clean_crew(raw_tables["crew"], valid_movie_ids)
    cleaned_keywords = clean_movie_keywords(raw_tables["movie_keywords"], valid_movie_ids)

    logger.info(
        f"Pipeline Summary -> Movies: {len(cleaned_movies):,}, Genres: {len(cleaned_genres):,}, "
        f"Movie-Genres: {len(cleaned_movie_genres):,}, Cast: {len(cleaned_cast):,}, "
        f"Crew: {len(cleaned_crew):,}, Keywords: {len(cleaned_keywords):,}"
    )

    return {
        "movies": cleaned_movies,
        "genres": cleaned_genres,
        "movie_genres": cleaned_movie_genres,
        "cast": cleaned_cast,
        "crew": cleaned_crew,
        "movie_keywords": cleaned_keywords,
    }


def compute_financial_metrics(df_movies: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Net Profit and ROI strictly for records with valid (strictly > 0) budget and revenue.
    Keeps zero-budget/revenue rows masked as NaN for analytical accuracy.
    Prevents division by zero or distorted infinite percentages.
    """
    df = df_movies.copy()
    valid_mask = (df["budget"] > 0) & (df["revenue"] > 0)
    
    df["profit"] = np.where(valid_mask, df["revenue"] - df["budget"], np.nan)
    df["roi_percentage"] = np.where(valid_mask, ((df["revenue"] - df["budget"]) / df["budget"]) * 100, np.nan)
    df["profit_margin"] = np.where(valid_mask, (df["profit"] / df["revenue"]) * 100, np.nan)
    df["is_profitable"] = np.where(valid_mask, df["profit"] > 0, np.nan)
    return df


if __name__ == "__main__":
    tables = execute_full_cleaning_pipeline()
    print("Cleaning pipeline executed successfully!")
