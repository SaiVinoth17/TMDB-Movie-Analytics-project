"""
src/analysis.py
Pandas Analytical Engine for Reel Insights 2.0 TMDB Movie Intelligence Platform.
Implements all 10 project specification Pandas analytical questions,
cast & crew intelligence, and commercial metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import DATA_DIR, logger
from src.cleaning import execute_full_cleaning_pipeline, compute_financial_metrics
from src.database import execute_query, DEFAULT_DB_PATH


# ==============================================================================
# SPECIFICATION: 10 PANDAS ANALYTICAL QUESTIONS IMPLEMENTATION
# ==============================================================================

def pandas_q1_entity_counts(tables: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """
    P1: List the total number of unique movies, genres, cast members,
    and crew members collected across all tables.
    """
    unique_movies = tables["movies"]["movie_id"].nunique()
    unique_genres = tables["genres"]["genre_id"].nunique()
    unique_cast = tables["cast"]["person_id"].nunique()
    unique_crew = tables["crew"]["person_id"].nunique()
    
    counts = {
        "unique_movies": int(unique_movies),
        "unique_genres": int(unique_genres),
        "unique_cast_members": int(unique_cast),
        "unique_crew_members": int(unique_crew),
    }
    logger.info(f"P1 Entity counts: {counts}")
    return counts


def pandas_q2_missing_financials(movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    P2: List all movies with missing or zero budget/revenue values,
    and describe how these should be treated for the rest of the analysis.
    """
    zero_budget = movies_df["budget"] <= 0
    zero_revenue = movies_df["revenue"] <= 0

    missing_df = movies_df[zero_budget | zero_revenue].copy()
    zb = missing_df["budget"] <= 0
    zr = missing_df["revenue"] <= 0
    missing_df["missing_type"] = np.where(
        zb & zr, "Both Budget & Revenue Unknown",
        np.where(zb, "Budget Unknown (0)", "Revenue Unknown (0)")
    )

    summary = {
        "total_movies": len(movies_df),
        "zero_budget_count": int(zero_budget.sum()),
        "zero_revenue_count": int(zero_revenue.sum()),
        "both_zero_count": int((zero_budget & zero_revenue).sum()),
        "valid_financial_records": int((~zero_budget & ~zero_revenue).sum()),
        "treatment_rationale": (
            "TMDB uses 0 to represent unreported or unavailable financial figures, not a $0 cost. "
            "To prevent severe distortions in ROI, profit margins, and averages, these rows must NOT "
            "be filled with 0 for financial ratios. In Pandas, budget/revenue <= 0 are masked as NaN "
            "during Profit/ROI calculation, while preserved in the raw database for non-financial queries."
        )
    }
    return missing_df, summary


def pandas_q3_profit_and_roi(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    P3: List the profit (revenue - budget) and return on investment for
    every movie that has valid budget and revenue figures.
    """
    valid = movies_df[(movies_df["budget"] > 0) & (movies_df["revenue"] > 0)].copy()
    valid["profit"] = valid["revenue"] - valid["budget"]
    valid["roi_percentage"] = ((valid["profit"]) / valid["budget"]) * 100
    valid["profit_margin"] = (valid["profit"] / valid["revenue"]) * 100

    cols = ["movie_id", "title", "release_date", "budget", "revenue", "profit", "roi_percentage", "profit_margin"]
    return valid[cols].sort_values(by="profit", ascending=False).reset_index(drop=True)


def pandas_q4_top_15_roi(movies_df: pd.DataFrame, min_budget: float = 1_000_000.0) -> pd.DataFrame:
    """
    P4: List the top 15 movies by return on investment, considering only
    movies with a budget of at least $1 million.
    """
    valid = movies_df[(movies_df["budget"] >= min_budget) & (movies_df["revenue"] > 0)].copy()
    valid["profit"] = valid["revenue"] - valid["budget"]
    valid["roi_percentage"] = ((valid["profit"]) / valid["budget"]) * 100

    cols = ["movie_id", "title", "release_date", "budget", "revenue", "profit", "roi_percentage"]
    top15 = valid[cols].sort_values(by="roi_percentage", ascending=False).head(15).reset_index(drop=True)
    return top15


def pandas_q5_releases_per_year(movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    P5: List the number of movies released per year, and identify which
    year had the highest movie output in the dataset.
    """
    df = movies_df.copy()
    if "release_year" not in df.columns or df["release_year"].isnull().any():
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    yearly_counts = (
        df.dropna(subset=["release_year"])
        .groupby("release_year")
        .size()
        .reset_index(name="movie_count")
        .sort_values(by="release_year")
    )
    yearly_counts["release_year"] = yearly_counts["release_year"].astype(int)

    peak_row = yearly_counts.loc[yearly_counts["movie_count"].idxmax()]
    summary = {
        "peak_year": int(peak_row["release_year"]),
        "peak_movie_count": int(peak_row["movie_count"])
    }
    return yearly_counts, summary


def pandas_q6_runtime_outliers(movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    P6: List all movies whose runtime is a statistical outlier compared to
    the rest of the dataset using the standard Interquartile Range (IQR) rule.
    """
    df = movies_df[movies_df["runtime"] > 0].copy()
    q1 = df["runtime"].quantile(0.25)
    q3 = df["runtime"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df["runtime"] < lower_bound) | (df["runtime"] > upper_bound)].copy()
    outliers["outlier_type"] = np.where(outliers["runtime"] > upper_bound, "High Outlier (Epic/Long)", "Low Outlier (Short)")
    
    cols = ["movie_id", "title", "runtime", "release_date", "vote_average", "outlier_type"]
    outliers_sorted = outliers[cols].sort_values(by="runtime", ascending=False).reset_index(drop=True)

    bounds = {
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
        "lower_bound": round(float(lower_bound), 2),
        "upper_bound": round(float(upper_bound), 2),
        "outlier_count": len(outliers_sorted)
    }
    return outliers_sorted, bounds


def pandas_q7_primary_genre_popularity(
    movies_df: pd.DataFrame,
    movie_genres_df: pd.DataFrame,
    genres_df: pd.DataFrame
) -> pd.DataFrame:
    """
    P7: List each movie's primary genre (the first genre associated with the movie)
    and compare average popularity across primary genres.
    """
    primary_mg = movie_genres_df.drop_duplicates(subset=["movie_id"], keep="first")
    merged = primary_mg.merge(genres_df, on="genre_id", how="left")
    merged = movies_df.merge(merged[["movie_id", "genre_name"]], on="movie_id", how="left")
    merged["primary_genre"] = merged["genre_name"].fillna("Uncategorized")

    genre_pop = merged.groupby("primary_genre").agg(
        movie_count=("movie_id", "count"),
        avg_popularity=("popularity", "mean"),
        median_popularity=("popularity", "median"),
        avg_vote_average=("vote_average", "mean"),
        avg_revenue=("revenue", lambda x: x[x > 0].mean() if (x > 0).any() else 0.0)
    ).reset_index().sort_values(by="avg_popularity", ascending=False).reset_index(drop=True)

    genre_pop["avg_popularity"] = genre_pop["avg_popularity"].round(2)
    genre_pop["median_popularity"] = genre_pop["median_popularity"].round(2)
    genre_pop["avg_vote_average"] = genre_pop["avg_vote_average"].round(2)
    genre_pop["avg_revenue_m"] = (genre_pop["avg_revenue"] / 1e6).round(2)

    return genre_pop


def pandas_q8_prolific_actors(cast_df: pd.DataFrame, movies_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    P8: List the top 10 most prolific actors by number of movies,
    along with their average movie rating across those movies.
    """
    merged = cast_df.merge(movies_df[["movie_id", "vote_average", "title"]], on="movie_id", how="inner")
    
    actor_stats = merged.groupby("actor_name").agg(
        movie_count=("movie_id", "nunique"),
        avg_rating=("vote_average", "mean")
    ).reset_index()

    top_actors = actor_stats.sort_values(by=["movie_count", "avg_rating"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    top_actors["avg_rating"] = top_actors["avg_rating"].round(2)
    return top_actors


def pandas_q9_duplicate_titles_check(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    P9: List any movie titles that appear more than once in the dataset,
    along with their release years, to check for duplicates or remakes.
    """
    df = movies_df.copy()
    if "release_year" not in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    duplicated_titles = df[df.duplicated(subset=["title"], keep=False)].copy()
    cols = ["movie_id", "title", "release_year", "release_date", "original_language", "revenue", "budget"]
    available_cols = [c for c in cols if c in df.columns]
    
    sorted_dups = duplicated_titles[available_cols].sort_values(by=["title", "release_date"]).reset_index(drop=True)
    return sorted_dups


def pandas_q10_popularity_vote_divergence(movies_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    P10: List movies whose popularity score and vote count don't follow
    the general pattern between the two (e.g., unusually high popularity
    with a low vote count, or vice versa) using standardized Z-score divergence.
    """
    df = movies_df[(movies_df["vote_count"] > 10) & (movies_df["popularity"] > 0)].copy()

    log_pop = np.log1p(df["popularity"])
    log_votes = np.log1p(df["vote_count"])

    z_pop = (log_pop - log_pop.mean()) / log_pop.std()
    z_votes = (log_votes - log_votes.mean()) / log_votes.std()

    df["divergence"] = (z_pop - z_votes).round(2)
    df["divergence_type"] = np.where(
        df["divergence"] > 1.5, "Viral/Recent Spike (High Pop, Low Votes)",
        np.where(df["divergence"] < -1.5, "Legacy Classic (High Votes, Low Pop)", "Balanced")
    )

    divergent = df[df["divergence_type"] != "Balanced"].copy()
    divergent["abs_divergence"] = divergent["divergence"].abs()
    top_divergent = divergent.sort_values(by="abs_divergence", ascending=False).head(top_n).reset_index(drop=True)

    cols = ["movie_id", "title", "release_date", "popularity", "vote_count", "vote_average", "divergence", "divergence_type"]
    return top_divergent[cols]


# ==============================================================================
# CAST & CREW INTELLIGENCE HELPERS
# ==============================================================================

def get_director_analytics(min_movies: int = 3) -> pd.DataFrame:
    """Analyzes top directors by catalog volume, gross box-office, and audience rating."""
    query = """
    SELECT 
        c.person_name AS director_name,
        COUNT(DISTINCT m.movie_id) AS total_films,
        ROUND(SUM(CASE WHEN m.revenue > 0 THEN m.revenue ELSE 0 END) / 1e6, 1) AS total_box_office_m,
        ROUND(AVG(CASE WHEN m.revenue > 0 THEN m.revenue ELSE NULL END) / 1e6, 1) AS avg_box_office_m,
        ROUND(AVG(m.vote_average), 2) AS avg_rating,
        ROUND(AVG(m.popularity), 1) AS avg_popularity
    FROM crew c
    JOIN movies m ON c.movie_id = m.movie_id
    WHERE c.job = 'Director'
    GROUP BY c.person_name
    HAVING COUNT(DISTINCT m.movie_id) >= ?
    ORDER BY total_box_office_m DESC;
    """
    return execute_query(query, params=(min_movies,))


def get_actor_filmography(actor_name: str) -> pd.DataFrame:
    """Retrieves full filmography with box-office and character metadata for an actor."""
    query = """
    SELECT 
        m.title,
        m.release_date,
        c.character_name,
        c.cast_order,
        m.revenue,
        m.budget,
        m.vote_average
    FROM cast c
    JOIN movies m ON c.movie_id = m.movie_id
    WHERE c.actor_name LIKE ?
    ORDER BY m.release_date DESC;
    """
    return execute_query(query, params=(f"%{actor_name}%",))


def compute_comprehensive_kpis(movies_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes high-level summary KPIs for Dashboard and EDA."""
    total_movies = len(movies_df)
    valid_fin = movies_df[(movies_df["budget"] > 0) & (movies_df["revenue"] > 0)]
    
    total_rev = valid_fin["revenue"].sum()
    total_bud = valid_fin["budget"].sum()
    total_prof = total_rev - total_bud
    
    overall_margin = (total_prof / total_rev * 100) if total_rev > 0 else 0.0
    median_roi = (((valid_fin["revenue"] - valid_fin["budget"]) / valid_fin["budget"]) * 100).median()
    profitable_pct = ((valid_fin["revenue"] > valid_fin["budget"]).sum() / len(valid_fin) * 100) if len(valid_fin) > 0 else 0.0

    top_revenue_movie = movies_df.loc[movies_df["revenue"].idxmax()] if not movies_df.empty else None
    top_rated_movie = movies_df.loc[movies_df["vote_average"].idxmax()] if not movies_df.empty else None

    return {
        "total_movies": total_movies,
        "valid_financial_movies": len(valid_fin),
        "total_box_office": float(total_rev),
        "total_budget": float(total_bud),
        "total_profit": float(total_prof),
        "overall_margin_pct": round(float(overall_margin), 2),
        "median_roi_pct": round(float(median_roi), 2),
        "profitable_ratio_pct": round(float(profitable_pct), 1),
        "avg_vote_average": round(float(movies_df["vote_average"].mean()), 2),
        "avg_runtime": int(round(movies_df["runtime"].mean())),
        "top_grossing_title": top_revenue_movie["title"] if top_revenue_movie is not None else "N/A",
        "top_grossing_revenue": float(top_revenue_movie["revenue"]) if top_revenue_movie is not None else 0.0,
        "top_rated_title": top_rated_movie["title"] if top_rated_movie is not None else "N/A",
        "top_rated_score": float(top_rated_movie["vote_average"]) if top_rated_movie is not None else 0.0,
    }


if __name__ == "__main__":
    tables = execute_full_cleaning_pipeline()
    print("Pandas analytical engine operational.")
