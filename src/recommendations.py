"""
src/recommendations.py
Content-Based Movie Recommendation & Similarity Engine for Reel Insights 2.0.
Combines film overviews, keywords, genres, and top-billed cast using TF-IDF and Cosine Similarity.
"""

from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ML_CONFIG, MODELS_DIR, logger
from src.database import execute_query, DEFAULT_DB_PATH


class MovieRecommendationEngine:
    """Computes and serves content-based movie recommendations using TF-IDF similarity."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.movies_df = pd.DataFrame()
        self.similarity_matrix = None
        self.indices = {}
        self._build_or_load_engine()

    def _build_or_load_engine(self) -> None:
        """Loads metadata, builds semantic feature soup, and computes cosine similarity."""
        sim_cache_path = MODELS_DIR / "cosine_similarity.joblib"
        meta_cache_path = MODELS_DIR / "rec_movies_metadata.joblib"

        if sim_cache_path.exists() and meta_cache_path.exists():
            logger.info("Loading cached movie recommendation engine artifacts...")
            self.similarity_matrix = joblib.load(sim_cache_path)
            self.movies_df = joblib.load(meta_cache_path)
            self.indices = pd.Series(self.movies_df.index, index=self.movies_df["title"].str.lower()).drop_duplicates()
            return

        logger.info("Building content-based movie similarity engine from SQLite...")
        query = """
        SELECT 
            m.movie_id,
            m.title,
            m.release_date,
            m.vote_average,
            m.popularity,
            m.overview,
            IFNULL(g_agg.genres, '') AS genres,
            IFNULL(k_agg.keywords, '') AS keywords,
            IFNULL(c_agg.cast_members, '') AS cast_members,
            IFNULL(cr_agg.director, '') AS director
        FROM movies m
        LEFT JOIN (
            SELECT mg.movie_id, GROUP_CONCAT(g.genre_name, ' ') AS genres
            FROM movie_genres mg JOIN genres g ON mg.genre_id = g.genre_id GROUP BY mg.movie_id
        ) g_agg ON m.movie_id = g_agg.movie_id
        LEFT JOIN (
            SELECT movie_id, GROUP_CONCAT(keyword_name, ' ') AS keywords
            FROM movie_keywords GROUP BY movie_id
        ) k_agg ON m.movie_id = k_agg.movie_id
        LEFT JOIN (
            SELECT movie_id, GROUP_CONCAT(actor_name, ' ') AS cast_members
            FROM cast WHERE cast_order <= 3 GROUP BY movie_id
        ) c_agg ON m.movie_id = c_agg.movie_id
        LEFT JOIN (
            SELECT movie_id, person_name AS director
            FROM crew WHERE job = 'Director' GROUP BY movie_id
        ) cr_agg ON m.movie_id = cr_agg.movie_id
        ORDER BY m.popularity DESC;
        """
        df = execute_query(query, db_path=self.db_path)

        # Build feature soup: weighted mix of genres, keywords, overview, director, and cast
        df["soup"] = (
            (df["genres"] + " ") * 3 +
            (df["keywords"] + " ") * 2 +
            (df["director"] + " ") * 2 +
            (df["cast_members"] + " ") +
            df["overview"].fillna("")
        ).str.lower()

        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=ML_CONFIG["tfidf_max_features"],
            ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf.fit_transform(df["soup"])
        
        # Compute pairwise Cosine Similarity
        logger.info(f"Computing cosine similarity matrix for {len(df):,} movies...")
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Cache artifacts
        joblib.dump(sim_matrix, sim_cache_path)
        joblib.dump(df[["movie_id", "title", "release_date", "vote_average", "popularity", "genres", "director", "overview"]], meta_cache_path)

        self.similarity_matrix = sim_matrix
        self.movies_df = df
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df["title"].str.lower()).drop_duplicates()
        logger.info("Recommendation engine successfully initialized.")

    def recommend(self, movie_title: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Returns top N content-similar movies based on genre, keyword, director, and cast signatures.
        """
        clean_title = movie_title.strip().lower()
        if clean_title not in self.indices:
            # Fuzzy match attempt
            matching = [t for t in self.indices.index if clean_title in t]
            if matching:
                clean_title = matching[0]
            else:
                return []

        idx = self.indices[clean_title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Skip index 0 (self)
        top_matches = sim_scores[1:top_n + 1]

        recommendations = []
        for i, score in top_matches:
            row = self.movies_df.iloc[i]
            recommendations.append({
                "movie_id": int(row["movie_id"]),
                "title": str(row["title"]),
                "release_date": str(row["release_date"]),
                "genres": str(row["genres"]),
                "director": str(row["director"]),
                "vote_average": float(row["vote_average"]),
                "similarity_score": round(float(score) * 100, 1),
                "overview": str(row["overview"])
            })

        return recommendations


# Singleton instance
_rec_engine: Optional[MovieRecommendationEngine] = None


def get_recommendation_engine() -> MovieRecommendationEngine:
    global _rec_engine
    if _rec_engine is None:
        _rec_engine = MovieRecommendationEngine()
    return _rec_engine

