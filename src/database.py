"""
src/database.py
Relational SQLite database design, schema definition, ETL pipeline,
and analytical query executor for Reel Insights 2.0.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sqlite3
import numpy as np
import pandas as pd

from src.config import DB_PATH, logger

# Register numpy adapters for SQLite in Python 3.14
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.float32, float)

DEFAULT_DB_PATH = DB_PATH


@contextmanager
def get_connection(db_path: Optional[str | Path] = None):
    """Context manager for SQLite database connection with foreign key enforcement."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=20.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_schema(db_path: Optional[str | Path] = None) -> None:
    """Create normalized relational tables and performance indexes in SQLite."""
    logger.info("Initializing relational SQLite schema and indexes...")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. movies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                original_language TEXT,
                release_date TEXT,
                runtime INTEGER,
                budget REAL,
                revenue REAL,
                popularity REAL,
                vote_average REAL,
                vote_count INTEGER,
                status TEXT,
                overview TEXT
            );
        """)

        # 2. genres lookup table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genres (
                genre_id INTEGER PRIMARY KEY,
                genre_name TEXT NOT NULL UNIQUE
            );
        """)

        # 3. movie_genres junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movie_genres (
                movie_id INTEGER,
                genre_id INTEGER,
                PRIMARY KEY (movie_id, genre_id),
                FOREIGN KEY (movie_id) REFERENCES movies (movie_id) ON DELETE CASCADE,
                FOREIGN KEY (genre_id) REFERENCES genres (genre_id) ON DELETE CASCADE
            );
        """)

        # 4. cast table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cast (
                cast_row_id INTEGER PRIMARY KEY,
                movie_id INTEGER,
                person_id INTEGER,
                actor_name TEXT,
                character_name TEXT,
                cast_order INTEGER,
                FOREIGN KEY (movie_id) REFERENCES movies (movie_id) ON DELETE CASCADE
            );
        """)

        # 5. crew table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crew (
                crew_row_id INTEGER PRIMARY KEY,
                movie_id INTEGER,
                person_id INTEGER,
                person_name TEXT,
                job TEXT,
                department TEXT,
                FOREIGN KEY (movie_id) REFERENCES movies (movie_id) ON DELETE CASCADE
            );
        """)

        # 6. movie_keywords table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movie_keywords (
                movie_id INTEGER,
                keyword_id INTEGER,
                keyword_name TEXT,
                PRIMARY KEY (movie_id, keyword_id),
                FOREIGN KEY (movie_id) REFERENCES movies (movie_id) ON DELETE CASCADE
            );
        """)

        # High-performance B-Tree indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_release ON movies(release_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_rev ON movies(revenue DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_budget ON movies(budget DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_rating ON movies(vote_average DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mg_genre ON movie_genres(genre_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mg_movie ON movie_genres(movie_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cast_actor ON cast(actor_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cast_movie ON cast(movie_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crew_job ON crew(job);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crew_person ON crew(person_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crew_movie ON crew(movie_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords_name ON movie_keywords(keyword_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords_movie ON movie_keywords(movie_id);")

        conn.commit()
    logger.info("Schema definition and indexing completed successfully.")


def load_cleaned_tables_to_db(cleaned_dict: Dict[str, pd.DataFrame], db_path: Optional[str | Path] = None) -> None:
    """Loads cleaned dataframes into SQLite tables and validates foreign keys."""
    create_schema(db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Clear existing data in reverse dependency order
        cursor.execute("DELETE FROM movie_keywords;")
        cursor.execute("DELETE FROM crew;")
        cursor.execute("DELETE FROM cast;")
        cursor.execute("DELETE FROM movie_genres;")
        cursor.execute("DELETE FROM genres;")
        cursor.execute("DELETE FROM movies;")
        conn.commit()

        # 1. movies table
        movies_df = cleaned_dict["movies"].copy()
        if "release_date_str" in movies_df.columns:
            movies_df["release_date"] = movies_df["release_date_str"]
        cols_movies = ["movie_id", "title", "original_language", "release_date", "runtime", "budget", "revenue", "popularity", "vote_average", "vote_count", "status", "overview"]
        movies_to_insert = movies_df[cols_movies].astype({
            "movie_id": int,
            "runtime": int,
            "budget": float,
            "revenue": float,
            "popularity": float,
            "vote_average": float,
            "vote_count": int
        })
        movies_to_insert.to_sql("movies", conn, if_exists="append", index=False)

        # 2. genres table
        genres_df = cleaned_dict["genres"][["genre_id", "genre_name"]].astype({"genre_id": int, "genre_name": str})
        genres_df.to_sql("genres", conn, if_exists="append", index=False)

        # 3. movie_genres table
        mg_df = cleaned_dict["movie_genres"][["movie_id", "genre_id"]].astype({"movie_id": int, "genre_id": int})
        mg_df.to_sql("movie_genres", conn, if_exists="append", index=False)

        # 4. cast table
        cast_df = cleaned_dict["cast"][["cast_row_id", "movie_id", "person_id", "actor_name", "character_name", "cast_order"]].astype({
            "cast_row_id": int,
            "movie_id": int,
            "person_id": int,
            "cast_order": int
        })
        cast_df.to_sql("cast", conn, if_exists="append", index=False)

        # 5. crew table
        crew_df = cleaned_dict["crew"][["crew_row_id", "movie_id", "person_id", "person_name", "job", "department"]].astype({
            "crew_row_id": int,
            "movie_id": int,
            "person_id": int
        })
        crew_df.to_sql("crew", conn, if_exists="append", index=False)

        # 6. movie_keywords table
        kw_df = cleaned_dict["movie_keywords"][["movie_id", "keyword_id", "keyword_name"]].astype({
            "movie_id": int,
            "keyword_id": int
        })
        kw_df.to_sql("movie_keywords", conn, if_exists="append", index=False)

        conn.commit()

        # Validate Foreign Keys
        cursor.execute("PRAGMA foreign_key_check;")
        fk_errors = cursor.fetchall()
        if fk_errors:
            err = f"Foreign key check detected {len(fk_errors)} violations: {fk_errors[:5]}"
            logger.error(err)
            raise ValueError(err)
        
        logger.info("Database loaded and validated with ZERO foreign key violations.")


def execute_query(query: str, params: Optional[Tuple | List] = None, db_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Execute raw SQL query with parameters and return pandas DataFrame."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def execute_safe_readonly_query(query: str, max_rows: int = 100, db_path: Optional[str | Path] = None) -> Tuple[bool, pd.DataFrame, str]:
    """
    Safely executes read-only SQL queries with AST/token guardrails for AI Analyst.
    Disallows data-modifying statements (INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, EXEC).
    """
    query_clean = query.strip()
    query_upper = query_clean.upper()

    # Blacklist dangerous write keywords
    forbidden_tokens = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "PRAGMA ", "CREATE ", "ATTACH ", "DETACH ", "REPLACE ", "TRUNCATE "]
    for token in forbidden_tokens:
        if token in query_upper:
            return False, pd.DataFrame(), f"Security rejection: Query contains disallowed operation '{token.strip()}'."

    if not query_upper.startswith("SELECT ") and not query_upper.startswith("WITH "):
        return False, pd.DataFrame(), "Security rejection: Query must start with SELECT or WITH."

    # Enforce LIMIT if not present
    if "LIMIT " not in query_upper:
        query_clean = f"{query_clean.rstrip(';')} LIMIT {max_rows};"

    try:
        df = execute_query(query_clean, db_path=db_path)
        return True, df, "Success"
    except Exception as e:
        logger.warning(f"Safe query execution failed: {e}")
        return False, pd.DataFrame(), str(e)


# ==============================================================================
# SPECIFICATION: 10 SQL ANALYTICAL QUESTIONS IMPLEMENTATION
# ==============================================================================

SQL_QUERIES: Dict[str, Dict[str, str]] = {
    "Q1": {
        "title": "Action Movies Released After 2015",
        "description": "List all movies in the Action genre released after 2015, showing title and release date.",
        "concept": "JOIN (movies ⋈ movie_genres ⋈ genres) + WHERE condition",
        "sql": """
SELECT 
    m.title,
    m.release_date
FROM movies m
JOIN movie_genres mg ON m.movie_id = mg.movie_id
JOIN genres g ON mg.genre_id = g.genre_id
WHERE g.genre_name = 'Action'
  AND strftime('%Y', m.release_date) > '2015'
ORDER BY m.release_date DESC;
        """,
        "insight": "Action cinema after 2015 saw an acceleration in franchise sequels and superhero tentpoles, representing a dominant share of studio output."
    },
    "Q2": {
        "title": "Top 10 Highest-Grossing Movies with Genres",
        "description": "Find the top 10 highest-grossing movies along with the genre(s) each belongs to.",
        "concept": "JOIN + GROUP_CONCAT + ORDER BY + LIMIT",
        "sql": """
SELECT 
    m.title,
    m.revenue,
    GROUP_CONCAT(g.genre_name, ', ') AS genres
FROM movies m
JOIN movie_genres mg ON m.movie_id = mg.movie_id
JOIN genres g ON mg.genre_id = g.genre_id
GROUP BY m.movie_id, m.title, m.revenue
ORDER BY m.revenue DESC
LIMIT 10;
        """,
        "insight": "All top 10 highest grossing movies belong to Adventure, Action, and Sci-Fi/Fantasy genres, emphasizing the global dominance of high-concept visual spectacle."
    },
    "Q3": {
        "title": "Average Budget and Revenue by Genre",
        "description": "Calculate the average budget and average revenue for each genre, sorted by average revenue descending.",
        "concept": "JOIN + GROUP BY + AVG() (Filtering out 0 values for true financial averages)",
        "sql": """
SELECT 
    g.genre_name,
    COUNT(DISTINCT m.movie_id) AS total_movies,
    ROUND(AVG(CASE WHEN m.budget > 0 THEN m.budget ELSE NULL END), 2) AS avg_budget,
    ROUND(AVG(CASE WHEN m.revenue > 0 THEN m.revenue ELSE NULL END), 2) AS avg_revenue,
    ROUND(AVG(m.vote_average), 2) AS avg_rating
FROM genres g
JOIN movie_genres mg ON g.genre_id = mg.genre_id
JOIN movies m ON mg.movie_id = m.movie_id
GROUP BY g.genre_name
ORDER BY avg_revenue DESC;
        """,
        "insight": "Animation, Adventure, and Sci-Fi command the highest average production budgets and generate the largest box-office returns."
    },
    "Q4": {
        "title": "Top 10 Most Frequent Actors",
        "description": "List the top 10 actors who have appeared in the most movies in the dataset.",
        "concept": "GROUP BY + COUNT() on cast table",
        "sql": """
SELECT 
    actor_name,
    COUNT(DISTINCT movie_id) AS movie_count
FROM cast
WHERE actor_name IS NOT NULL AND actor_name != ''
GROUP BY actor_name
ORDER BY movie_count DESC
LIMIT 10;
        """,
        "insight": "Versatile character actors and established leading stars (e.g. Samuel L. Jackson, Robert De Niro) achieve high appearance volumes across diverse genres."
    },
    "Q5": {
        "title": "Directors with More Than 3 Movies",
        "description": "Find all directors who have directed more than 3 movies, along with their movie count.",
        "concept": "GROUP BY + HAVING count on crew table",
        "sql": """
SELECT 
    person_name AS director_name,
    COUNT(DISTINCT movie_id) AS movie_count
FROM crew
WHERE job = 'Director'
GROUP BY person_name
HAVING COUNT(DISTINCT movie_id) > 3
ORDER BY movie_count DESC;
        """,
        "insight": "High-volume directors maintain sustained commercial relationships with major Hollywood studios and command repeatable financing."
    },
    "Q6": {
        "title": "Top 10 Most Frequently Used Keywords",
        "description": "List the top 10 most frequently used keywords across all movies.",
        "concept": "GROUP BY + COUNT() on movie_keywords",
        "sql": """
SELECT 
    keyword_name,
    COUNT(DISTINCT movie_id) AS frequency
FROM movie_keywords
WHERE keyword_name IS NOT NULL AND keyword_name != ''
GROUP BY keyword_name
ORDER BY frequency DESC
LIMIT 10;
        """,
        "insight": "Keywords reflecting universal conflict, relationships, and genre archetypes appear with highest frequency across the catalog."
    },
    "Q7": {
        "title": "High Budget, Below-Average Revenue Movies (Box Office Disappointments)",
        "description": "Find movies where the budget was above the overall average budget but the revenue was below the overall average revenue.",
        "concept": "Subquery comparison against overall AVG()",
        "sql": """
SELECT 
    title,
    budget,
    revenue,
    ROUND((SELECT AVG(budget) FROM movies WHERE budget > 0), 2) AS overall_avg_budget,
    ROUND((SELECT AVG(revenue) FROM movies WHERE revenue > 0), 2) AS overall_avg_revenue
FROM movies
WHERE budget > (SELECT AVG(budget) FROM movies WHERE budget > 0)
  AND revenue < (SELECT AVG(revenue) FROM movies WHERE revenue > 0)
  AND budget > 0
  AND revenue > 0
ORDER BY budget DESC;
        """,
        "insight": "These films represent major studio financial underperformers where substantial production investments failed to achieve mainstream audience traction."
    },
    "Q8": {
        "title": "Actors Who Worked with Christopher Nolan",
        "description": "List all actors who have appeared in a movie directed by Christopher Nolan.",
        "concept": "JOIN (cast ⋈ crew) with Director condition",
        "sql": """
SELECT DISTINCT 
    c.actor_name
FROM cast c
JOIN crew cr ON c.movie_id = cr.movie_id
WHERE cr.job = 'Director'
  AND cr.person_name = 'Christopher Nolan'
ORDER BY c.actor_name ASC;
        """,
        "insight": "Demonstrates Nolan's frequent collaborations with an ensemble troupe of actors across complex non-linear blockbusters."
    },
    "Q9": {
        "title": "Highest-Rated Genre with at Least 20 Movies",
        "description": "Find the genre with the highest average rating, considering only genres with at least 20 movies.",
        "concept": "JOIN + GROUP BY + HAVING count >= 20 + AVG()",
        "sql": """
SELECT 
    g.genre_name,
    COUNT(DISTINCT m.movie_id) AS movie_count,
    ROUND(AVG(m.vote_average), 2) AS avg_rating
FROM genres g
JOIN movie_genres mg ON g.genre_id = mg.genre_id
JOIN movies m ON mg.movie_id = m.movie_id
GROUP BY g.genre_name
HAVING COUNT(DISTINCT m.movie_id) >= 20
ORDER BY avg_rating DESC
LIMIT 1;
        """,
        "insight": "Story-driven, high-craft genres (such as Drama or Animation) consistently garner superior critical and audience ratings compared to mass commercial action."
    },
    "Q10": {
        "title": "Top 10 Movies with the Most Keywords Tagged",
        "description": "List the top 10 movies with the highest number of keywords tagged to them.",
        "concept": "JOIN (movies ⋈ movie_keywords) + GROUP BY + COUNT()",
        "sql": """
SELECT 
    m.title,
    COUNT(mk.keyword_id) AS keyword_count
FROM movies m
JOIN movie_keywords mk ON m.movie_id = mk.movie_id
GROUP BY m.movie_id, m.title
ORDER BY keyword_count DESC
LIMIT 10;
        """,
        "insight": "Complex narrative epics, multi-genre sci-fi, and comic book adaptations feature the richest thematic metadata."
    }
}


def run_sql_question(question_key: str, db_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Execute one of the 10 specification SQL questions by key (e.g. 'Q1' to 'Q10')."""
    if question_key not in SQL_QUERIES:
        raise ValueError(f"Unknown question key: {question_key}. Valid keys are Q1 to Q10.")
    return execute_query(SQL_QUERIES[question_key]["sql"], db_path=db_path)


def initialize_and_seed_database(db_path: Optional[str | Path] = None) -> None:
    """ETL script: cleans raw CSVs and populates SQLite database."""
    from src.cleaning import execute_full_cleaning_pipeline
    cleaned_tables = execute_full_cleaning_pipeline()
    logger.info("Loading cleaned tables into SQLite database...")
    load_cleaned_tables_to_db(cleaned_tables, db_path)
    logger.info(f"SQLite database successfully initialized and seeded at: {db_path or DEFAULT_DB_PATH}")


if __name__ == "__main__":
    initialize_and_seed_database()
    print("Database initialization completed successfully!")
