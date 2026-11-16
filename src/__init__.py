"""
src package initialization
Reel Insights 2.0 - TMDB Movie Intelligence Platform
"""

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    DB_PATH,
    MODELS_DIR,
    LOGS_DIR,
    ML_CONFIG,
    THEME_COLORS,
    logger
)

from src.cleaning import (
    load_all_raw_datasets,
    clean_movies,
    clean_genres,
    clean_movie_genres,
    clean_cast,
    clean_crew,
    clean_movie_keywords,
    execute_full_cleaning_pipeline,
    compute_financial_metrics
)

from src.database import (
    create_schema,
    load_cleaned_tables_to_db,
    execute_query,
    execute_safe_readonly_query,
    run_sql_question,
    SQL_QUERIES,
    initialize_and_seed_database,
    DEFAULT_DB_PATH
)

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
    get_director_analytics,
    get_actor_filmography,
    compute_comprehensive_kpis
)

from src.visualization import (
    plot_univariate_budget_distribution,
    plot_univariate_runtime_distribution,
    plot_univariate_rating_distribution,
    plot_univariate_movies_per_genre,
    plot_bivariate_budget_vs_revenue,
    plot_bivariate_popularity_vs_votes,
    plot_bivariate_rating_by_genre,
    plot_bivariate_yearly_releases,
    plot_multivariate_correlation_heatmap,
    plot_multivariate_bubble_budget_revenue,
    plot_multivariate_genre_decade_heatmap,
    plot_top_10_highest_grossing_stacked
)

from src.ml_models import (
    train_and_evaluate_revenue_models,
    train_and_evaluate_rating_models,
    train_genre_classifier,
    predict_single_movie_revenue
)
from src.recommendations import MovieRecommendationEngine, get_recommendation_engine
