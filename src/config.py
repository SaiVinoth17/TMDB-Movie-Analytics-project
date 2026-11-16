"""
src/config.py
Centralized Configuration and Logging Manager for ReelInsights 2.0.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Root Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
DB_NAME = "movies.db"
DB_PATH = DATA_DIR / DB_NAME

# Raw Dataset Paths
RAW_DATA_PATHS = {
    "movies": DATA_DIR / "movies.csv",
    "genres": DATA_DIR / "genres.csv",
    "movie_genres": DATA_DIR / "movie_genres.csv",
    "cast": DATA_DIR / "cast.csv",
    "crew": DATA_DIR / "crew.csv",
    "movie_keywords": DATA_DIR / "movie_keywords.csv",
}

# Machine Learning & Analytics Parameters
ML_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
    "revenue_features": [
        "budget", "popularity", "vote_count", "vote_average", "runtime", "release_year"
    ],
    "rating_features": [
        "runtime", "budget", "popularity", "vote_count", "release_year"
    ],
    "tfidf_max_features": 5000,
    "recommendation_top_n": 10
}

# Theme Palette (Cinematic Dark / Charcoal & Electric Accents)
THEME_COLORS = {
    "bg_main": "#0D0E15",
    "bg_secondary": "#161823",
    "card_bg": "#1C1F2E",
    "card_border": "#282C3F",
    "accent_primary": "#E50914",     # Cinematic Red
    "accent_cyan": "#00D4FF",        # Electric Blue
    "accent_purple": "#8A2BE2",      # Neon Purple
    "accent_green": "#00F59B",       # Neon Green
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8"
}


# ==============================================================================
# STRUCTURED LOGGING SETUP
# ==============================================================================

def setup_logger(name: str = "reel_insights", log_file: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """Configures a thread-safe, structured logger with console and file handlers."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    log_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

