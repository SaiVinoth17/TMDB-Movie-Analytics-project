"""
src/visualization.py
Exploratory Data Visualization Suite for Reel Insights TMDB Movie Analytics.
Generates interactive Plotly and static Matplotlib/Seaborn figures covering
Univariate, Bivariate, and Multivariate analyses.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEME = "plotly_white"


# ==============================================================================
# UNIVARIATE VISUALIZATIONS
# ==============================================================================

def plot_univariate_budget_distribution(movies_df: pd.DataFrame) -> go.Figure:
    """Chart 1: Distribution of production budgets (valid budgets > $0)."""
    valid_budgets = movies_df[movies_df["budget"] > 0]["budget"] / 1e6
    fig = px.histogram(
        valid_budgets,
        x="budget",
        nbins=50,
        marginal="box",
        title="<b>Univariate: Production Budget Distribution ($ Millions)</b>",
        labels={"budget": "Budget ($M)"},
        color_discrete_sequence=["#2E86AB"],
        template=THEME
    )
    fig.update_layout(xaxis_title="Budget ($ Millions)", yaxis_title="Movie Count", height=450)
    return fig


def plot_univariate_runtime_distribution(movies_df: pd.DataFrame) -> go.Figure:
    """Chart 2: Distribution of runtimes across the catalog."""
    valid_runtimes = movies_df[movies_df["runtime"] > 0]
    fig = px.box(
        valid_runtimes,
        y="runtime",
        points="outliers",
        title="<b>Univariate: Theatrical Runtime Distribution (Minutes)</b>",
        labels={"runtime": "Runtime (Minutes)"},
        color_discrete_sequence=["#F6AE2D"],
        template=THEME
    )
    fig.update_layout(yaxis_title="Runtime (Minutes)", height=450)
    return fig


def plot_univariate_rating_distribution(movies_df: pd.DataFrame) -> go.Figure:
    """Chart 3: Distribution of vote_average scores."""
    fig = px.histogram(
        movies_df,
        x="vote_average",
        nbins=30,
        marginal="violin",
        title="<b>Univariate: Audience Vote Average Distribution (0-10 Scale)</b>",
        labels={"vote_average": "Vote Average"},
        color_discrete_sequence=["#86BBD8"],
        template=THEME
    )
    fig.update_layout(xaxis_title="Vote Average", yaxis_title="Movie Count", height=450)
    return fig


def plot_univariate_movies_per_genre(genres_agg_df: pd.DataFrame) -> go.Figure:
    """Chart 4: Count of movies per genre."""
    sorted_df = genres_agg_df.sort_values(by="total_movies", ascending=True)
    fig = px.bar(
        sorted_df,
        x="total_movies",
        y="genre_name",
        orientation="h",
        color="total_movies",
        color_continuous_scale="Blues",
        title="<b>Univariate: Movie Catalog Distribution by Genre</b>",
        labels={"total_movies": "Total Movies", "genre_name": "Genre"},
        template=THEME
    )
    fig.update_layout(coloraxis_showscale=False, height=500)
    return fig


# ==============================================================================
# BIVARIATE VISUALIZATIONS
# ==============================================================================

def plot_bivariate_budget_vs_revenue(movies_df: pd.DataFrame) -> go.Figure:
    """Chart 5: Budget vs Revenue with 1x breakeven reference line and OLS trendline."""
    valid = movies_df[(movies_df["budget"] > 0) & (movies_df["revenue"] > 0)].copy()
    valid["budget_m"] = (valid["budget"] / 1e6).round(2)
    valid["revenue_m"] = (valid["revenue"] / 1e6).round(2)

    fig = px.scatter(
        valid,
        x="budget_m",
        y="revenue_m",
        hover_name="title",
        title="<b>Bivariate: Production Budget vs Box Office Revenue ($M)</b>",
        labels={"budget_m": "Budget ($ Millions)", "revenue_m": "Revenue ($ Millions)"},
        color_discrete_sequence=["#33658A"],
        template=THEME
    )
    
    # Calculate OLS linear regression line directly (no statsmodels dependency)
    if len(valid) > 1:
        slope, intercept = np.polyfit(valid["budget_m"], valid["revenue_m"], 1)
        x_trend = np.linspace(valid["budget_m"].min(), valid["budget_m"].max(), 100)
        y_trend = slope * x_trend + intercept
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            name=f"OLS Trendline (slope: {slope:.2f})",
            line=dict(color="#2B2D42", width=2)
        ))

    # Add 1:1 Breakeven line
    max_bound = min(valid["budget_m"].max(), valid["revenue_m"].max())
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=max_bound, y1=max_bound,
        line=dict(color="red", dash="dash", width=1.5)
    )
    fig.add_annotation(
        x=max_bound * 0.75, y=max_bound * 0.78,
        text="1x Breakeven Line (Revenue = Budget)",
        showarrow=False,
        font=dict(color="red", size=10)
    )
    fig.update_layout(height=500)
    return fig


def plot_bivariate_popularity_vs_votes(movies_df: pd.DataFrame) -> go.Figure:
    """Chart 6: Popularity vs Vote Count."""
    valid = movies_df[(movies_df["popularity"] > 0) & (movies_df["vote_count"] > 0)]
    fig = px.scatter(
        valid,
        x="vote_count",
        y="popularity",
        hover_name="title",
        log_x=True,
        log_y=True,
        title="<b>Bivariate: TMDB Popularity Score vs Vote Count (Log Scale)</b>",
        labels={"vote_count": "Vote Count (Log)", "popularity": "Popularity Score (Log)"},
        color_discrete_sequence=["#F26419"],
        template=THEME
    )
    fig.update_layout(height=480)
    return fig


def plot_bivariate_rating_by_genre(genres_agg_df: pd.DataFrame) -> go.Figure:
    """Chart 7: Average user rating by genre."""
    rating_col = "avg_rating" if "avg_rating" in genres_agg_df.columns else "vote_average"
    sorted_df = genres_agg_df.sort_values(by=rating_col, ascending=True)
    fig = px.bar(
        sorted_df,
        x=rating_col,
        y="genre_name",
        orientation="h",
        color=rating_col,
        color_continuous_scale="Purples",
        title="<b>Bivariate: Average Audience Rating by Genre</b>",
        labels={rating_col: "Average Rating (0-10)", "genre_name": "Genre"},
        template=THEME
    )
    fig.update_layout(coloraxis_showscale=False, height=500)
    return fig


def plot_bivariate_yearly_releases(yearly_df: pd.DataFrame) -> go.Figure:
    """Chart 8: Number of movies released per year."""
    fig = px.line(
        yearly_df,
        x="release_year",
        y="movie_count",
        markers=True,
        title="<b>Bivariate: Annual Movie Release Volume Over Time</b>",
        labels={"release_year": "Release Year", "movie_count": "Movie Releases"},
        color_discrete_sequence=["#1A936F"],
        template=THEME
    )
    fig.update_layout(height=450)
    return fig


# ==============================================================================
# MULTIVARIATE VISUALIZATIONS
# ==============================================================================

def plot_multivariate_correlation_heatmap(movies_df: pd.DataFrame) -> go.Figure:
    """Chart 9: Correlation matrix heatmap across numeric features."""
    cols = ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]
    valid = movies_df[cols].replace(0, np.nan).dropna()
    corr = valid.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="<b>Multivariate: Numerical Feature Correlation Heatmap</b>",
        template=THEME
    )
    fig.update_layout(height=480)
    return fig


def plot_multivariate_bubble_budget_revenue(movies_df: pd.DataFrame, primary_genres_df: pd.DataFrame) -> go.Figure:
    """Chart 10: Budget vs Revenue colored by Primary Genre with bubble size = Vote Count."""
    valid = movies_df[(movies_df["budget"] > 0) & (movies_df["revenue"] > 0)].copy()
    valid = valid.merge(primary_genres_df[["movie_id", "primary_genre"]], on="movie_id", how="left")
    valid["primary_genre"] = valid["primary_genre"].fillna("Uncategorized")
    valid["budget_m"] = (valid["budget"] / 1e6).round(1)
    valid["revenue_m"] = (valid["revenue"] / 1e6).round(1)

    fig = px.scatter(
        valid,
        x="budget_m",
        y="revenue_m",
        color="primary_genre",
        size="vote_count",
        size_max=35,
        hover_name="title",
        title="<b>Multivariate: Budget vs Revenue by Primary Genre (Bubble Size = Vote Count)</b>",
        labels={"budget_m": "Budget ($M)", "revenue_m": "Revenue ($M)", "primary_genre": "Primary Genre"},
        template=THEME
    )
    fig.update_layout(height=550)
    return fig


def plot_multivariate_genre_decade_heatmap(movies_df: pd.DataFrame, movie_genres_df: pd.DataFrame, genres_df: pd.DataFrame) -> go.Figure:
    """Chart 11: Average Revenue by Genre and Release Decade Heatmap."""
    valid_movies = movies_df[(movies_df["revenue"] > 0)].copy()
    if valid_movies.empty:
        fig = px.imshow([[0]], title="<b>No revenue data available for selected filters</b>", template=THEME)
        fig.update_layout(height=400)
        return fig

    # Ensure release_decade is available
    if "release_decade" not in valid_movies.columns:
        if "release_year" not in valid_movies.columns:
            rel_date = pd.to_datetime(valid_movies["release_date"], errors="coerce")
            valid_movies["release_year"] = rel_date.dt.year
        valid_movies["release_decade"] = (valid_movies["release_year"] // 10 * 10).fillna(0).astype(int).astype(str) + "s"
    
    # Exclude invalid decade '0s'
    valid_movies = valid_movies[valid_movies["release_decade"] != "0s"]

    merged = valid_movies.merge(movie_genres_df, on="movie_id")
    merged = merged.merge(genres_df, on="genre_id")
    
    if merged.empty:
        fig = px.imshow([[0]], title="<b>No genre/decade records available for selected filters</b>", template=THEME)
        fig.update_layout(height=400)
        return fig

    pivot = merged.pivot_table(
        index="genre_name",
        columns="release_decade",
        values="revenue",
        aggfunc="mean"
    ) / 1e6
    pivot = pivot.round(1).fillna(0)

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="<b>Multivariate: Average Box Office ($M) by Genre Across Decades</b>",
        labels={"x": "Decade", "y": "Genre", "color": "Avg Rev ($M)"},
        template=THEME
    )
    fig.update_layout(height=520)
    return fig


def plot_top_10_highest_grossing_stacked(top10_df: pd.DataFrame) -> go.Figure:
    """Chart 12: Top 10 highest-grossing movies stacked bar chart."""
    df = top10_df.sort_values(by="revenue", ascending=True)
    df["budget_m"] = df["budget"] / 1e6
    df["profit_m"] = (df["revenue"] - df["budget"]) / 1e6

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["title"],
        x=df["budget_m"],
        name="Production Budget ($M)",
        orientation="h",
        marker_color="#E76F51"
    ))
    fig.add_trace(go.Bar(
        y=df["title"],
        x=df["profit_m"],
        name="Net Profit ($M)",
        orientation="h",
        marker_color="#2A9D8F"
    ))

    fig.update_layout(
        title="<b>Top 10 Highest-Grossing Movies (Budget vs Net Profit Stack)</b>",
        barmode="stack",
        xaxis_title="Amount ($ Millions)",
        template=THEME,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480
    )
    return fig
