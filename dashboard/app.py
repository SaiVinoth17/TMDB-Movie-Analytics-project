"""
dashboard/app.py
ReelInsights 2.0 — AI-Powered TMDB Movie Intelligence Platform.
Cinematic Dark UI with 7 Core Intelligence Workspaces:
1. Executive Overview
2. Movie Intelligence
3. Genre Intelligence
4. Financial Intelligence
5. Cast & Crew Intelligence
6. SQL Query Explorer
7. AI Movie Analyst (Ollama LLM + ML Lab)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    DB_PATH,
    THEME_COLORS,
    logger
)
from src.database import (
    SQL_QUERIES,
    execute_query,
    execute_safe_readonly_query,
    initialize_and_seed_database
)
from src.analysis import (
    compute_comprehensive_kpis,
    get_director_analytics,
    get_actor_filmography,
    pandas_q5_releases_per_year
)
from src.visualization import (
    plot_bivariate_budget_vs_revenue,
    plot_bivariate_popularity_vs_votes,
    plot_multivariate_correlation_heatmap,
    plot_multivariate_genre_decade_heatmap,
    plot_top_10_highest_grossing_stacked,
    plot_univariate_budget_distribution,
    plot_univariate_rating_distribution,
    plot_univariate_runtime_distribution
)
from src.ml_models import predict_single_movie_revenue, train_and_evaluate_revenue_models
from src.recommendations import get_recommendation_engine

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & CINEMATIC STYLING
# ==============================================================================

st.set_page_config(
    page_title="ReelInsights 2.0 | Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Cinematic Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0B0D14;
        color: #F8FAFC;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #11131F !important;
        border-right: 1px solid #1E2235;
    }

    /* Header styling */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(120deg, #FFFFFF 20%, #A5B4FC 60%, #E50914 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(26, 29, 45, 0.75) 0%, rgba(18, 20, 32, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(229, 9, 20, 0.4);
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .kpi-delta {
        font-size: 0.8rem;
        margin-top: 6px;
        color: #00F59B;
        font-weight: 500;
    }

    /* Cinematic Callout Banner */
    .cinematic-banner {
        background: linear-gradient(90deg, rgba(229, 9, 20, 0.15) 0%, rgba(138, 43, 226, 0.1) 100%);
        border-left: 4px solid #E50914;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 18px 0;
        color: #E2E8F0;
        font-size: 0.95rem;
    }

    /* SQL Terminal View */
    .sql-terminal {
        background-color: #08090E;
        border: 1px solid #1E2235;
        border-radius: 8px;
        padding: 16px;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        overflow-x: auto;
    }

    /* Badge indicators */
    .status-badge-online {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        background-color: rgba(0, 245, 155, 0.15);
        color: #00F59B;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(0, 245, 155, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING & CACHING PIPELINE
# ==============================================================================

@st.cache_data(ttl=3600)
def load_movie_intelligence_dataset():
    """Load core relational data from SQLite database with caching."""
    if not DB_PATH.exists():
        initialize_and_seed_database(DB_PATH)

    movies = execute_query("SELECT * FROM movies;", db_path=DB_PATH)
    genres = execute_query("SELECT * FROM genres;", db_path=DB_PATH)
    movie_genres = execute_query("SELECT * FROM movie_genres;", db_path=DB_PATH)

    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["release_year"] = movies["release_date"].dt.year
    movies["release_month"] = movies["release_date"].dt.month
    movies["release_decade"] = (movies["release_year"] // 10 * 10).fillna(0).astype(int).astype(str) + "s"
    
    # Financial metrics
    valid_fin = (movies["budget"] > 0) & (movies["revenue"] > 0)
    movies["profit"] = np.where(valid_fin, movies["revenue"] - movies["budget"], np.nan)
    movies["roi_percentage"] = np.where(valid_fin, (movies["revenue"] - movies["budget"]) / movies["budget"] * 100, np.nan)
    return movies, genres, movie_genres


movies_df, genres_df, mg_df = load_movie_intelligence_dataset()


# ==============================================================================
# SIDEBAR FILTERS (PERSISTENT VIA ST.SESSION_STATE)
# ==============================================================================

st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
    <span style="font-size: 1.8rem;">🎬</span>
    <div>
        <div style="font-weight: 800; font-size: 1.15rem; color: #FFFFFF;">REELINSIGHTS</div>
        <div style="font-size: 0.72rem; color: #E50914; letter-spacing: 0.15em; font-weight: 700;">INTELLIGENCE 2.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Session state initialization
if "filter_genres" not in st.session_state:
    st.session_state.filter_genres = []
if "filter_min_year" not in st.session_state:
    st.session_state.filter_min_year = int(movies_df["release_year"].dropna().min())
if "filter_max_year" not in st.session_state:
    st.session_state.filter_max_year = int(movies_df["release_year"].dropna().max())
if "filter_min_rating" not in st.session_state:
    st.session_state.filter_min_rating = 0.0
if "filter_language" not in st.session_state:
    st.session_state.filter_language = "All"

# 1. Genre Filter
all_genres = sorted(genres_df["genre_name"].dropna().unique().tolist())
sel_genres = st.sidebar.multiselect(
    "Genres",
    options=all_genres,
    default=st.session_state.filter_genres,
    help="Filter movies across one or more genres."
)
st.session_state.filter_genres = sel_genres

# 2. Year Range
y_min_bound = int(movies_df["release_year"].dropna().min())
y_max_bound = int(movies_df["release_year"].dropna().max())
sel_year_range = st.sidebar.slider(
    "Release Window",
    min_value=y_min_bound,
    max_value=y_max_bound,
    value=(st.session_state.filter_min_year, st.session_state.filter_max_year),
    step=1
)
st.session_state.filter_min_year, st.session_state.filter_max_year = sel_year_range

# 3. Rating Slider
sel_rating = st.sidebar.slider(
    "Minimum Audience Rating",
    min_value=0.0,
    max_value=10.0,
    value=st.session_state.filter_min_rating,
    step=0.5
)
st.session_state.filter_min_rating = sel_rating

# 4. Language Filter
lang_list = ["All"] + sorted([l for l in movies_df["original_language"].dropna().unique() if l != "Unknown"])
lang_idx = lang_list.index(st.session_state.filter_language) if st.session_state.filter_language in lang_list else 0
sel_lang = st.sidebar.selectbox("Original Language", options=lang_list, index=lang_idx)
st.session_state.filter_language = sel_lang

if st.sidebar.button("🔄 Reset Filter Stack", use_container_width=True):
    st.session_state.filter_genres = []
    st.session_state.filter_min_year = y_min_bound
    st.session_state.filter_max_year = y_max_bound
    st.session_state.filter_min_rating = 0.0
    st.session_state.filter_language = "All"
    st.rerun()


# Apply filters
filtered_movies = movies_df[
    (movies_df["release_year"] >= st.session_state.filter_min_year) &
    (movies_df["release_year"] <= st.session_state.filter_max_year) &
    (movies_df["vote_average"] >= st.session_state.filter_min_rating)
].copy()

if st.session_state.filter_language != "All":
    filtered_movies = filtered_movies[filtered_movies["original_language"] == st.session_state.filter_language]

if st.session_state.filter_genres:
    matched_g_ids = genres_df[genres_df["genre_name"].isin(st.session_state.filter_genres)]["genre_id"]
    valid_m_ids = mg_df[mg_df["genre_id"].isin(matched_g_ids)]["movie_id"].unique()
    filtered_movies = filtered_movies[filtered_movies["movie_id"].isin(valid_m_ids)]

# Sidebar Match Status
st.sidebar.markdown("---")
pct_catalog = (len(filtered_movies) / len(movies_df)) * 100
st.sidebar.markdown(f"**Catalog Sliced:** `{len(filtered_movies):,}` of `{len(movies_df):,}` films ({pct_catalog:.1f}%)")
st.sidebar.progress(max(0.01, min(1.0, pct_catalog / 100.0)))

# Sidebar Navigation Mode
st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Application Navigation")
app_nav_mode = st.sidebar.radio(
    "Choose Page:",
    [
        "📊 Page 2 — Dashboard",
        "🎯 Page 1 — Filters",
        "🔍 Page 3 — Query Explorer"
    ],
    index=0
)


# ==============================================================================
# PAGE 1 VIEW: DEDICATED FILTERS & SLICER PREVIEW
# ==============================================================================
if app_nav_mode == "🎯 Page 1 — Filters":
    st.markdown('<div class="hero-title">Reel Insights: TMDB Movie Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Page 1 — Filters & Interactive Catalog Slicer</div>', unsafe_allow_html=True)

    st.header("🎯 Live Catalog Filter Stack")
    st.markdown("Inspect active filter criteria and explore the dynamically sliced TMDB dataset. All filters persist across all pages via `st.session_state`.")

    cf1, cf2, cf3, cf4 = st.columns(4)
    with cf1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Selected Genres</div>
            <div class="kpi-value" style="font-size: 1.15rem;">{', '.join(st.session_state.filter_genres) if st.session_state.filter_genres else 'All 19 Genres'}</div>
            <div class="kpi-delta">Categorical Filter</div>
        </div>
        """, unsafe_allow_html=True)
    with cf2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Release Window</div>
            <div class="kpi-value" style="font-size: 1.15rem;">{st.session_state.filter_min_year} – {st.session_state.filter_max_year}</div>
            <div class="kpi-delta">Historical Temporal Scope</div>
        </div>
        """, unsafe_allow_html=True)
    with cf3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Minimum Rating</div>
            <div class="kpi-value" style="font-size: 1.15rem;">≥ {st.session_state.filter_min_rating:.1f} / 10</div>
            <div class="kpi-delta">Quality Consensus Threshold</div>
        </div>
        """, unsafe_allow_html=True)
    with cf4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Original Language</div>
            <div class="kpi-value" style="font-size: 1.15rem;">{st.session_state.filter_language}</div>
            <div class="kpi-delta">Linguistic Scope</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cinematic-banner">
        <b>Live Match Status:</b> Current filters match <b>{len(filtered_movies):,} films</b> out of <b>{len(movies_df):,} total catalog movies</b> ({pct_catalog:.1f}%).
        Use the sidebar to adjust genres, release years, ratings, or original language in real time.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📋 Sliced Catalog Preview")
    disp_cols = ["title", "release_year", "vote_average", "vote_count", "budget", "revenue", "original_language"]
    disp_df = filtered_movies[disp_cols].copy()
    disp_df["budget"] = disp_df["budget"].apply(lambda x: f"${x/1e6:,.1f}M" if x > 0 else "Unknown")
    disp_df["revenue"] = disp_df["revenue"].apply(lambda x: f"${x/1e6:,.1f}M" if x > 0 else "Unknown")
    st.dataframe(disp_df, use_container_width=True, hide_index=True)
    st.stop()


# ==============================================================================
# PAGE 3 VIEW: DEDICATED SQL QUERY EXPLORER
# ==============================================================================
if app_nav_mode == "🔍 Page 3 — Query Explorer":
    st.markdown('<div class="hero-title">Reel Insights: TMDB Movie Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Page 3 — SQL Query Explorer (All 10 Specification Questions)</div>', unsafe_allow_html=True)

    st.header("🔍 Relational SQL Query Explorer")
    st.markdown("Execute all **10 Project Specification SQL Questions** directly against `data/movies.db` using Python and SQLite.")

    q_options = [f"{k}: {SQL_QUERIES[k]['title']}" for k in sorted(SQL_QUERIES.keys(), key=lambda x: int(x[1:]))]
    chosen_q = st.selectbox("Select SQL Question:", options=q_options, key="select_sql_page_q")

    q_key = chosen_q.split(":")[0]
    q_data = SQL_QUERIES[q_key]

    st.markdown(f"**Objective:** {q_data['description']}")
    st.markdown(f"**SQL Concepts:** `{q_data['concept']}`")
    st.markdown(f'<div class="sql-terminal">{q_data["sql"]}</div>', unsafe_allow_html=True)

    res_df = execute_query(q_data["sql"], db_path=DB_PATH)
    col_out_l, col_out_r = st.columns([1, 1])
    with col_out_l:
        st.markdown(f"**Query Results:** `{len(res_df):,}` rows returned.")
        st.dataframe(res_df, use_container_width=True)
        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(f"📥 Download {q_key} Results (CSV)", data=csv_bytes, file_name=f"{q_key}_results.csv", mime="text/csv", key=f"btn_dl_page_{q_key}")

    with col_out_r:
        if q_key == "Q1":
            yearly_action = res_df.copy()
            yearly_action["year"] = pd.to_datetime(yearly_action["release_date"]).dt.year
            action_counts = yearly_action["year"].value_counts().sort_index().reset_index()
            action_counts.columns = ["Year", "Action Movies"]
            st.plotly_chart(px.bar(action_counts, x="Year", y="Action Movies", title="Action Releases per Year (>2015)"), use_container_width=True, key=f"chart_page_{q_key}")
        elif q_key == "Q2":
            q2_df = res_df.copy()
            q2_df["rev_m"] = q2_df["revenue"] / 1e6
            st.plotly_chart(px.bar(q2_df.sort_values("rev_m", ascending=True), x="rev_m", y="title", orientation="h", title="Top 10 Grossing Films ($M)"), use_container_width=True, key=f"chart_page_{q_key}")
        elif q_key == "Q3":
            q3_df = res_df.copy()
            st.plotly_chart(px.bar(q3_df.sort_values("avg_revenue", ascending=True), x="avg_revenue", y="genre_name", orientation="h", title="Avg Revenue by Genre ($M)"), use_container_width=True, key=f"chart_page_{q_key}")
        elif q_key == "Q4":
            st.plotly_chart(px.bar(res_df.sort_values("movie_count", ascending=True), x="movie_count", y="actor_name", orientation="h", title="Top 10 Most Frequent Actors"), use_container_width=True, key=f"chart_page_{q_key}")
        elif q_key == "Q5":
            st.plotly_chart(px.bar(res_df.head(15).sort_values("movie_count", ascending=True), x="movie_count", y="director_name", orientation="h", title="Top Directors (>3 Movies)"), use_container_width=True, key=f"chart_page_{q_key}")
        elif q_key == "Q6":
            st.plotly_chart(px.bar(res_df.sort_values("frequency", ascending=True), x="frequency", y="keyword_name", orientation="h", title="Top 10 Thematic Keywords"), use_container_width=True, key=f"chart_page_{q_key}")
        elif q_key == "Q7":
            st.plotly_chart(px.scatter(res_df.head(50), x="budget", y="revenue", hover_name="title", title="High Budget, Below-Average Revenue"), use_container_width=True, key=f"chart_page_{q_key}")
        elif q_key == "Q8":
            sample_actors = res_df.head(20).copy()
            sample_actors["count"] = 1
            st.plotly_chart(px.bar(sample_actors, x="count", y="actor_name", orientation="h", title="Actors in Nolan Films"), use_container_width=True, key=f"chart_page_{q_key}")
        elif q_key == "Q9":
            st.success(f"Top Genre: **{res_df['genre_name'].iloc[0]}** with avg rating of **{res_df['avg_rating'].iloc[0]}** across {res_df['movie_count'].iloc[0]} movies.")
        elif q_key == "Q10":
            st.plotly_chart(px.bar(res_df.sort_values("keyword_count", ascending=True), x="keyword_count", y="title", orientation="h", title="Top 10 Movies by Keyword Tags"), use_container_width=True, key=f"chart_page_{q_key}")

    st.markdown(f"""
    <div class="cinematic-banner">
        <b>Analytical Takeaway:</b> {q_data['insight']}
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ==============================================================================
# PAGE 2 VIEW: DASHBOARD (ANALYTICS & INTELLIGENCE HUB)
# ==============================================================================

st.markdown('<div class="hero-title">Reel Insights: TMDB Movie Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Page 2 — Comprehensive Movie Analytics Dashboard • Box-Office Econometrics</div>', unsafe_allow_html=True)

nav_tabs = st.tabs([
    "📊 Executive Overview",
    "🎬 Movie Intelligence",
    "🎭 Genre Intelligence",
    "💰 Financial Intelligence",
    "👥 Cast & Crew Intelligence",
    "🔮 Predictive Modeling & Recs"
])


# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================

with nav_tabs[0]:
    if filtered_movies.empty:
        st.warning("No movies match the active filter criteria. Please broaden your filters.")
        st.stop()

    # Precompute metrics
    valid_fin = filtered_movies[(filtered_movies["budget"] > 0) & (filtered_movies["revenue"] > 0)]
    total_rev = valid_fin["revenue"].sum()
    total_bud = valid_fin["budget"].sum()
    total_prof = total_rev - total_bud
    avg_rating = filtered_movies["vote_average"].mean()

    # Count distinct genres and actors in filtered subset
    movie_ids_tuple = tuple(filtered_movies["movie_id"].tolist()) if len(filtered_movies) > 1 else f"({filtered_movies['movie_id'].iloc[0]})"
    genre_cnt = execute_query(f"SELECT COUNT(DISTINCT genre_id) FROM movie_genres WHERE movie_id IN {movie_ids_tuple};").iloc[0, 0]
    actor_cnt = execute_query(f"SELECT COUNT(DISTINCT person_id) FROM cast WHERE movie_id IN {movie_ids_tuple};").iloc[0, 0]

    top_grossing = filtered_movies.loc[filtered_movies["revenue"].idxmax()] if not filtered_movies.empty else None
    top_rated = filtered_movies[filtered_movies["vote_count"] >= 100].sort_values("vote_average", ascending=False).head(1)
    top_rated_title = top_rated["title"].iloc[0] if not top_rated.empty else "N/A"
    top_rated_val = top_rated["vote_average"].iloc[0] if not top_rated.empty else 0.0

    # 4 Primary KPI Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Catalog Movies</div>
            <div class="kpi-value">{len(filtered_movies):,}</div>
            <div class="kpi-delta">{pct_catalog:.1f}% of entire database</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Rating</div>
            <div class="kpi-value">{avg_rating:.2f} <span style="font-size: 1rem; color: #94A3B8;">/ 10</span></div>
            <div class="kpi-delta">Audience Consensus</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Gross Box Office</div>
            <div class="kpi-value">${total_rev / 1e9:.2f}B</div>
            <div class="kpi-delta">${total_prof / 1e9:.2f}B Net Profit</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Top Billed Actors</div>
            <div class="kpi-value">{actor_cnt:,}</div>
            <div class="kpi-delta">{genre_cnt} Genres Represented</div>
        </div>
        """, unsafe_allow_html=True)

    # Narrative Banner
    st.markdown(f"""
    <div class="cinematic-banner">
        <b>Executive Brief:</b> Active selection encompasses <b>{len(filtered_movies):,} feature films</b>. 
        Highest-grossing: <b>{top_grossing['title']} (${top_grossing['revenue']/1e6:,.1f}M)</b>. 
        Highest-rated (min 100 votes): <b>{top_rated_title} ({top_rated_val} / 10)</b>.
    </div>
    """, unsafe_allow_html=True)

    c_ov1, c_ov2 = st.columns([3, 2])
    with c_ov1:
        st.subheader("Top 10 Highest-Grossing Movies (Budget vs Net Profit)")
        top10_filtered = filtered_movies.nlargest(10, "revenue")
        st.plotly_chart(plot_top_10_highest_grossing_stacked(top10_filtered), use_container_width=True, key="chart_ov_top10")

    with c_ov2:
        st.subheader("Genre Volume Distribution")
        mg_filtered = mg_df[mg_df["movie_id"].isin(filtered_movies["movie_id"])].merge(genres_df, on="genre_id")
        genre_dist = mg_filtered["genre_name"].value_counts().reset_index()
        genre_dist.columns = ["Genre", "Count"]

        fig_pie = px.pie(
            genre_dist.head(8),
            names="Genre",
            values="Count",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="<b>Top Active Genres Share</b>"
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig_pie, use_container_width=True, key="chart_ov_genre_dist")

    c_sc1, c_sc2 = st.columns([3, 2])
    with c_sc1:
        st.subheader("Budget vs Revenue (with 1x Breakeven Line)")
        st.plotly_chart(plot_bivariate_budget_vs_revenue(filtered_movies), use_container_width=True, key="chart_ov_budget_vs_rev")

    with c_sc2:
        st.subheader("Annual Movie Production & Release Trajectory")
        yearly_df, _ = pandas_q5_releases_per_year(filtered_movies)
        fig_yr = px.area(
            yearly_df,
            x="release_year",
            y="movie_count",
            markers=True,
            title="<b>Theatrical Release Volume Over Time</b>",
            color_discrete_sequence=["#E50914"]
        )
        fig_yr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig_yr, use_container_width=True, key="chart_ov_yearly_releases")

    # Dynamic text insight summarizing the filtered view (Page 2 Requirement)
    top_genre_name = genre_dist.iloc[0]["Genre"] if not genre_dist.empty else "N/A"
    st.markdown(f"""
    <div class="cinematic-banner">
        <b>Filtered View Summary Insight:</b> The highest-represented genre in this selection is <b>{top_genre_name}</b>. 
        Across these {len(filtered_movies):,} matching movies, the average audience consensus is <b>{avg_rating:.2f} / 10</b> with 
        <b>${total_rev/1e9:.2f}B</b> in cumulative box office revenue and <b>${total_prof/1e9:.2f}B</b> in estimated net profit.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: MOVIE INTELLIGENCE
# ==============================================================================

with nav_tabs[1]:
    st.header("🎬 Deep Movie Intelligence & Head-to-Head Comparison")

    col_srch, col_sort = st.columns([3, 1])
    with col_srch:
        search_query = st.text_input("🔍 Search Movie by Title or Keyword:", "")
    with col_sort:
        sort_choice = st.selectbox("Sort By:", ["Revenue (High to Low)", "Rating (High to Low)", "Popularity (High to Low)", "Budget (High to Low)"])

    sort_map = {
        "Revenue (High to Low)": ("revenue", False),
        "Rating (High to Low)": ("vote_average", False),
        "Popularity (High to Low)": ("popularity", False),
        "Budget (High to Low)": ("budget", False)
    }
    sort_col, sort_asc = sort_map[sort_choice]

    disp_movies = filtered_movies.copy()
    if search_query:
        disp_movies = disp_movies[disp_movies["title"].str.contains(search_query, case=False, na=False)]

    disp_movies = disp_movies.sort_values(sort_col, ascending=sort_asc)

    # Display Movie Table
    table_cols = ["movie_id", "title", "release_date", "budget", "revenue", "vote_average", "vote_count", "popularity", "runtime"]
    formatted_table = disp_movies[table_cols].copy()
    formatted_table["budget"] = formatted_table["budget"].apply(lambda x: f"${x/1e6:,.1f}M" if x > 0 else "Unknown")
    formatted_table["revenue"] = formatted_table["revenue"].apply(lambda x: f"${x/1e6:,.1f}M" if x > 0 else "Unknown")
    formatted_table["release_date"] = formatted_table["release_date"].dt.strftime("%Y-%m-%d")

    st.dataframe(formatted_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("⚔️ Head-to-Head Movie Comparison")
    all_titles = movies_df["title"].dropna().unique().tolist()
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        movie_a = st.selectbox("Select Film A:", options=all_titles, index=all_titles.index("Inception") if "Inception" in all_titles else 0)
    with col_c2:
        movie_b = st.selectbox("Select Film B:", options=all_titles, index=all_titles.index("Interstellar") if "Interstellar" in all_titles else 1)

    if movie_a and movie_b:
        mA = movies_df[movies_df["title"] == movie_a].iloc[0]
        mB = movies_df[movies_df["title"] == movie_b].iloc[0]

        comp_data = {
            "Metric": ["Release Date", "Production Budget", "Gross Box Office", "Net Profit", "Audience Rating", "Vote Count", "Popularity", "Runtime"],
            movie_a: [
                str(mA["release_date"].strftime("%Y-%m-%d")),
                f"${mA['budget']/1e6:,.1f}M" if mA["budget"] > 0 else "Unknown",
                f"${mA['revenue']/1e6:,.1f}M" if mA["revenue"] > 0 else "Unknown",
                f"${(mA['revenue'] - mA['budget'])/1e6:,.1f}M" if mA["budget"] > 0 and mA["revenue"] > 0 else "N/A",
                f"{mA['vote_average']} / 10",
                f"{mA['vote_count']:,}",
                f"{mA['popularity']}",
                f"{mA['runtime']} min"
            ],
            movie_b: [
                str(mB["release_date"].strftime("%Y-%m-%d")),
                f"${mB['budget']/1e6:,.1f}M" if mB["budget"] > 0 else "Unknown",
                f"${mB['revenue']/1e6:,.1f}M" if mB["revenue"] > 0 else "Unknown",
                f"${(mB['revenue'] - mB['budget'])/1e6:,.1f}M" if mB["budget"] > 0 and mB["revenue"] > 0 else "N/A",
                f"{mB['vote_average']} / 10",
                f"{mB['vote_count']:,}",
                f"{mB['popularity']}",
                f"{mB['runtime']} min"
            ]
        }
        st.table(pd.DataFrame(comp_data).set_index("Metric"))


# ==============================================================================
# TAB 3: GENRE INTELLIGENCE
# ==============================================================================

with nav_tabs[2]:
    st.header("🎭 Genre Commercial Dynamics & Decadal Evolution")

    # Exploded genre metrics
    mg_joined = mg_df.merge(genres_df, on="genre_id").merge(filtered_movies, on="movie_id")
    
    genre_perf = mg_joined.groupby("genre_name").agg(
        movie_count=("movie_id", "nunique"),
        avg_revenue=("revenue", lambda x: x[x > 0].mean() if (x > 0).any() else 0.0),
        avg_budget=("budget", lambda x: x[x > 0].mean() if (x > 0).any() else 0.0),
        avg_rating=("vote_average", "mean"),
        total_revenue=("revenue", "sum")
    ).reset_index()

    genre_perf["avg_rev_m"] = (genre_perf["avg_revenue"] / 1e6).round(1)
    genre_perf["avg_bud_m"] = (genre_perf["avg_budget"] / 1e6).round(1)
    genre_perf["avg_rating"] = genre_perf["avg_rating"].round(2)
    genre_perf["profit_margin_pct"] = np.where(
        genre_perf["avg_revenue"] > 0,
        ((genre_perf["avg_revenue"] - genre_perf["avg_budget"]) / genre_perf["avg_revenue"]) * 100,
        0.0
    ).round(1)

    cg1, cg2 = st.columns(2)
    with cg1:
        st.subheader("Average Box-Office Revenue by Genre ($M)")
        fig_g_rev = px.bar(
            genre_perf.sort_values("avg_rev_m", ascending=True),
            x="avg_rev_m",
            y="genre_name",
            orientation="h",
            color="avg_rev_m",
            color_continuous_scale="Viridis",
            labels={"avg_rev_m": "Avg Revenue ($M)", "genre_name": "Genre"}
        )
        fig_g_rev.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig_g_rev, use_container_width=True, key="chart_g_rev")

    with cg2:
        st.subheader("Average Audience Rating by Genre")
        fig_g_rat = px.bar(
            genre_perf.sort_values("avg_rating", ascending=True),
            x="avg_rating",
            y="genre_name",
            orientation="h",
            color="avg_rating",
            color_continuous_scale="Purples",
            labels={"avg_rating": "Average Rating (0-10)", "genre_name": "Genre"}
        )
        fig_g_rat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig_g_rat, use_container_width=True, key="chart_g_rat")

    st.subheader("Average Box-Office Across Decades Heatmap ($M)")
    fig_dec = plot_multivariate_genre_decade_heatmap(filtered_movies, mg_df, genres_df)
    st.plotly_chart(fig_dec, use_container_width=True, key="chart_g_decade_heatmap")


# ==============================================================================
# TAB 4: FINANCIAL INTELLIGENCE
# ==============================================================================

with nav_tabs[3]:
    st.header("💰 Financial Econometrics & Break-Even Risk Analysis")

    # Data Coverage Indicator
    total_f = len(filtered_movies)
    v_fin = filtered_movies[(filtered_movies["budget"] > 0) & (filtered_movies["revenue"] > 0)]
    coverage_pct = (len(v_fin) / total_f * 100) if total_f > 0 else 0.0

    st.markdown(f"""
    <div style="background-color: #161823; border: 1px solid #282C3F; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
        <b>Financial Reporting Data Coverage:</b> {len(v_fin):,} of {total_f:,} films ({coverage_pct:.1f}%) contain complete budget and revenue records.
        <i>Notice: Zero values are audited and treated strictly as missing/unreported. Never present revenue as profit.</i>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Production Budget vs Box Office Revenue (with 1x Breakeven Line)")
    st.plotly_chart(plot_bivariate_budget_vs_revenue(filtered_movies), use_container_width=True, key="chart_fin_budget_vs_rev")

    cf1, cf2 = st.columns(2)
    with cf1:
        st.subheader("Budget Outliers & Skewness")
        st.plotly_chart(plot_univariate_budget_distribution(filtered_movies), use_container_width=True, key="chart_fin_budget_dist")

    with cf2:
        st.subheader("Correlation Heatmap across Financial & Engagement Attributes")
        st.plotly_chart(plot_multivariate_correlation_heatmap(filtered_movies), use_container_width=True, key="chart_fin_corr_heatmap")


# ==============================================================================
# TAB 5: CAST & CREW INTELLIGENCE
# ==============================================================================

with nav_tabs[4]:
    st.header("👥 Cast & Crew Talent Econometrics")

    col_act, col_dir = st.columns(2)
    with col_act:
        st.subheader("Top Prolific Actors in Selection")
        actor_stats = execute_query(f"""
            SELECT c.actor_name, COUNT(DISTINCT c.movie_id) as movie_count, ROUND(AVG(m.vote_average), 2) as avg_rating
            FROM cast c JOIN movies m ON c.movie_id = m.movie_id
            WHERE c.movie_id IN {movie_ids_tuple} AND c.actor_name != 'Unknown'
            GROUP BY c.actor_name ORDER BY movie_count DESC LIMIT 15;
        """)
        fig_act = px.bar(
            actor_stats.sort_values("movie_count", ascending=True),
            x="movie_count",
            y="actor_name",
            orientation="h",
            color="avg_rating",
            color_continuous_scale="Teal",
            labels={"movie_count": "Films Billed", "actor_name": "Actor", "avg_rating": "Avg Rating"}
        )
        fig_act.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig_act, use_container_width=True, key="chart_cast_prolific")

    with col_dir:
        st.subheader("Top Directors by Total Box-Office ($M)")
        dir_stats = get_director_analytics(min_movies=2).head(15)
        fig_dir = px.bar(
            dir_stats.sort_values("total_box_office_m", ascending=True),
            x="total_box_office_m",
            y="director_name",
            orientation="h",
            color="avg_rating",
            color_continuous_scale="Reds",
            labels={"total_box_office_m": "Total Gross ($M)", "director_name": "Director"}
        )
        fig_dir.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig_dir, use_container_width=True, key="chart_cast_directors")

    st.subheader("🔍 Actor Filmography Lookup")
    actor_input = st.text_input("Enter Actor Name (e.g. Leonardo DiCaprio, Scarlett Johansson):", "Leonardo DiCaprio")
    if actor_input:
        filmography_df = get_actor_filmography(actor_input)
        if not filmography_df.empty:
            filmography_df["budget"] = filmography_df["budget"].apply(lambda x: f"${x/1e6:,.1f}M" if x > 0 else "Unknown")
            filmography_df["revenue"] = filmography_df["revenue"].apply(lambda x: f"${x/1e6:,.1f}M" if x > 0 else "Unknown")
            st.dataframe(filmography_df, use_container_width=True, hide_index=True)
        else:
            st.info("No matching credits found for that actor.")


# ==============================================================================
# TAB 6: PREDICTIVE MODELING & SIMULATION
# ==============================================================================

with nav_tabs[5]:
    st.header("🔮 Box-Office Econometric Simulation & Recommendations")

    pred_sub_mode = st.radio("Choose Capability:", ["🔮 Predictive Revenue Forecaster", "🎯 Content-Based Movie Recommender"], horizontal=True, key="radio_pred_mode")

    if pred_sub_mode == "🔮 Predictive Revenue Forecaster":
        st.subheader("Box-Office Revenue Simulation Engine")
        st.markdown("Simulate greenlight scenarios and forecast projected gross earnings using our trained Random Forest Regressor ($R^2 = 0.666$).")

        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            in_budget = st.slider("Planned Production Budget ($ Millions):", 1.0, 350.0, 100.0, 5.0, key="slide_pred_budget")
            in_runtime = st.slider("Theatrical Runtime (Minutes):", 60, 240, 130, 5, key="slide_pred_runtime")
        with cp2:
            in_pop = st.slider("Anticipated TMDB Popularity:", 5.0, 300.0, 80.0, 5.0, key="slide_pred_pop")
            in_votes = st.slider("Projected Total Votes:", 100, 35000, 8000, 500, key="slide_pred_votes")
        with cp3:
            in_rating = st.slider("Target Critical Rating (0-10):", 3.0, 9.5, 7.8, 0.1, key="slide_pred_rating")
            in_year = st.slider("Release Year:", 2024, 2030, 2026, 1, key="slide_pred_year")

        if st.button("🔮 Forecast Box-Office Revenue", use_container_width=True, key="btn_pred_rev"):
            pred_val = predict_single_movie_revenue(
                budget_m=in_budget,
                runtime=in_runtime,
                popularity=in_pop,
                vote_count=in_votes,
                vote_average=in_rating,
                release_year=in_year
            )
            pred_profit = pred_val - in_budget
            pred_roi = (pred_profit / in_budget * 100) if in_budget > 0 else 0.0

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Predicted Gross Revenue", f"${pred_val:,.1f}M")
            with k2:
                st.metric("Projected Net Profit", f"${pred_profit:,.1f}M", delta=f"{'+' if pred_profit >= 0 else ''}${pred_profit:,.1f}M")
            with k3:
                st.metric("Projected ROI", f"{pred_roi:,.1f}%", delta="Capital Yield")

    elif pred_sub_mode == "🎯 Content-Based Movie Recommender":
        st.subheader("Semantic Content-Based Recommendation Engine")
        st.markdown("Enter any movie in the TMDB catalog to uncover structurally similar films based on semantic TF-IDF feature soup (overviews, keywords, genres, and cast).")

        rec_engine = get_recommendation_engine()
        all_m_titles = sorted(movies_df["title"].dropna().unique().tolist())
        target_movie = st.selectbox("Select Target Film:", options=all_m_titles, index=all_m_titles.index("Inception") if "Inception" in all_m_titles else 0, key="select_rec_target")
        top_k = st.slider("Number of Recommendations:", 3, 15, 6, key="slide_rec_topk")

        if st.button("🎯 Generate Recommendations", use_container_width=True, key="btn_gen_rec"):
            recs = rec_engine.recommend(target_movie, top_n=top_k)
            if recs:
                st.markdown(f"### Films Semantically Similar to **{target_movie}**")
                cols = st.columns(3)
                for idx, r in enumerate(recs):
                    col = cols[idx % 3]
                    with col:
                        st.markdown(f"""
                        <div class="kpi-card" style="margin-bottom: 16px;">
                            <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">{r['title']}</div>
                            <div style="color: #E50914; font-weight: 700; font-size: 0.85rem; margin: 4px 0;">{r['similarity_score']}% Semantic Match</div>
                            <div style="color: #94A3B8; font-size: 0.8rem;"><b>Director:</b> {r['director'] or 'Unknown'}</div>
                            <div style="color: #94A3B8; font-size: 0.8rem;"><b>Genres:</b> {r['genres']}</div>
                            <div style="color: #00F59B; font-size: 0.8rem;"><b>Rating:</b> {r['vote_average']} / 10</div>
                            <p style="font-size: 0.78rem; color: #CBD5E1; margin-top: 8px; line-height: 1.3;">{r['overview'][:140]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No matching recommendations found.")
