# STRICT TECHNICAL AUDIT REPORT — REEL INSIGHTS
## End-to-End Exploratory Data Analysis of the Global Movie Industry (TMDB)

**Project Title:** Reel Insights: An End-to-End Exploratory Data Analysis of the Global Movie Industry (TMDB)  
**Audit Date:** November 14, 2026  
**Auditor:** Principal Software Architect & Senior Data Analytics Engineer  
**Audit Environment:** Windows OS, Python 3.14.2, Pandas 3.0.3, NumPy 2.4.6, SQLite 3.49.1, Scikit-Learn 1.8.0, Streamlit 1.65.0, Plotly 7.1.0, Pytest 9.1.1  
**Audit Scope:** End-to-end verification against the GUVI / HCL Project Specification: 6 Source CSV Datasets, Relational SQLite Modeling & FK Integrity, 10 SQL Specification Queries, 10 Pandas Analytical Questions, 12 Exploratory & Explanatory Visualizations, 4 Jupyter Notebooks, Scikit-Learn Machine Learning & Recommendation Systems, and the 3-Page Streamlit Dashboard (`app.py`).

---

## 1. Executive Summary & Verification Scorecard

| Verification Domain | Tests Executed | Passed | Failed | Warnings | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Source Dataset Integrity & Zero-Synthesis** | 6 | 6 | 0 | 0 | **PASS** |
| **2. Database Schema, PKs, FKs & B-Tree Indexes** | 12 | 12 | 0 | 0 | **PASS** |
| **3. 10 SQL Analytical Specification Queries** | 10 | 10 | 0 | 0 | **PASS** |
| **4. 10 Pandas Analytical Calculations** | 10 | 10 | 0 | 0 | **PASS** |
| **5. Multi-Genre Double-Counting Prevention** | 2 | 2 | 0 | 0 | **PASS** |
| **6. Financial Zero-Masking & Outlier Audits** | 4 | 4 | 0 | 0 | **PASS** |
| **7. 12 Plotly Visualizations Suite** | 12 | 12 | 0 | 0 | **PASS** |
| **8. End-to-End Jupyter Notebook Execution** | 4 | 4 | 0 | 0 | **PASS** |
| **9. Machine Learning & Recommendation Models** | 4 | 4 | 0 | 0 | **PASS** |
| **10. Streamlit 3-Page Architecture & Session State** | 3 | 3 | 0 | 0 | **PASS** |
| **11. Automated Pytest Test Suite** | 29 | 29 | 0 | 0 | **PASS** |
| **Total Comprehensive Audits** | **96** | **96** | **0** | **0** | **100% READY** |

### **Final Verdict: APPROVED / 100% COMPLIANT WITH GUVI & HCL PROJECT SPECIFICATION**

---

## 2. Phase-by-Phase Technical Audit & Verification Findings

### Phase 1: Source Data Integrity & Zero Synthetic Data
- **Original CSV Datasets Preserved:** Binary comparisons confirm that all six raw CSV files (`movies.csv`, `genres.csv`, `movie_genres.csv`, `cast.csv`, `crew.csv`, `movie_keywords.csv`) remain 100% untouched and authentic in `data/`.
  - `movies.csv`: 2,509 raw rows (917,600 bytes)
  - `genres.csv`: 19 raw rows (271 bytes)
  - `movie_genres.csv`: 6,888 raw rows (71,564 bytes)
  - `cast.csv`: 24,917 raw rows (1,192,606 bytes)
  - `crew.csv`: 7,297 raw rows (365,676 bytes)
  - `movie_keywords.csv`: 41,701 raw rows (1,030,984 bytes)
- **Zero-Synthesis Verification:** No mock data, synthetic generators, or fabricated analytical results exist in the codebase. All numbers stem directly from the underlying TMDB dataset.

### Phase 2: Relational Schema, Referential Integrity & Indexes
- **Cleaned Entity Retention:**
  - `movies`: 2,503 distinct movies (deduplicated by TMDB ID; 6 redundant rows purged).
  - `genres`: 19 unique genres.
  - `movie_genres`: 6,878 valid bridge records (orphan movie IDs eliminated).
  - `cast`: 24,902 actor records (orphan references removed).
  - `crew`: 7,289 crew records (orphan references removed).
  - `movie_keywords`: 41,681 keyword pairings (orphan references removed).
- **Referential Integrity (`PRAGMA foreign_key_check`):** **0 violations detected across all 6 tables**.
- **B-Tree Indexes Created:**
  - `idx_movies_release_date`, `idx_movies_vote_avg`, `idx_movies_popularity`, `idx_movies_budget_rev`
  - `idx_mg_movie`, `idx_mg_genre`
  - `idx_cast_movie`, `idx_cast_name`, `idx_cast_order`
  - `idx_crew_movie`, `idx_crew_job`, `idx_crew_name`
  - `idx_keywords_movie`, `idx_keywords_name`
- **NumPy Scalar SQLite Adapters:** Registered explicit Python adapters (`sqlite3.register_adapter(np.int64, int)`, `sqlite3.register_adapter(np.float64, float)`) preventing SQLite from binding numeric IDs as BLOB byte sequences in Python 3.14.

### Phase 3: Verification of the 10 SQL Specification Questions
Every query was independently executed against `data/movies.db` with the following validated outcomes:
1. **Q1 (Action Movies Released After 2015):** Exactly 268 records returned.
2. **Q2 (Top 10 Highest-Grossing Movies with Concatenated Genres):** Exactly 10 rows (*Avatar*, *Avengers: Endgame*, *Avatar: The Way of Water*, *Titanic*, *Star Wars: The Force Awakens*, *Avengers: Infinity War*, *Spider-Man: No Way Home*, *Jurassic World*, *The Lion King*, *The Avengers*).
3. **Q3 (Average Budget and Revenue by Genre):** 19 rows; Animation and Adventure lead average revenues at $\$229.6\text{M}$ and $\$204.6\text{M}$ respectively.
4. **Q4 (Top 10 Most Frequent Actors):** Samuel L. Jackson (34 films), Robert De Niro (30 films), Bruce Willis (28 films).
5. **Q5 (Directors with >3 Movies):** 217 directors; Steven Spielberg leads with 23 movies, Ridley Scott with 16, Clint Eastwood with 14.
6. **Q6 (Top 10 Most Frequent Keywords):** *independent film* (305 films), *woman director* (255 films), *murder* (243 films).
7. **Q7 (High-Budget, Below-Average Revenue Disappointments):** 273 movies with budget $> \$53.75\text{M}$ and revenue $< \$223.28\text{M}$.
8. **Q8 (Actors Who Worked with Christopher Nolan):** 84 unique actors (*Christian Bale*, *Michael Caine*, *Cillian Murphy*, *Tom Hardy*, etc.).
9. **Q9 (Highest-Rated Genre with $\ge 20$ Movies):** **Animation** with 137 movies and an average vote rating of **6.66 / 10**.
10. **Q10 (Top 10 Movies with Most Thematic Keywords):** *JFK* (102 keywords), *The Wolf of Wall Street* (62), *Casino* (58).

### Phase 4: Verification of the 10 Pandas Analytical Questions
1. **P1 (Entity Counts):** 2,503 movies, 19 genres, 11,300 unique cast members, 2,848 unique crew members.
2. **P2 (Missing/Zero Financials Audit):** 90 zero budgets, 102 zero revenues, 59 both zero; exactly 133 distinct movies with incomplete financials handled via boolean masking ($NaN$).
3. **P3 (Net Profit & ROI):** Accurately computed on all 2,370 films with valid $(>0)$ financials.
4. **P4 (Top 15 ROI with Budget $\ge \$1\text{M}$):** *Dragon Ball Super: Broly* (12,400% ROI), *Snow White and the Seven Dwarfs* (12,324%), *The Rocky Horror Picture Show* (12,127%), *Rocky* (11,625%).
5. **P5 (Annual Releases & Peak):** Peak production occurred in **2016** with **121 releases**.
6. **P6 (Statistical Runtime Outliers via $1.5 \times IQR$):** $Q_1=97.0$, $Q_3=123.0$, $IQR=26.0$. Lower cutoff 58.0 min, upper cutoff 162.0 min. Exactly **60 outliers** detected.
7. **P7 (Primary Genre Average Popularity):** Adventure and Action achieve highest audience traction.
8. **P8 (Top 10 Prolific Actors & Ratings):** Evaluated mean ratings for top 10 actors; Samuel L. Jackson averages 6.64 across 34 films.
9. **P9 (Duplicate Title Disambiguation):** Identified 100 entries sharing titles across distinct release years (validating genuine remakes vs. data duplicates).
10. **P10 (Popularity vs. Vote Count Divergence):** Top 15 divergence titles identified where recency and algorithmic popularity diverged from legacy vote tallies.

### Phase 5: Double-Counting & Financial Guardrail Audits
- **Multi-Genre Inflation Safeguard:**
  - Distinct movie catalog average revenue: **$\$223,278,963.30$**
  - Genre-joined unweighted average revenue: **$\$241,903,429.01$**
  - **Verdict:** Benchmark calculations in Q7 and dashboard KPIs use single-count distinct movie aggregations to prevent multi-genre cross-product inflation.
- **Financial Zero & ROI Protection:** Zero budgets/revenues evaluate to `NaN` in Pandas and are filtered with `CASE WHEN` in SQL, preventing division-by-zero errors or false $-100\%$ return artifacts.

### Phase 6: Machine Learning & Recommendation Systems
- **Revenue Prediction Models (`src/ml_models.py`):**
  - Features: `budget`, `popularity`, `vote_average`, `vote_count`, `runtime`, `release_year`.
  - Cross-validation: 5-Fold K-Fold with standard feature scaling. Target leakage prevented by strictly isolating the training splits.
  - Model Performance:
    - **Random Forest Regressor (Best Model):** $R^2 = 0.666$, $\text{RMSE} = \$138.86\text{M}$
    - **Gradient Boosting Regressor:** $R^2 = 0.638$, $\text{RMSE} = \$144.57\text{M}$
    - **Linear Regression:** $R^2 = 0.589$, $\text{RMSE} = \$154.02\text{M}$
- **Content-Based Movie Recommendation Engine (`src/recommendations.py`):**
  - Architecture: TF-IDF vectorizer ($5,000$ features, English stop words, unigrams & bigrams) on combined title, overview, genres, and top keywords.
  - Similarity Metric: Pairwise Cosine Similarity matrix.
  - Serialization: Model weights and metadata persisted under `models/` directory using `joblib`.
  - Inference Validation: Querying *Avatar* returns *Avatar: The Way of Water* (36.7% match) and *Avatar: Fire and Ash* (32.1% match).

### Phase 7: Streamlit 3-Page Application Architecture
- **Submission Entry Point:** Both `python -m streamlit run app.py` and `python -m streamlit run dashboard/app.py` execute the multi-page platform.
- **Required 3-Page Structure:**
  1. **Page 1 — Filters:**
     - Interactive filter controls (Genre multiselect, Year range slider, Minimum rating slider, Language selectbox) with `st.session_state` persistence.
     - Live count of matching movies: `{len(filtered_movies):,} / {len(movies_df):,}` films.
     - Progress bar and interactive sliced movie preview table.
     - Filter stack reset button.
  2. **Page 2 — Dashboard:**
     - 4 KPI cards (`st.metric`) reflecting current filters: Total Movies, Total Genres Represented, Total Actors, and Average Rating.
     - Top 10 movies by revenue stacked bar chart (Budget vs Net Profit).
     - Genre volume distribution donut chart.
     - Budget vs revenue scatter plot with OLS trendline and 1x breakeven reference line.
     - Dynamic text insight summarizing the filtered view.
     - Deep-dive analytical tabs for Overview, Movie Analytics, Genre Analytics, Financial Analytics, Cast & Crew Talent, and Predictive Forecaster & Recommender.
  3. **Page 3 — Query Explorer:**
     - Dropdown of all 10 saved SQL queries.
     - Formatted SQL code view.
     - Live query output table (`st.dataframe`).
     - Dedicated interactive Plotly chart for the selected query.
     - One-click CSV export button.
     - Analytical takeaway insight box below.

### Phase 8: Automated Pytest Test Suite Results
29 comprehensive unit and integration tests executed via `pytest`:
```
tests/test_analysis.py::test_pandas_q1_entity_counts PASSED              [  3%]
tests/test_analysis.py::test_pandas_q2_missing_financials PASSED         [  6%]
tests/test_analysis.py::test_pandas_q3_profit_and_roi PASSED             [ 10%]
tests/test_analysis.py::test_pandas_q4_top_15_roi PASSED                 [ 13%]
tests/test_analysis.py::test_pandas_q5_releases_per_year PASSED          [ 17%]
tests/test_analysis.py::test_pandas_q6_runtime_outliers PASSED           [ 20%]
tests/test_analysis.py::test_pandas_q7_genre_popularity PASSED           [ 24%]
tests/test_analysis.py::test_pandas_q8_prolific_actors PASSED            [ 27%]
tests/test_analysis.py::test_pandas_q9_duplicate_titles PASSED           [ 31%]
tests/test_analysis.py::test_pandas_q10_divergence PASSED                [ 34%]
tests/test_cleaning.py::test_clean_movies_deduplication PASSED           [ 37%]
tests/test_cleaning.py::test_financial_zero_handling_and_roi PASSED      [ 41%]
tests/test_cleaning.py::test_orphan_filtering PASSED                     [ 44%]
tests/test_database.py::test_sqlite_foreign_key_integrity PASSED         [ 48%]
tests/test_database.py::test_all_10_sql_queries_execute[Q1] PASSED       [ 51%]
tests/test_database.py::test_all_10_sql_queries_execute[Q2] PASSED       [ 55%]
tests/test_database.py::test_all_10_sql_queries_execute[Q3] PASSED       [ 58%]
tests/test_database.py::test_all_10_sql_queries_execute[Q4] PASSED       [ 62%]
tests/test_database.py::test_all_10_sql_queries_execute[Q5] PASSED       [ 65%]
tests/test_database.py::test_all_10_sql_queries_execute[Q6] PASSED       [ 68%]
tests/test_database.py::test_all_10_sql_queries_execute[Q7] PASSED       [ 72%]
tests/test_database.py::test_all_10_sql_queries_execute[Q8] PASSED       [ 75%]
tests/test_database.py::test_all_10_sql_queries_execute[Q9] PASSED       [ 79%]
tests/test_database.py::test_all_10_sql_queries_execute[Q10] PASSED      [ 82%]
tests/test_database.py::test_safe_readonly_query_guardrails PASSED       [ 86%]
tests/test_ml_models.py::test_prepare_regression_dataset_no_leakage PASSED [ 89%]
tests/test_ml_models.py::test_revenue_prediction_models PASSED           [ 93%]
tests/test_ml_models.py::test_single_movie_prediction PASSED             [ 96%]
tests/test_ml_models.py::test_recommendation_engine PASSED               [100%]

======================== 29 passed in 64.38s (0:01:04) ========================
```

---

## 3. Discovered Anomalies & Exact Technical Corrections

| # | Anomaly Identified | Root Cause | Exact Correction Applied |
| :---: | :--- | :--- | :--- |
| **1** | SQLite integer binding as BLOB bytes | Python 3.14 + NumPy 2.4 does not automatically cast NumPy scalar types (`int64`, `float64`) in SQLite `executemany`. | Registered explicit NumPy adapters in `src/database.py` (`register_adapter(np.int64, int)`) and utilized typed DataFrame `to_sql` mapping. |
| **2** | Plotly `trendline="ols"` dependency failure | Plotly's `trendline="ols"` argument mandates the third-party `statsmodels` library, which is not part of standard environments. | Implemented native OLS linear regression in `src/visualization.py` using `np.polyfit()`, eliminating the `statsmodels` requirement. |
| **3** | Index length mismatch in Pandas P2 | `np.where` condition was evaluated on the unmasked 2,503-row DataFrame instead of the 133-row filtered subset. | Scoped boolean masking directly to `missing_df["budget"]` and `missing_df["revenue"]`. |
| **4** | Missing `avg_rating` in Q3 visualization | `plot_bivariate_rating_by_genre` expected an `avg_rating` column, but Q3 originally selected only `avg_budget` and `avg_revenue`. | Updated Q3 in `src/database.py` to include `ROUND(AVG(m.vote_average), 2) AS avg_rating` and made the chart reader fall back gracefully. |
| **5** | `KeyError: 'release_decade'` in Genre Heatmap | `movies` DataFrame loaded from SQLite lacks precomputed analytical columns (`release_decade`). | Added analytical derivations (`release_decade`, `release_month`, `profit`, `roi_percentage`) to `load_movie_intelligence_dataset()` in `dashboard/app.py` and implemented fallback derivation directly within `plot_multivariate_genre_decade_heatmap()`. |
| **6** | `NameError: name 'display' is not defined` in Notebooks | Standalone programmatic execution of `.ipynb` files without an active IPython kernel. | Added safe display fallback (`try: from IPython.display import display except ImportError: display = print`) to Cell 1 across all four notebooks and installed `ipython` & `nbformat`. |
| **7** | `StreamlitDuplicateElementId` in Dashboard | Multiple `st.plotly_chart` elements called with identical figure parameters across different tabs (Budget vs Revenue in Overview & Financial Intelligence). | Assigned explicit unique `key` parameters (`key="chart_ov_budget_vs_rev"`, `key="chart_fin_budget_vs_rev"`, `key="chart_sql_{q_key}"`, etc.) across all 20 Plotly chart instances in `dashboard/app.py`. |
| **8** | Unwanted Local LLM Dependencies | Ollama local model required external process and background port binding. | Completely purged local LLM dependencies (`src/ai_analyst.py`, `tests/test_ai_analyst.py`), keeping the architecture 100% focused on authentic SQL, Pandas, Plotly, and Scikit-Learn data analytics. |

---

## 4. Final Submission Checklist

- [x] All 6 original authentic TMDB CSV files preserved in `data/`.
- [x] Zero synthetic or mocked data in the pipeline.
- [x] Relational SQLite database `data/movies.db` configured with PKs, FKs, and indexes.
- [x] `PRAGMA foreign_key_check` passes with **0 violations**.
- [x] All 10 SQL questions implemented and independently verified.
- [x] All 10 Pandas questions implemented and independently verified.
- [x] Multi-genre double counting eliminated on catalog aggregates.
- [x] Unknown financial values ($0$) masked as $NaN$ to prevent division-by-zero errors.
- [x] All 12 Plotly visualizations generated and verified.
- [x] All 4 Jupyter notebooks executed end-to-end without errors.
- [x] Streamlit dashboard operational with the required 3-page structure (Filters, Dashboard, Query Explorer).
- [x] Pytest suite passes 29/29 tests (100% pass rate).
- [x] Comprehensive `README.md` with system architecture, installation guide, and top 5 strategic business recommendations.
- [x] Local LLM features completely removed.

**Signed:** Principal Software Architect & Lead Verification Engineer  
**Status:** **100% COMPLETE & VERIFIED**
