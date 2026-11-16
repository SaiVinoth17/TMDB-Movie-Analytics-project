"""
app.py - Root Entry Point for ReelInsights TMDB Movie Analytics.
Enables running the application via both `streamlit run app.py` and `streamlit run dashboard/app.py`.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Execute the main dashboard application
dashboard_app = PROJECT_ROOT / "dashboard" / "app.py"
with open(dashboard_app, "r", encoding="utf-8") as f:
    app_code = f.read()

exec(compile(app_code, str(dashboard_app), "exec"), globals())

