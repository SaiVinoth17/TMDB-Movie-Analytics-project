"""
tests/test_database.py
Tests for SQLite relational integrity, 10 SQL analytical questions, and safe query execution.
"""

import sqlite3
import pytest
import pandas as pd
from src.config import DB_PATH
from src.database import (
    SQL_QUERIES,
    execute_query,
    execute_safe_readonly_query,
    get_connection,
    run_sql_question
)


def test_sqlite_foreign_key_integrity():
    with get_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_key_check;")
        violations = cursor.fetchall()
        assert len(violations) == 0, f"Foreign key check failed with violations: {violations}"


@pytest.mark.parametrize("q_key", [f"Q{i}" for i in range(1, 11)])
def test_all_10_sql_queries_execute(q_key):
    # Verify that every SQL question executes and returns valid records
    df = run_sql_question(q_key, db_path=DB_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, f"Query {q_key} returned 0 rows!"


def test_safe_readonly_query_guardrails():
    # Test that valid SELECT queries pass
    ok, df, msg = execute_safe_readonly_query("SELECT title, release_date FROM movies LIMIT 5;", db_path=DB_PATH)
    assert ok is True
    assert len(df) == 5

    # Test that dangerous mutation queries are blocked
    forbidden = [
        "DROP TABLE movies;",
        "DELETE FROM cast WHERE movie_id = 1;",
        "UPDATE movies SET title = 'Hacked';",
        "INSERT INTO genres VALUES (999, 'Test');",
        "PRAGMA writable_schema = 1;"
    ]
    for bad_sql in forbidden:
        ok, df, msg = execute_safe_readonly_query(bad_sql, db_path=DB_PATH)
        assert ok is False, f"Expected security block for: {bad_sql}"
        assert "Security rejection" in msg

