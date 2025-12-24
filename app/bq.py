import re
from google.cloud import bigquery
import pandas as pd

CSV_MAX_ROWS = 5000  # hard cap; override via env in slack_app.py if you want.

def _remove_trailing_limit(sql: str) -> str:
    """Remove a trailing LIMIT N at end of query (with optional semicolon)."""
    return re.sub(r"\s+LIMIT\s+\d+\s*;?\s*$", "", (sql or "").strip(), flags=re.IGNORECASE)

def run_sql_df(sql: str, max_rows: int = 500) -> pd.DataFrame:
    """Run SQL with enforced LIMIT and return dataframe."""
    client = bigquery.Client()  # uses attached service account creds on Dataproc/Compute Engine

    max_rows = int(max_rows)
    max_rows = min(max_rows, CSV_MAX_ROWS)

    sql_limited = f"{_remove_trailing_limit(sql)}\nLIMIT {max_rows}"
    results = client.query(sql_limited).result()
    return results.to_dataframe()
