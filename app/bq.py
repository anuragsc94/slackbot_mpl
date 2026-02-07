import re
from google.cloud import bigquery
import pandas as pd

CSV_MAX_ROWS = 5000  # hard cap; override via env in slack_app.py if you want.

def _remove_trailing_limit(sql: str) -> str:
    """Remove a trailing LIMIT N at end of query (with optional semicolon)."""
    return re.sub(r"\s+LIMIT\s+\d+\s*;?\s*$", "", (sql or "").strip(), flags=re.IGNORECASE)

def run_sql_df(sql: str, max_rows: int = 500, project_id: str | None = None) -> pd.DataFrame:
    """Run SQL with enforced LIMIT and return dataframe.

    Notes:
      - If `project_id` is provided, it becomes the default project for query execution.
      - This allows one process to serve multiple Slack bots, each tied to a distinct
        BigQuery project, while keeping SQL/table names unchanged (dataset.table).
    """
    client = bigquery.Client(project=project_id) if project_id else bigquery.Client()

    max_rows = int(max_rows)
    max_rows = min(max_rows, CSV_MAX_ROWS)

    sql_limited = f"{_remove_trailing_limit(sql)}\nLIMIT {max_rows}"
    results = client.query(sql_limited).result()
    return results.to_dataframe()
