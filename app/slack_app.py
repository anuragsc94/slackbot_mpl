import os
import re
import tempfile
import logging
from typing import Optional

import pandas as pd
from tabulate import tabulate
from slack_bolt import App
from slack_sdk.web.client import WebClient

from app.bq import run_sql_df
from app.sql_summary import summarize_sql_for_user_sqlglot, format_sql_summary_for_slack
from app.scope_fallback import should_fallback, SCOPE_FALLBACK_MSG

# ✅ RAG + SQL orchestrator
from app.query_orchestrator import handle_query_with_rag


logger = logging.getLogger("slackbot")

CSV_MAX_ROWS = int(os.getenv("CSV_MAX_ROWS", "5000"))
PREVIEW_ROWS = int(os.getenv("PREVIEW_ROWS", "20"))


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _strip_mention(text: str) -> str:
    """Remove the <@U123> mention prefix Slack adds to app_mention events."""
    return re.sub(r"<@[^>]+>\s*", "", (text or "")).strip()


def _df_to_slack_table(df: pd.DataFrame, max_rows: int = 20, max_chars: int = 2800) -> str:
    if df is None or df.empty:
        return "No rows returned."
    out = tabulate(df.head(max_rows), headers="keys", tablefmt="github", showindex=False)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n…(truncated)"
    return f"```{out}```"


def _upload_df_csv(
    client: WebClient,
    df: pd.DataFrame,
    channel: str,
    thread_ts: Optional[str],
    filename_prefix: str = "result",
) -> None:
    if df is None or df.empty:
        return

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", prefix=f"{filename_prefix}_", delete=False) as f:
        df.to_csv(f.name, index=False)
        tmp_path = f.name

    try:
        client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            file=tmp_path,
            filename=os.path.basename(tmp_path),
            title=f"{filename_prefix}.csv",
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------
# Slack App Builder
# ---------------------------------------------------------------------

def build_slack_app(slack_bot_token: str) -> App:
    app = App(token=slack_bot_token)

    # ================================================================
    # APP MENTION HANDLER (@bot ...)
    # ================================================================
    @app.event("app_mention")
    def on_mention(event, say, client, logger):
        channel = event.get("channel")
        thread_ts = event.get("ts")

        User_question_on_slack = _strip_mention(event.get("text"))
        logger.info(f"📩 app_mention received: {User_question_on_slack}")

        # Pre-SQL fallback (UNCHANGED)
        fallback, _ = should_fallback(User_question_on_slack)
        if fallback:
            say(text=SCOPE_FALLBACK_MSG, channel=channel, thread_ts=thread_ts)
            return

        if not User_question_on_slack:
            say(
                text="Ask me like: `@bot GMV and CM1 by dt for last 7 days`",
                channel=channel,
                thread_ts=thread_ts,
            )
            return

        try:
            ok, sql, err = handle_query_with_rag(User_question_on_slack)

            if not ok:
                say(text=err, channel=channel, thread_ts=thread_ts)
                return

            summary = summarize_sql_for_user_sqlglot(
                sql, user_question=User_question_on_slack
            )
            say(
                text=format_sql_summary_for_slack(summary),
                channel=channel,
                thread_ts=thread_ts,
            )

            df_preview = run_sql_df(sql, max_rows=PREVIEW_ROWS)
            df_csv = run_sql_df(sql, max_rows=CSV_MAX_ROWS)

            say(
                text=f"*Generated SQL:*\n```{sql}```",
                channel=channel,
                thread_ts=thread_ts,
            )

            say(
                text=f"*Preview (top {PREVIEW_ROWS} rows):*\n"
                     f"{_df_to_slack_table(df_preview, max_rows=PREVIEW_ROWS)}",
                channel=channel,
                thread_ts=thread_ts,
            )

            _upload_df_csv(
                client=client,
                df=df_csv,
                channel=channel,
                thread_ts=thread_ts,
                filename_prefix="csv_result",
            )

        except Exception as e:
            logger.exception("Handler failed")
            say(text=f"❌ Error: {e}", channel=channel, thread_ts=thread_ts)

    # ================================================================
    # MESSAGE HANDLER (non-mention messages)
    # ================================================================
    @app.event("message")
    def on_message(event, say, client, logger):
        if event.get("bot_id"):
            return

        channel = event.get("channel")
        thread_ts = event.get("ts")

        User_question_on_slack = _strip_mention(event.get("text"))
        logger.info(f"✅ message event received: {User_question_on_slack}")

        # Simple greeting response (UNCHANGED behavior)
        if User_question_on_slack.lower() in {"hi", "hello", "hey"}:
            say(
                text="Hi! I’m alive ✅ Try: `GMV and CM1 by dt for last 7 days`",
                channel=channel,
                thread_ts=thread_ts,
            )
            return

        try:
            ok, sql, err = handle_query_with_rag(User_question_on_slack)

            if not ok:
                say(text=err, channel=channel, thread_ts=thread_ts)
                return

            summary = summarize_sql_for_user_sqlglot(
                sql, user_question=User_question_on_slack
            )
            say(
                text=format_sql_summary_for_slack(summary),
                channel=channel,
                thread_ts=thread_ts,
            )

            df_preview = run_sql_df(sql, max_rows=PREVIEW_ROWS)
            df_csv = run_sql_df(sql, max_rows=CSV_MAX_ROWS)

            say(
                text=f"*Generated SQL:*\n```{sql}```",
                channel=channel,
                thread_ts=thread_ts,
            )

            say(
                text=f"*Preview (top {PREVIEW_ROWS} rows):*\n"
                     f"{_df_to_slack_table(df_preview, max_rows=PREVIEW_ROWS)}",
                channel=channel,
                thread_ts=thread_ts,
            )

            _upload_df_csv(
                client=client,
                df=df_csv,
                channel=channel,
                thread_ts=thread_ts,
                filename_prefix="csv_result",
            )

        except Exception as e:
            logger.exception("Message handler failed")
            say(text=f"❌ Error: {e}", channel=channel, thread_ts=thread_ts)

    return app
