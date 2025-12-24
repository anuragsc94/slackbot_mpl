# scope_fallback.py
# Centralized “out-of-scope” reply + lightweight guardrails for your Slack analytics bot.

from __future__ import annotations

from typing import Tuple

SCOPE_FALLBACK_MSG = """
Sorry — I can’t answer this yet. 🙏  

This request goes beyond my current analytics scope, but this capability is under active development.

⚠️ Important note on analysis scope
• I currently support aggregated metrics over a selected time period
• I do not support cohort-level analysis yet (e.g. user retention cohorts, LTV cohorts, funnel cohorts)

Here’s what I can help you with right now 👇  

📊 Metrics I support
• Gameplay (GPs) / Battles (count)
• Active Users (unique users)
• GMV
• Winnings
• MPL Margin
• TE, IAE
• GM (Gross Margin)
• Tax
• CM1 (Contribution Margin)
• Bonus Cash utilised (BC utilised)
• Deposit Cash utilised (DC utilised)
• Winning Cash utilised (WC utilised)
• Entry Fee
• Win Rate (Won GPs / Total GPs)
• RTP (Cash Winnings / GMV)
• Per-user metrics (GMVPU, MarginPU, CM1PU)
• Rank1, rank2,rank3 targets and actuals for HOF Lobbies

🧩 Dimensions I support
• Date (dt)
• Game Names
• Country (country_code)
• Platform (Android vs iOS)
• Lobby Type (Cash vs Gold)
• HOF vs Non-HOF
• Entry Fee buckets
• ZSH(free) vs Paid GPs

⏱ Time filters I support
• Today / Yesterday
• Last N days
• Date ranges
• Weekly (Monday start) & Monthly views

Try rephrasing your question using the metrics + dimensions above, and I’ll be happy to help 🙂  

Here are some sample questions you can ask me:
    "GMV and CM1 by game for last 7 days in country US"
    "Win rate by Android vs iOS for Cash lobbies yesterday in country US"
    "Top 5 games by GMV for US in last 14 days in country US"
    "Daily GPs and Active Users for Bingo Skill in last 30 days in country US"
    "CM1 and GM by country for Cash vs Gold lobbies last week in country US"
    "Entry fee wise win rate for HOF lobbies yesterday in country US"
    "RTP by game and country for last 7 days"
    "Weekly GMV trend for Solitaire in last 8 weeks"
    "Average entry fee and win rate by game for Cash lobbies last 7 days"
    "Per-user GMV and CM1 by platform (Android vs iOS) for last 14 days"


""".strip()

_INVALID_EXACT_PREFIX = "invalid analytics question"


def should_fallback(user_question: str) -> Tuple[bool, str]:
    """
    Returns (True, reason) if we should respond with SCOPE_FALLBACK_MSG instead of generating SQL.
    Keep this *lightweight* and deterministic.
    """
    q = (user_question or "").strip().lower()

    if not q or q in {"hi", "hello", "hey", "none", "test"}:
        return True, "empty_or_greeting"

    # Explicitly block cohort-style analysis (you said not supported)
    cohort_keywords = ("cohort", "retention", "ltv", "funnel cohort", "day 1", "d1", "d7", "d30")
    if any(k in q for k in cohort_keywords):
        return True, "cohort_not_supported"

    # Block obvious “dump table” requests
    dump_keywords = ("select *", "show everything", "entire table", "all columns", "schema", "what is this table")
    if any(k in q for k in dump_keywords):
        return True, "table_dump_or_schema_request"

    return False, "ok"


def sql_is_invalid_message(sql: str) -> bool:
    """
    If your LLM is instructed to return a specific invalid message,
    detect it here and trigger fallback.
    """
    s = (sql or "").strip().lower()
    return s.startswith(_INVALID_EXACT_PREFIX)
