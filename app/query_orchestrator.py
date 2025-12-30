# fallback checks(for invalid user prompt), RAG retrieval, SQL generation

from __future__ import annotations
from typing import Tuple, Optional

from rag.rag_store import retrieve_context
from app.sqlGEN_RAG_based import generate_bq_sql
from app.scope_fallback import should_fallback, SCOPE_FALLBACK_MSG, sql_is_invalid_message


def handle_query_with_rag(user_question: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Orchestrates:
      - fallback checks
      - RAG retrieval
      - SQL generation

    Returns:
      (ok, sql, error_msg)

      ok = True  → sql is valid
      ok = False → error_msg should be shown to user
    """

    # 1️⃣ PRE-SQL fallback (existing logic, unchanged)
    fallback, _ = should_fallback(user_question)
    if fallback:
        return False, None, SCOPE_FALLBACK_MSG

    # 2️⃣ Retrieve YAML context (RAG)
    retrieved_chunks = retrieve_context(user_question, k=10)

    if not retrieved_chunks:
        return False, None, (
            "I couldn’t find enough context to answer this question safely.\n"
            "Please rephrase or be more specific."
        )

    # 3️⃣ Generate SQL using existing generator (RAG-augmented)
    sql = generate_bq_sql(
        user_query=user_question,
        retrieved_chunks=retrieved_chunks,
    )

    # 4️⃣ POST-SQL fallback (existing safety net)
    if sql_is_invalid_message(sql):
        return False, None, SCOPE_FALLBACK_MSG

    return True, sql, None
