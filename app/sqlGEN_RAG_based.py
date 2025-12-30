from __future__ import annotations

import os
from typing import List, Dict
from google import genai


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


# ------------------------------------------------------------------
# SYSTEM PROMPT — POLICY-DRIVEN
# ------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an MPL Analytics Text-to-SQL engine.

Refer to mpl_business_glossary, MetricCatalog, and PolicyDoc YAML files for definitions.
Follow all rules defined in PolicyDoc.
Do NOT invent metrics or columns.
Use only tables and columns declared in TableCard.
Use BigQuery SQL dialect.
No markdown. No explanation. SQL only.

If the question is unclear, empty, or not analytics-related:
→ DO NOT generate SQL
→ Respond with exactly:
  "Invalid analytics question. Please ask about metrics, time range, and dimensions."
"""


# ------------------------------------------------------------------
# PROMPT BUILDER — RAG CONTEXT IS ADDED HERE
# ------------------------------------------------------------------

def _build_prompt(user_query: str, retrieved_chunks: List[Dict]) -> str:
    context_blocks: List[str] = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "")
        doc_type = chunk.get("doc_type", "")
        chunk_type = chunk.get("chunk_type", "")
        content = chunk.get("content", "")

        context_blocks.append(
            f"""--- Context {i} ---
source: {source}
doc_type: {doc_type}
chunk_type: {chunk_type}

{content}
"""
        )

    retrieved_text = "\n".join(context_blocks)  # ✅ compute outside f-string

    return f"""{SYSTEM_PROMPT}

# Retrieved Knowledge (YAML)
{retrieved_text}

# User Question
{user_query}

# Task
Generate the correct BigQuery SQL query.
"""


# ------------------------------------------------------------------
# FINAL PUBLIC FUNCTION — QUERY_ORCHESTRATOR CALLS THIS
# ------------------------------------------------------------------

def generate_bq_sql(user_query: str, retrieved_chunks: List[Dict]) -> str:
    prompt = _build_prompt(user_query, retrieved_chunks)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return (response.text or "").strip()
