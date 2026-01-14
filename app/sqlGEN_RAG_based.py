# app/sqlRAG_generator.py

from __future__ import annotations

import os
from typing import List, Dict

from google import genai


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


SYSTEM_PROMPT = """
You are an expert Text-to-SQL assistant.

You MUST obey the RULEBOOK (PolicyDoc) provided below.
PolicyDoc is the authoritative source of truth.
If any conflict exists between user wording, other YAML context, or PolicyDoc, PolicyDoc wins.

Refer to mpl_business_glossary, MetricCatalog, and TableCard YAML files for definitions.
Do NOT invent metrics or columns.
Use only tables and columns declared in TableCard.
Use BigQuery SQL dialect.
No markdown. No explanation. SQL only.

If the question is unclear, empty, or not analytics-related:
→ DO NOT generate SQL
→ Respond with exactly:
  "Invalid analytics question. Please ask about metrics, time range, and dimensions."

Use following case when statements when grouping by 
1.  game_id- CASE
      WHEN game_id = '1002044' THEN 'Win Patti Skill'
      WHEN game_id = '1002099' THEN 'Bingo Skill'
      WHEN game_id = '1000295' THEN 'Solitaire'
      WHEN game_id = '1000220' THEN 'Gin Rummy'
      WHEN game_id = '1002100' THEN 'Draw 4'
      WHEN game_id = '1000243' THEN 'Ultra Ludo'
      WHEN game_id = '1002053' THEN 'Snakes & Ladders'
      WHEN game_id = '1000235' THEN 'Spades'
      WHEN game_id = '1000240' THEN 'Brick Blast'
      WHEN game_id = '1003002' THEN 'Survivor'
      WHEN game_id = '1003011' THEN 'Pool Server'
      ELSE game_id
    END
2. App_type- CASE
      WHEN LOWER(app_type) = 'cash' THEN 'Android'
      WHEN LOWER(app_type) = 'ios' THEN 'iOS'
      ELSE app_type
    END

"""


def _build_prompt(user_query: str, retrieved_chunks: List[Dict]) -> str:
    policy_chunks = [c for c in retrieved_chunks if c.get("doc_type") == "PolicyDoc"]
    other_chunks = [c for c in retrieved_chunks if c.get("doc_type") != "PolicyDoc"]

    rulebook_text = "\n\n".join(c.get("content", "") for c in policy_chunks).strip()

    context_blocks: List[str] = []
    for i, chunk in enumerate(other_chunks, start=1):
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

    retrieved_text = "\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

# RULEBOOK (BINDING) — MUST FOLLOW
{rulebook_text}

# Retrieved Knowledge (Reference)
{retrieved_text}

# User Question
{user_query}

# Task
Generate exactly ONE correct BigQuery SQL query. SQL only.
"""


def generate_bq_sql(user_query: str, retrieved_chunks: List[Dict]) -> str:
    prompt = _build_prompt(user_query, retrieved_chunks)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return (response.text or "").strip()
