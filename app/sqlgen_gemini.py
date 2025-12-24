import os
import re
import google.generativeai as genai

# Safety blocklist for DDL/DML; adjust as needed.
_BLOCKLIST = re.compile(r"(?is)\b(drop|delete|truncate|update|insert|merge|alter|create|grant|revoke)\b")

def _load_schema_prompt() -> str:
    """Load long schema/policy prompt from file so code stays clean."""
    path = os.getenv("SQL_SCHEMA_PROMPT_PATH", "schema_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Keep running even if prompt file missing.
        return ""

def generate_bq_sql(user_question: str) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    system_instr = (
        "You generate ONLY a BigQuery SQL query.\n"
        "Return ONLY SQL, no explanation, no markdown.\n"
        "Must start with SELECT.\n"
        "Always add a LIMIT 1000 unless user explicitly asks for all rows.\n"
    )

    schema_prompt = _load_schema_prompt()

    prompt = f"{system_instr}\n\n{schema_prompt}\n\nUser question:\n{user_question}\n\nSQL:"
    resp = model.generate_content(prompt)
    sql = (getattr(resp, "text", None) or "").strip()

    # Basic guards
    if not sql:
        raise ValueError("Model returned empty SQL.")
    if _BLOCKLIST.search(sql):
        raise ValueError("Request not allowed: generated SQL contains DDL/DML keywords.")
    if not re.match(r"(?is)^\s*select\b", sql):
        raise ValueError("Request not allowed: response is not a SELECT query.")

    return sql
