# rag/chunker.py

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# -----------------------------
# Config
# -----------------------------
RAG_DIR = Path(__file__).resolve().parent
RAG_DOCS_DIR = RAG_DIR  # YAMLs are directly in rag/ folder
CHROMA_DIR = RAG_DIR / "vector_db_chroma"
COLLECTION_NAME = "mpl_yaml_docs"
GEMINI_EMBED_MODEL = "text-embedding-004"


# -----------------------------
# Helpers
# -----------------------------
def stable_id(*parts: str) -> str:
    s = "||".join(parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def try_parse_yaml(text: str) -> Optional[Any]:
    try:
        return yaml.safe_load(text)
    except Exception:
        return None


def dump_yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).strip()


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma metadata values must be: str, int, float, bool, None.
    YAML parsing produces lists/dicts -> convert those to YAML strings.
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, dict)):
            clean[k] = dump_yaml(v)
        else:
            clean[k] = str(v)
    return clean


def make_doc(text: str, meta: Dict[str, Any]) -> Document:
    return Document(page_content=text, metadata=sanitize_metadata(meta))


# -----------------------------
# Semantic formatting helpers
# -----------------------------
def format_list(items: List[str], prefix: str = "") -> str:
    if not items:
        return "None"
    if len(items) == 1:
        return f"{prefix}{items[0]}"
    if len(items) == 2:
        return f"{prefix}{items[0]} and {items[1]}"
    return f"{prefix}{', '.join(items[:-1])}, and {items[-1]}"


def make_searchable_aliases(aliases: List[str]) -> str:
    if not aliases:
        return ""
    return f"Also known as: {', '.join(aliases)}. User might ask for: {' or '.join(aliases)}."


# -----------------------------
# Chunking rules
# -----------------------------
def split_cards(parsed: Any, rel_path: str, raw_text: str) -> Tuple[List[Document], List[str]]:
    docs: List[Document] = []
    ids: List[str] = []

    base_meta = {
        "source": rel_path,
        "file_name": os.path.basename(rel_path),
    }

    if parsed is None or not isinstance(parsed, dict):
        text = f"""RAW_DOC
source: {rel_path}

content: |
{raw_text}
"""
        _id = stable_id(rel_path, "raw")
        docs.append(make_doc(text, {**base_meta, "doc_type": "Raw", "chunk_type": "raw"}))
        ids.append(_id)
        return docs, ids

    dtype = str(parsed.get("doc_type", "Unknown"))
    version = parsed.get("version")

    # -------------------------
    # ColumnCard: 1 per column
    # -------------------------
    if dtype == "ColumnCard":
        table_fqn = parsed.get("table_fqn")
        columns = parsed.get("columns", []) or []

        for col in columns:
            col_name = (col or {}).get("name")
            col_type = (col or {}).get("type", "")
            col_desc = (col or {}).get("description", "")
            col_tags = (col or {}).get("tags", [])

            tags_str = format_list(col_tags, "")

            text = f"""Column: {col_name}
Table: {table_fqn}
Data Type: {col_type}

Description: {col_desc}

Tags: {tags_str}

Technical Details:
{dump_yaml(col)}
"""
            meta = {
                **base_meta,
                "doc_type": "ColumnCard",
                "chunk_type": "column",
                "table_fqn": table_fqn,
                "column_name": col_name,
                "column_type": col_type,
                "tags": col_tags,
            }
            _id = stable_id(rel_path, "column", str(table_fqn), str(col_name))
            docs.append(make_doc(text, meta))
            ids.append(_id)
        return docs, ids

    # -------------------------
    # DimensionCard: 1 per dimension
    # -------------------------
    if dtype == "DimensionCard":
        dim = parsed.get("dimension_name")
        base_col = parsed.get("base_column")
        desc = parsed.get("description", "")
        mapping = parsed.get("mapping", {}) or {}
        mapping_values = list(mapping.values()) if mapping else []

        text = f"""Dimension: {dim}
Base Column: {base_col}
Scope: {parsed.get("scope", "global")}

Description: {desc}

Mapped values include: {format_list(mapping_values)}

SQL Rendering:
{dump_yaml(parsed.get("sql_rendering", {}))}
"""
        meta = {
            **base_meta,
            "doc_type": "DimensionCard",
            "chunk_type": "dimension",
            "dimension_name": dim,
            "base_column": base_col,
            "scope": parsed.get("scope"),
        }
        _id = stable_id(rel_path, "dimension", str(dim))
        return [make_doc(text, meta)], [_id]

    # -------------------------
    # ScopeCard: 1 per scope
    # -------------------------
    if dtype == "ScopeCard":
        scope = parsed.get("scope_name")
        desc = parsed.get("description", "")
        required = parsed.get("required_filters", []) or []
        forbidden = parsed.get("forbidden_filters", []) or []

        text = f"""Scope: {scope}

Description: {desc}

Required Filters:
{dump_yaml(required)}

Forbidden Filters:
{dump_yaml(forbidden)}
"""
        meta = {
            **base_meta,
            "doc_type": "ScopeCard",
            "chunk_type": "scope",
            "scope_name": scope,
            "required_filters": required,
            "forbidden_filters": forbidden,
        }
        _id = stable_id(rel_path, "scope", str(scope))
        return [make_doc(text, meta)], [_id]

    # -------------------------
    # TableCard: 1 per table
    # -------------------------
    if dtype == "TableCard":
        table_fqn = parsed.get("table_fqn")

        text = f"""Table: {table_fqn}

Timezone: {parsed.get("timezone", "UTC")}

Technical Details:
{dump_yaml(parsed)}
"""
        meta = {
            **base_meta,
            "doc_type": "TableCard",
            "chunk_type": "table",
            "table_fqn": table_fqn,
            "supported_scopes": parsed.get("supported_scopes", []) or [],
        }
        _id = stable_id(rel_path, "table", str(table_fqn))
        return [make_doc(text, meta)], [_id]

    # -------------------------
    # MetricCatalog: 1 per metric (+ header)
    # -------------------------
    if dtype == "MetricCatalog":
        default_scopes = parsed.get("default_scopes", []) or []
        header_text = f"""Metric Catalog Header
Default Timezone: {parsed.get("default_timezone")}
Default Scopes: {format_list(default_scopes)}

{dump_yaml({
    "doc_type": "MetricCatalog",
    "version": version,
    "default_timezone": parsed.get("default_timezone"),
    "default_scopes": default_scopes,
})}
"""
        header_meta = {**base_meta, "doc_type": "MetricCatalog", "chunk_type": "metric_catalog_header"}
        header_id = stable_id(rel_path, "metric_catalog_header")
        docs.append(make_doc(header_text, header_meta))
        ids.append(header_id)

        for m in parsed.get("metrics", []) or []:
            mk = (m or {}).get("metric_key")
            label = (m or {}).get("label", "")
            aliases = (m or {}).get("aliases", []) or []
            expression = (m or {}).get("expression", "")
            unit = (m or {}).get("unit", "")
            agg = (m or {}).get("agg", "")
            scope_refs = (m or {}).get("scope_refs", []) or []

            aliases_text = make_searchable_aliases(aliases)

            text = f"""Metric: {mk}
Label: {label}

{aliases_text}

SQL Expression: {expression}
Aggregation: {agg}
Unit: {unit}
Scope Refs: {format_list(scope_refs)}

Technical Details:
{dump_yaml(m)}
"""
            meta = {
                **base_meta,
                "doc_type": "MetricCatalog",
                "chunk_type": "metric",
                "metric_key": mk,
                "label": label,
                "aliases": aliases,
                "scope_refs": scope_refs,
                "expression": expression,
                "unit": unit,
                "agg": agg,
            }
            _id = stable_id(rel_path, "metric", str(mk))
            docs.append(make_doc(text, meta))
            ids.append(_id)

        return docs, ids

    # -------------------------
    # mpl_business_glossary: 1 per entry
    # -------------------------
    if dtype == "mpl_business_glossary":
        domain = parsed.get("domain")
        sections = parsed.get("sections", []) or []

        for sec in sections:
            sec_key = (sec or {}).get("section_key")
            sec_title = (sec or {}).get("title", "")
            entries = (sec or {}).get("entries", []) or []

            for e in entries:
                term = (e or {}).get("term")
                aliases = (e or {}).get("aliases", []) or []
                definition = (e or {}).get("definition", "")
                schema_mapping = (e or {}).get("schema_mapping", {}) or {}

                aliases_text = make_searchable_aliases(aliases)

                text = f"""Business Term: {term}
Category: {sec_title}
Domain: {domain}

{aliases_text}

Definition: {definition}

Schema Mapping:
{dump_yaml(schema_mapping)}

Full Entry:
{dump_yaml(e)}
"""
                meta = {
                    **base_meta,
                    "doc_type": "mpl_business_glossary",
                    "chunk_type": "glossary_entry",
                    "domain": domain,
                    "section_key": sec_key,
                    "term": term,
                    "aliases": aliases,
                }
                _id = stable_id(rel_path, "glossary", str(sec_key), str(term))
                docs.append(make_doc(text, meta))
                ids.append(_id)
        return docs, ids

    # -------------------------
    # PolicyDoc: ✅ add MASTER + 1 per section
    # -------------------------
    if dtype == "PolicyDoc":
        domain = parsed.get("domain")

        # ✅ MASTER rulebook chunk (binding)
        master_text = f"""Policy Rulebook (MASTER)
Domain: {domain}

This is the authoritative, binding rulebook for SQL generation.
If any conflict exists, follow PolicyDoc.

Full PolicyDoc:
{dump_yaml(parsed)}
"""
        master_meta = {
            **base_meta,
            "doc_type": "PolicyDoc",
            "chunk_type": "policy_master",
            "domain": domain,
            "policy_section": "MASTER",
        }
        master_id = stable_id(rel_path, "policy", "MASTER")
        docs.append(make_doc(master_text, master_meta))
        ids.append(master_id)

        # Existing: 1 per top-level policy section
        for k, v in parsed.items():
            if k in ("doc_type", "version", "domain", "description"):
                continue

            text = f"""Policy: {k}
Domain: {domain}

Policy Details:
{dump_yaml(v)}
"""
            meta = {
                **base_meta,
                "doc_type": "PolicyDoc",
                "chunk_type": "policy_section",
                "domain": domain,
                "policy_section": k,
            }
            _id = stable_id(rel_path, "policy", str(k))
            docs.append(make_doc(text, meta))
            ids.append(_id)

        return docs, ids

    # -------------------------
    # Fallback: 1 chunk per file
    # -------------------------
    text = f"""doc_type: {dtype}
version: {version}
source: {rel_path}

content:
{dump_yaml(parsed)}
"""
    meta = {**base_meta, "doc_type": dtype, "chunk_type": "fallback"}
    _id = stable_id(rel_path, "fallback")
    return [make_doc(text, meta)], [_id]


def load_yaml_files() -> List[Path]:
    if not RAG_DOCS_DIR.exists():
        raise RuntimeError(f"Docs directory does not exist: {RAG_DOCS_DIR}")
    yaml_files = list(RAG_DOCS_DIR.rglob("*.yaml")) + list(RAG_DOCS_DIR.rglob("*.yml"))
    return sorted(set(yaml_files))


def build_chroma() -> None:
    files = load_yaml_files()
    if not files:
        raise RuntimeError(f"No YAML files found under {RAG_DOCS_DIR}")

    print("📁 RAG_DOCS_DIR:", RAG_DOCS_DIR)
    print("📄 YAML files:", len(files))

    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)

    # Fresh rebuild to avoid duplicates
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    docs_all: List[Document] = []
    ids_all: List[str] = []

    for fp in files:
        rel_path = str(fp.relative_to(RAG_DIR)).replace("\\", "/")
        raw = read_text(fp)
        parsed = try_parse_yaml(raw)
        docs, ids = split_cards(parsed, rel_path, raw)
        docs_all.extend(docs)
        ids_all.extend(ids)

    print("🧩 Total chunks:", len(docs_all))

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    db.add_documents(documents=docs_all, ids=ids_all)
    db.persist()

    print("✅ Chroma built successfully")
    print("   persist_directory:", CHROMA_DIR)
    print("   collection_name: ", COLLECTION_NAME)
    print("   embedding_model: ", GEMINI_EMBED_MODEL)


if __name__ == "__main__":
    build_chroma()
