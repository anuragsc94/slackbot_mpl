# rag/chunker.py

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from langchain_core.documents import Document

# Keep your existing Chroma import (will show deprecation warning, but works).
# Optional upgrade shown at end.
from langchain_community.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings


# -----------------------------
# Config
# -----------------------------
RAG_DIR = Path(__file__).resolve().parent                  # .../slackbot_mpl/rag
RAG_DOCS_DIR = RAG_DIR / "docs"                            # .../slackbot_mpl/rag/docs
CHROMA_DIR = RAG_DIR / "vector_db_chroma"                  # .../slackbot_mpl/rag/vector_db_chroma
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
    # "validation skipped": we attempt parse; if it fails we fallback to raw
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
            clean[k] = dump_yaml(v)  # YAML string, not JSON
        else:
            clean[k] = str(v)
    return clean


def make_doc(text: str, meta: Dict[str, Any]) -> Document:
    return Document(page_content=text, metadata=sanitize_metadata(meta))


# -----------------------------
# Chunking rules (matches your YAML schema)
# -----------------------------
def split_cards(parsed: Any, rel_path: str, raw_text: str) -> Tuple[List[Document], List[str]]:
    """
    Convert one YAML file into 1..N Documents, with deterministic IDs.
    """
    docs: List[Document] = []
    ids: List[str] = []

    # Base metadata for every chunk from this file
    base_meta = {
        "source": rel_path,
        "file_name": os.path.basename(rel_path),
    }

    # If YAML parse failed -> raw chunk
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
            text = f"""doc_type: ColumnCard
version: {version}
source: {rel_path}
table_fqn: {table_fqn}
column_name: {col_name}

column:
{dump_yaml(col)}
"""
            meta = {
                **base_meta,
                "doc_type": "ColumnCard",
                "chunk_type": "column",
                "table_fqn": table_fqn,
                "column_name": col_name,
                "tags": (col or {}).get("tags", []),   # will be sanitized to YAML string
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
        text = f"""doc_type: DimensionCard
version: {version}
source: {rel_path}
dimension_name: {dim}
base_column: {base_col}
scope: {parsed.get("scope")}

description: {parsed.get("description")}

normalization:
{dump_yaml(parsed.get("normalization", {}))}

mapping_type: {parsed.get("mapping_type")}
mapping:
{dump_yaml(parsed.get("mapping", {}))}

fallback: {parsed.get("fallback")}

sql_rendering:
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
        text = f"""doc_type: ScopeCard
version: {version}
source: {rel_path}
scope_name: {scope}

description: {parsed.get("description")}

required_filters:
{dump_yaml(parsed.get("required_filters", []))}

forbidden_filters:
{dump_yaml(parsed.get("forbidden_filters", []))}
"""
        meta = {
            **base_meta,
            "doc_type": "ScopeCard",
            "chunk_type": "scope",
            "scope_name": scope,
            "required_filters": parsed.get("required_filters", []),   # sanitized
            "forbidden_filters": parsed.get("forbidden_filters", []), # sanitized
        }
        _id = stable_id(rel_path, "scope", str(scope))
        return [make_doc(text, meta)], [_id]

    # -------------------------
    # TableCard: 1 per table
    # -------------------------
    if dtype == "TableCard":
        table_fqn = parsed.get("table_fqn")
        text = f"""doc_type: TableCard
version: {version}
source: {rel_path}
table_fqn: {table_fqn}
timezone: {parsed.get("timezone")}

grain:
{dump_yaml(parsed.get("grain", {}))}

time:
{dump_yaml(parsed.get("time", {}))}

identifiers:
{dump_yaml(parsed.get("identifiers", []))}

dimensions:
{dump_yaml(parsed.get("dimensions", []))}

measures:
{dump_yaml(parsed.get("measures", {}))}

derived_dimensions:
{dump_yaml(parsed.get("derived_dimensions", []))}

supported_scopes:
{dump_yaml(parsed.get("supported_scopes", []))}

type_gotchas:
{dump_yaml(parsed.get("type_gotchas", []))}
"""
        meta = {
            **base_meta,
            "doc_type": "TableCard",
            "chunk_type": "table",
            "table_fqn": table_fqn,
            "supported_scopes": parsed.get("supported_scopes", []),  # sanitized
        }
        _id = stable_id(rel_path, "table", str(table_fqn))
        return [make_doc(text, meta)], [_id]

    # -------------------------
    # MetricCatalog: 1 per metric (+ header)
    # -------------------------
    if dtype == "MetricCatalog":
        header = {
            "doc_type": "MetricCatalog",
            "version": version,
            "default_timezone": parsed.get("default_timezone"),
            "default_scopes": parsed.get("default_scopes", []),
        }
        header_text = f"""doc_type: MetricCatalog
chunk_type: header
source: {rel_path}

{dump_yaml(header)}
"""
        header_meta = {**base_meta, "doc_type": "MetricCatalog", "chunk_type": "metric_catalog_header"}
        header_id = stable_id(rel_path, "metric_catalog_header")
        docs.append(make_doc(header_text, header_meta))
        ids.append(header_id)

        for m in parsed.get("metrics", []) or []:
            mk = (m or {}).get("metric_key")
            text = f"""doc_type: MetricCatalog
chunk_type: metric
source: {rel_path}
metric_key: {mk}

metric:
{dump_yaml(m)}
"""
            meta = {
                **base_meta,
                "doc_type": "MetricCatalog",
                "chunk_type": "metric",
                "metric_key": mk,
                "aliases": (m or {}).get("aliases", []),       # sanitized
                "scope_refs": (m or {}).get("scope_refs", []), # sanitized
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
            entries = (sec or {}).get("entries", []) or []
            for e in entries:
                term = (e or {}).get("term")
                text = f"""doc_type: mpl_business_glossary
version: {version}
source: {rel_path}
domain: {domain}
section_key: {sec_key}
term: {term}

entry:
{dump_yaml(e)}
"""
                meta = {
                    **base_meta,
                    "doc_type": "mpl_business_glossary",
                    "chunk_type": "glossary_entry",
                    "domain": domain,
                    "section_key": sec_key,
                    "term": term,
                    "aliases": (e or {}).get("aliases", []),  # sanitized
                }
                _id = stable_id(rel_path, "glossary", str(sec_key), str(term))
                docs.append(make_doc(text, meta))
                ids.append(_id)
        return docs, ids

    # -------------------------
    # PolicyDoc: 1 per top-level policy section
    # -------------------------
    if dtype == "PolicyDoc":
        domain = parsed.get("domain")
        for k, v in parsed.items():
            if k in ("doc_type", "version", "domain", "description"):
                continue
            text = f"""doc_type: PolicyDoc
version: {version}
source: {rel_path}
domain: {domain}
policy_section: {k}

section:
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

    # Debug prints (helpful)
    print("📁 RAG_DOCS_DIR:", RAG_DOCS_DIR)
    print("📄 YAML files:", len(files))

    embeddings = GoogleGenerativeAIEmbeddings(
    model=GEMINI_EMBED_MODEL,
    api_key=os.environ["GOOGLE_API_KEY"],)


    # Fresh rebuild to avoid duplicates
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    docs_all: List[Document] = []
    ids_all: List[str] = []

    for fp in files:
        rel_path = str(fp.relative_to(RAG_DIR)).replace("\\", "/")  # docs/... path
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
