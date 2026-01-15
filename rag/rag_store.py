# rag/rag_store.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Set

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "rag/vector_db_chroma"
COLLECTION_NAME = "mpl_yaml_docs"
EMBED_MODEL = "text-embedding-004"


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def _get_chunk_identity(metadata: dict, content: str) -> tuple:
    if metadata.get("metric_key"):
        # include table_fqn if present to avoid collisions across tables
        if metadata.get("table_fqn"):
            return ("metric", metadata["metric_key"], metadata["table_fqn"])
        return ("metric", metadata["metric_key"])

    if metadata.get("column_name") and metadata.get("table_fqn"):
        return ("column", metadata["table_fqn"], metadata["column_name"])

    if metadata.get("scope_name"):
        return ("scope", metadata["scope_name"])

    if metadata.get("dimension_name"):
        return ("dimension", metadata["dimension_name"])

    if metadata.get("term"):
        return ("glossary", metadata["term"])

    if metadata.get("policy_section"):
        return ("policy", metadata["policy_section"])

    if metadata.get("table_fqn") and metadata.get("chunk_type") == "table":
        return ("table", metadata["table_fqn"])

    source = metadata.get("source", "")
    chunk_type = metadata.get("chunk_type", "")
    if source and chunk_type:
        return ("source_chunk", source, chunk_type)

    return ("content_hash", hash(content))


def fetch_policy_rulebook(db: Chroma) -> List[Dict[str, Any]]:
    """
    Always return PolicyDoc MASTER chunk.
    This is appended to every query and treated as binding rulebook.
    """
    docs = []
    try:
        docs = db.similarity_search(
            "PolicyDoc MASTER rulebook",
            k=5,
            filter={"doc_type": "PolicyDoc", "chunk_type": "policy_master"},
        )
    except Exception:
        docs = db.similarity_search("PolicyDoc MASTER rulebook", k=10)

    if not docs:
        try:
            docs = db.similarity_search("PolicyDoc", k=10, filter={"doc_type": "PolicyDoc"})
        except Exception:
            docs = db.similarity_search("PolicyDoc", k=10)

    out: List[Dict[str, Any]] = []
    for d in docs:
        if d.metadata.get("doc_type") != "PolicyDoc":
            continue
        if d.metadata.get("chunk_type") != "policy_master":
            continue
        out.append({
            "source": d.metadata.get("source"),
            "doc_type": d.metadata.get("doc_type"),
            "chunk_type": d.metadata.get("chunk_type"),
            "content": d.page_content,
            "metadata": {
                "policy_section": d.metadata.get("policy_section"),
                "domain": d.metadata.get("domain"),
            },
        })

    if not out:
        for d in docs:
            if d.metadata.get("doc_type") != "PolicyDoc":
                continue
            out.append({
                "source": d.metadata.get("source"),
                "doc_type": d.metadata.get("doc_type"),
                "chunk_type": d.metadata.get("chunk_type"),
                "content": d.page_content,
                "metadata": {
                    "policy_section": d.metadata.get("policy_section"),
                    "domain": d.metadata.get("domain"),
                },
            })

    out.sort(key=lambda x: 0 if x.get("chunk_type") == "policy_master" else 1)
    return out[:1]  # ✅ one binding rulebook chunk


def _ensure_mandatory_coverage(
    db: Chroma,
    merged: List[Dict[str, Any]],
    seen_keys: set,
    query: str,
    per_type_k: int = 3,
) -> List[Dict[str, Any]]:
    doc_types_present = {item["doc_type"] for item in merged}

    # ✅ ColumnCard added as mandatory
    mandatory_types = {
        "MetricCatalog",
        "mpl_business_glossary",
        "TableCard",
        "ColumnCard",
    }
    missing_types = mandatory_types - doc_types_present

    if not missing_types:
        return merged

    for dtype in missing_types:
        targeted_docs = []
        try:
            targeted_docs = db.similarity_search(query, k=per_type_k, filter={"doc_type": dtype})
        except Exception:
            targeted_docs = db.similarity_search(f"{dtype} {query}", k=per_type_k)

        if not targeted_docs:
            targeted_docs = db.similarity_search(dtype, k=per_type_k)

        for doc in targeted_docs:
            chunk_key = _get_chunk_identity(doc.metadata, doc.page_content)
            if chunk_key in seen_keys:
                continue
            seen_keys.add(chunk_key)

            merged.append({
                "source": doc.metadata.get("source"),
                "doc_type": doc.metadata.get("doc_type"),
                "chunk_type": doc.metadata.get("chunk_type"),
                "content": doc.page_content,
                "metadata": {
                    "metric_key": doc.metadata.get("metric_key"),
                    "scope_name": doc.metadata.get("scope_name"),
                    "table_fqn": doc.metadata.get("table_fqn"),
                    "column_name": doc.metadata.get("column_name"),
                    "dimension_name": doc.metadata.get("dimension_name"),
                    "term": doc.metadata.get("term"),
                    "policy_section": doc.metadata.get("policy_section"),
                    "aliases": doc.metadata.get("aliases"),
                    "scope_refs": doc.metadata.get("scope_refs"),
                    "required_filters": doc.metadata.get("required_filters"),
                    "forbidden_filters": doc.metadata.get("forbidden_filters"),
                    "supported_scopes": doc.metadata.get("supported_scopes"),
                    "column_type": doc.metadata.get("column_type"),
                    "tags": doc.metadata.get("tags"),
                }
            })

    return merged


def _extract_scope_names(scope_refs: Any) -> List[str]:
    """
    scope_refs can be:
      - list of strings: ["scopecards/scope_x.yaml", ...]
      - YAML string (because of metadata sanitization): "- scopecards/scope_x.yaml\n- ..."
      - empty / None
    Returns list of scope_name like "scope_x".
    """
    if not scope_refs:
        return []

    refs: List[str] = []

    if isinstance(scope_refs, list):
        refs = [r for r in scope_refs if isinstance(r, str)]
    elif isinstance(scope_refs, str):
        for line in scope_refs.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                line = line.lstrip("-").strip()
            refs.append(line)
    else:
        return []

    scope_names: List[str] = []
    for ref in refs:
        if not isinstance(ref, str):
            continue
        last = ref.split("/")[-1]
        if last.endswith(".yaml"):
            last = last[:-5]
        if last:
            scope_names.append(last)

    seen = set()
    out = []
    for s in scope_names:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _parse_supported_scopes(value: Any) -> List[str]:
    """
    supported_scopes can be stored as list or YAML string due to metadata sanitization.
    Return list of scope_name strings.
    """
    if not value:
        return []
    if isinstance(value, list):
        return [x for x in value if isinstance(x, str)]
    if isinstance(value, str):
        out = []
        for line in value.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                line = line.lstrip("-").strip()
            out.append(line)
        return out
    return []


def _target_tables_from_metrics(merged: List[Dict[str, Any]]) -> Set[str]:
    tables = set()
    for item in merged:
        if item.get("doc_type") != "MetricCatalog":
            continue
        tf = item.get("metadata", {}).get("table_fqn")
        if tf:
            tables.add(tf)
    return tables


def _ensure_table_coverage_for_selected_metrics(
    db: Chroma,
    merged: List[Dict[str, Any]],
    seen_keys: set,
) -> List[Dict[str, Any]]:
    target_tables = _target_tables_from_metrics(merged)
    if not target_tables:
        return merged

    present_tables = {
        item.get("metadata", {}).get("table_fqn")
        for item in merged
        if item.get("doc_type") == "TableCard"
    }
    missing_tables = target_tables - present_tables
    if not missing_tables:
        return merged

    for table_fqn in missing_tables:
        try:
            docs = db.similarity_search(
                table_fqn,
                k=2,
                filter={"doc_type": "TableCard", "table_fqn": table_fqn},
            )
        except Exception:
            docs = db.similarity_search(f"TableCard {table_fqn}", k=6)

        for doc in docs:
            key = _get_chunk_identity(doc.metadata, doc.page_content)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append({
                "source": doc.metadata.get("source"),
                "doc_type": "TableCard",
                "chunk_type": doc.metadata.get("chunk_type"),
                "content": doc.page_content,
                "metadata": {
                    "table_fqn": doc.metadata.get("table_fqn"),
                    "supported_scopes": doc.metadata.get("supported_scopes"),
                }
            })

    return merged


def _ensure_column_coverage_for_target_tables(
    db: Chroma,
    merged: List[Dict[str, Any]],
    seen_keys: set,
    query: str,
    per_table_k: int = 8,
) -> List[Dict[str, Any]]:
    target_tables = _target_tables_from_metrics(merged)
    if not target_tables:
        return merged

    # inject columns relevant to the query for each target table
    for table_fqn in target_tables:
        try:
            docs = db.similarity_search(
                query,
                k=per_table_k,
                filter={"doc_type": "ColumnCard", "table_fqn": table_fqn},
            )
        except Exception:
            docs = db.similarity_search(f"{table_fqn} {query}", k=per_table_k)

        for doc in docs:
            key = _get_chunk_identity(doc.metadata, doc.page_content)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append({
                "source": doc.metadata.get("source"),
                "doc_type": "ColumnCard",
                "chunk_type": doc.metadata.get("chunk_type"),
                "content": doc.page_content,
                "metadata": {
                    "table_fqn": doc.metadata.get("table_fqn"),
                    "column_name": doc.metadata.get("column_name"),
                    "column_type": doc.metadata.get("column_type"),
                    "tags": doc.metadata.get("tags"),
                }
            })

    return merged


def _ensure_metric_scope_coverage(
    db: Chroma,
    merged: List[Dict[str, Any]],
    seen_keys: set,
) -> List[Dict[str, Any]]:
    """
    For every MetricCatalog chunk with scope_refs,
    ensure the referenced ScopeCard chunks are present.
    """
    required_scopes = set()

    for item in merged:
        if item.get("doc_type") != "MetricCatalog":
            continue
        scope_refs = item.get("metadata", {}).get("scope_refs")
        for scope_name in _extract_scope_names(scope_refs):
            required_scopes.add(scope_name)

    if not required_scopes:
        return merged

    present_scopes = {
        item.get("metadata", {}).get("scope_name")
        for item in merged
        if item.get("doc_type") == "ScopeCard"
    }

    missing = required_scopes - present_scopes
    if not missing:
        return merged

    print(f"⚠️  Injecting ScopeCard(s) required by metrics: {missing}")

    for scope_name in missing:
        try:
            docs = db.similarity_search(
                scope_name,
                k=2,
                filter={"doc_type": "ScopeCard", "scope_name": scope_name},
            )
        except Exception:
            docs = db.similarity_search(scope_name, k=4)

        for doc in docs:
            key = _get_chunk_identity(doc.metadata, doc.page_content)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append({
                "source": doc.metadata.get("source"),
                "doc_type": "ScopeCard",
                "chunk_type": doc.metadata.get("chunk_type"),
                "content": doc.page_content,
                "metadata": {
                    "scope_name": doc.metadata.get("scope_name"),
                    "required_filters": doc.metadata.get("required_filters"),
                    "forbidden_filters": doc.metadata.get("forbidden_filters"),
                }
            })

    return merged


def _prune_unreferenced_scopes(merged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enforce your rule:
      - Only ScopeCard chunks that are referenced by retrieved MetricCatalog.scope_refs are allowed.
      - Any other ScopeCard chunk MUST be removed.
    """
    referenced = set()
    for item in merged:
        if item.get("doc_type") != "MetricCatalog":
            continue
        scope_refs = item.get("metadata", {}).get("scope_refs")
        for scope_name in _extract_scope_names(scope_refs):
            referenced.add(scope_name)

    if not referenced:
        return [x for x in merged if x.get("doc_type") != "ScopeCard"]

    pruned: List[Dict[str, Any]] = []
    for item in merged:
        if item.get("doc_type") != "ScopeCard":
            pruned.append(item)
            continue
        sname = item.get("metadata", {}).get("scope_name")
        if sname in referenced:
            pruned.append(item)

    return pruned


def _prune_scopes_not_supported_by_table(merged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Additional safety:
      - Keep ONLY scopes that are supported by the TableCard(s) for the metric's target table_fqn.
      - If multiple target tables exist, allow scopes supported by ANY of them (conservative).
    """
    # table_fqn -> supported_scopes
    table_supported = {}
    for item in merged:
        if item.get("doc_type") != "TableCard":
            continue
        tf = item.get("metadata", {}).get("table_fqn")
        ss = _parse_supported_scopes(item.get("metadata", {}).get("supported_scopes"))
        if tf:
            table_supported[tf] = set(ss)

    target_tables = _target_tables_from_metrics(merged)
    if not target_tables or not table_supported:
        return merged

    allowed_scopes = set()
    for tf in target_tables:
        allowed_scopes |= table_supported.get(tf, set())

    pruned = []
    for item in merged:
        if item.get("doc_type") != "ScopeCard":
            pruned.append(item)
            continue
        sname = item.get("metadata", {}).get("scope_name")
        if not sname:
            continue
        if sname in allowed_scopes:
            pruned.append(item)

    return pruned


def retrieve_context(query: str, k: int = 10) -> List[Dict[str, Any]]:
    db = get_vectorstore()

    subqueries = [
        query,
        f"calculate metric {query}",
        f"SQL generation rule {query}",
        f"business definition {query}",
        f"table column {query}",
    ]

    seen_keys = set()
    merged: List[Dict[str, Any]] = []

    docs_per_query = max(4, k // 2)

    for sq in subqueries:
        docs = db.similarity_search(sq, k=docs_per_query)

        for doc in docs:
            chunk_key = _get_chunk_identity(doc.metadata, doc.page_content)
            if chunk_key in seen_keys:
                continue
            seen_keys.add(chunk_key)

            merged.append({
                "source": doc.metadata.get("source"),
                "doc_type": doc.metadata.get("doc_type"),
                "chunk_type": doc.metadata.get("chunk_type"),
                "content": doc.page_content,
                "metadata": {
                    "metric_key": doc.metadata.get("metric_key"),
                    "scope_name": doc.metadata.get("scope_name"),
                    "table_fqn": doc.metadata.get("table_fqn"),
                    "column_name": doc.metadata.get("column_name"),
                    "dimension_name": doc.metadata.get("dimension_name"),
                    "term": doc.metadata.get("term"),
                    "policy_section": doc.metadata.get("policy_section"),
                    "aliases": doc.metadata.get("aliases"),
                    "scope_refs": doc.metadata.get("scope_refs"),
                    "required_filters": doc.metadata.get("required_filters"),
                    "forbidden_filters": doc.metadata.get("forbidden_filters"),
                    "supported_scopes": doc.metadata.get("supported_scopes"),
                    "column_type": doc.metadata.get("column_type"),
                    "tags": doc.metadata.get("tags"),
                }
            })

    merged = _ensure_mandatory_coverage(db, merged, seen_keys, query, per_type_k=3)

    # ✅ NEW: ensure TableCard for any retrieved metric.table_fqn
    merged = _ensure_table_coverage_for_selected_metrics(db, merged, seen_keys)

    # ✅ NEW: ensure ColumnCards for the target tables
    merged = _ensure_column_coverage_for_target_tables(db, merged, seen_keys, query, per_table_k=8)

    # ✅ existing: ensure scope cards for any metric scope_refs
    merged = _ensure_metric_scope_coverage(db, merged, seen_keys)

    # ✅ existing: enforce "no other scope cards allowed"
    merged = _prune_unreferenced_scopes(merged)

    # ✅ NEW: prune scopes not supported by the target table(s)
    merged = _prune_scopes_not_supported_by_table(merged)

    policy_rulebook = fetch_policy_rulebook(db)
    final = policy_rulebook + merged
    return final[: k + 10]
