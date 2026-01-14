# test_retriever.py

from pprint import pprint
from collections import defaultdict

from rag.rag_store import retrieve_context


def print_chunks(chunks):
    """
    Pretty-print retrieved chunks grouped by doc_type.
    """
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["doc_type"]].append(c)

    print("\n================ RETRIEVAL RESULT =================")
    print(f"Total chunks returned: {len(chunks)}\n")

    for dtype, items in grouped.items():
        print(f"--- {dtype} ({len(items)} chunks) ---")
        for i, item in enumerate(items, start=1):
            meta = item.get("metadata", {})
            print(f"\n[{i}] source: {item.get('source')}")
            print(f"    chunk_type: {item.get('chunk_type')}")

            # Important metadata per type
            if dtype == "MetricCatalog":
                print(f"    metric_key: {meta.get('metric_key')}")
                print(f"    scope_refs: {meta.get('scope_refs')}")

            if dtype == "ScopeCard":
                print(f"    scope_name: {meta.get('scope_name')}")
                print(f"    required_filters: {meta.get('required_filters')}")
                print(f"    forbidden_filters: {meta.get('forbidden_filters')}")

            if dtype == "PolicyDoc":
                print(f"    policy_section: {meta.get('policy_section')}")

            if dtype == "TableCard":
                print(f"    table_fqn: {meta.get('table_fqn')}")

            if dtype == "ColumnCard":
                print(f"    column_name: {meta.get('column_name')}")

            if dtype == "mpl_business_glossary":
                print(f"    term: {meta.get('term')}")

            # Print content preview (first N chars)
            content = item.get("content", "").strip().replace("\n", " ")
            preview = content[:1000] + ("..." if len(content) > 1000 else "")
            print(f"    content_preview: {preview}")

        print("\n")


def run_test(query: str, k: int = 10):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    chunks = retrieve_context(query, k=k)
    print_chunks(chunks)


if __name__ == "__main__":
    # 🔍 Add / remove test queries freely
    test_queries = [
        "show me month over month GMV for cash lobby last 60 days in US",
    ]

    for q in test_queries:
        run_test(q, k=10)

