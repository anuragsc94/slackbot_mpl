from __future__ import annotations
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  # pip install -U langchain-chroma

CHROMA_DIR = "rag/vector_db_chroma"   # run from repo root
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

def retrieve_context(query: str, k: int = 10) -> list[dict]:
    db = get_vectorstore()

    # Targeted sub-queries (helps pull MetricCatalog/Policy/Glossary consistently)
    # Defining mandatory YAML docs to be included in context retrieval
    subqueries = [
        query,
        f"MetricCatalog {query}",
        f"PolicyDoc {query}",
        f"mpl_business_glossary {query}",
    ]

    seen = set()
    merged = []

    for sq in subqueries:
        docs = db.similarity_search(sq, k=max(3, k // 2))
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("chunk_type"), d.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append({
                "source": d.metadata.get("source"),
                "doc_type": d.metadata.get("doc_type"),
                "chunk_type": d.metadata.get("chunk_type"),
                "content": d.page_content
            })

    return merged[:k]


