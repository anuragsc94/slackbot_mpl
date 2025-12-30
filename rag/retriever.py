from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CHROMA_DIR = "vector_db_chroma"
COLLECTION_NAME = "mpl_yaml_docs"
EMBED_MODEL = "text-embedding-004"

def main():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    queries = [
        "What column should I use for date filtering?",
        "What is scope_cash_lobby_default?",
        "What does free lobby (ZSH) mean and how is it determined?",
        "Is ZSH inferred from entry_fee?",
        "Give me GM definition and whether transaction_type filter should be applied",
        "How do I map app_type to a normalized platform?",
        "How do I map game_id to game_name?",
        "What is win_rate formula?",
        "Do you support cohort analysis / LTV cohorts?",
    ]

    for q in queries:
        print("\n" + "=" * 90)
        print("Q:", q)
        docs = db.similarity_search(q, k=5)
        for i, d in enumerate(docs, 1):
            print(f"\n--- Hit {i} ---")
            print("source:", d.metadata.get("source"))
            print("doc_type:", d.metadata.get("doc_type"))
            print("chunk_type:", d.metadata.get("chunk_type"))
            print(d.page_content[:350])

if __name__ == "__main__":
    main()
