from embeddings.embedder import CodeEmbedder
from retrieval.search import search_code
from vector_store.indexer import index_chunks
from vector_store.qdrant_store import QdrantStore


def main():
    chunks = [
        {
            "chunk_id": "demo-auth",
            "repository": "demo-repository",
            "file_path": "src/auth.py",
            "language": "python",
            "symbol": "authenticate_user",
            "start_line": 1,
            "end_line": 6,
            "content": """
def authenticate_user(username, password):
    user = find_user(username)
    if user and verify_password(password, user.password_hash):
        return create_session(user)
    return None
""".strip(),
        },
        {
            "chunk_id": "demo-payment",
            "repository": "demo-repository",
            "file_path": "src/payment.py",
            "language": "python",
            "symbol": "process_payment",
            "start_line": 1,
            "end_line": 6,
            "content": """
def process_payment(amount, card):
    validate_card(card)
    charge_card(card, amount)
    return True
""".strip(),
        },
        {
            "chunk_id": "demo-user",
            "repository": "demo-repository",
            "file_path": "src/user.py",
            "language": "python",
            "symbol": "create_user",
            "start_line": 1,
            "end_line": 6,
            "content": """
def create_user(name, email):
    user = User(name, email)
    save_user(user)
    return user
""".strip(),
        },
    ]

    embedder = CodeEmbedder()

    store = QdrantStore(
        path="data/qdrant",
        collection_name="demo_code_chunks",
        vector_size=384,
    )

    index_chunks(
        chunks=chunks,
        embedder=embedder,
        store=store,
    )

    query = "How is user authentication implemented?"

    results = search_code(
        query=query,
        top_k=3,
        embedder=embedder,
        store=store,
    )

    print(f"\nQuery: {query}\n")
    print("Results:\n")

    for index, result in enumerate(results, start=1):
        payload = result["payload"]

        print(f"{index}. {payload['symbol']}")
        print(f"   File: {payload['file_path']}")
        print(f"   Language: {payload['language']}")
        print(f"   Lines: {payload['start_line']}-{payload['end_line']}")
        print(f"   Score: {result['score']:.4f}")
        print()


if __name__ == "__main__":
    main()