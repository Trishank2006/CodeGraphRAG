from retrieval.bm25 import BM25Retriever
from retrieval.models import RetrievalResult


def create_chunks():
    return [
        {
            "chunk_id": "chunk-payment",
            "repository": "test-repository",
            "file_path": "src/payment/service.py",
            "language": "python",
            "symbol": "process_payment",
            "start_line": 10,
            "end_line": 20,
            "content": (
                "def process_payment(amount): "
                "charge_card(amount)"
            ),
        },
        {
            "chunk_id": "chunk-auth",
            "repository": "test-repository",
            "file_path": "src/auth/service.py",
            "language": "python",
            "symbol": "authenticate_user",
            "start_line": 5,
            "end_line": 15,
            "content": (
                "def authenticate_user(username, password): "
                "verify_password(password)"
            ),
        },
        {
            "chunk_id": "chunk-user",
            "repository": "test-repository",
            "file_path": "src/user/service.py",
            "language": "python",
            "symbol": "create_user",
            "start_line": 1,
            "end_line": 8,
            "content": (
                "def create_user(name, email): "
                "save_user(name, email)"
            ),
        },
    ]


def test_bm25_returns_matching_identifier():
    retriever = BM25Retriever()
    retriever.index(create_chunks())

    results = retriever.search_bm25(
        query="authenticate_user",
        top_k=3,
    )

    assert len(results) > 0
    assert results[0].id == "chunk-auth"
    assert results[0].symbol == "authenticate_user"
    assert results[0].source == "bm25"


def test_bm25_matches_exact_file_path():
    retriever = BM25Retriever()
    retriever.index(create_chunks())

    results = retriever.search_bm25(
        query="src/payment/service.py",
        top_k=3,
    )

    assert len(results) > 0
    assert results[0].file_path == "src/payment/service.py"


def test_bm25_returns_retrieval_results():
    retriever = BM25Retriever()
    retriever.index(create_chunks())

    results = retriever.search_bm25(
        query="payment charge card",
        top_k=3,
    )

    assert len(results) > 0
    assert isinstance(results[0], RetrievalResult)


def test_bm25_respects_top_k():
    retriever = BM25Retriever()
    retriever.index(create_chunks())

    results = retriever.search_bm25(
        query="python",
        top_k=2,
    )

    assert len(results) <= 2