from retrieval.models import RetrievalResult


def test_retrieval_result():
    result = RetrievalResult(
        id="chunk-1",
        source="vector",
        score=0.91,
        repository="test-repository",
        file_path="src/payment.py",
        language="python",
        symbol="process_payment",
        start_line=10,
        end_line=25,
        content="def process_payment():\n    pass",
    )

    assert result.id == "chunk-1"
    assert result.source == "vector"
    assert result.score == 0.91
    assert result.file_path == "src/payment.py"
    assert result.symbol == "process_payment"