from chunking.code_chunker import CodeChunk, create_chunk


def test_create_code_chunk():
    chunk = create_chunk(
        repository="test-repository",
        file_path="src/payment.py",
        language="python",
        symbol="PaymentService.process_payment",
        start_line=10,
        end_line=25,
        content="def process_payment():\n    pass",
    )

    assert isinstance(chunk, CodeChunk)
    assert chunk.repository == "test-repository"
    assert chunk.file_path == "src/payment.py"
    assert chunk.language == "python"
    assert chunk.symbol == "PaymentService.process_payment"
    assert chunk.start_line == 10
    assert chunk.end_line == 25
    assert chunk.content == "def process_payment():\n    pass"
    assert chunk.chunk_id == "test-repository:src/payment.py:10:25"