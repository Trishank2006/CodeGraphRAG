from parser.models import CodeEntity

from chunking.ast_chunker import chunk_entities
from chunking.code_chunker import CodeChunk


def test_chunk_entities():
    content = """class PaymentService:
    def process_payment(self, amount):
        total = amount
        return total

def calculate_fee(amount):
    return amount * 0.02
"""

    entities = [
        CodeEntity(
            id="src/payment.py:class:PaymentService",
            type="class",
            name="PaymentService",
            file_path="src/payment.py",
            start_line=0,
            end_line=3,
        ),
        CodeEntity(
            id="src/payment.py:function:calculate_fee",
            type="function",
            name="calculate_fee",
            file_path="src/payment.py",
            start_line=5,
            end_line=6,
        ),
    ]

    chunks = chunk_entities(
        repository="test-repository",
        file_path="src/payment.py",
        language="python",
        content=content,
        entities=entities,
    )

    assert len(chunks) == 2

    assert isinstance(chunks[0], CodeChunk)
    assert chunks[0].symbol == "PaymentService"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 4
    assert "class PaymentService:" in chunks[0].content

    assert chunks[1].symbol == "calculate_fee"
    assert chunks[1].start_line == 6
    assert chunks[1].end_line == 7
    assert "return amount * 0.02" in chunks[1].content