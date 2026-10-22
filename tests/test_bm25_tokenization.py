import pytest

from retrieval.bm25 import _tokenize


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("BM25Retriever", {"bm25retriever", "bm25", "retriever"}),
        ("LLMClient", {"llmclient", "llm", "client"}),
        ("Neo4jStore", {"neo4jstore", "neo4j", "store"}),
        ("HTTPResponseParser", {"httpresponseparser", "http", "response", "parser"}),
        ("SHA3Hasher", {"sha3hasher", "sha3", "hasher"}),
        ("process_payment", {"process_payment", "process", "payment"}),
        ("src/retrieval/bm25.py", {"src/retrieval/bm25.py", "src", "retrieval", "bm25", "py"}),
    ],
)
def test_tokenizer_splits_identifier_components(identifier, expected):
    assert expected.issubset(set(_tokenize(identifier)))


def test_tokenizer_does_not_duplicate_plain_words():
    assert _tokenize("context") == ["context"]
