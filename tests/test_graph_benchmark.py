from unittest.mock import MagicMock
from evaluation.graph_benchmark import GraphBenchmark
from evaluation.graph_dataset import GraphEvaluationExample
from retrieval.models import RetrievalResult


def test_graph_benchmark_runner():
    mock_retriever = MagicMock()
    mock_grounding = MagicMock()

    # Mock search results returning expected symbol
    mock_retriever.search.return_value = [
        RetrievalResult(
            id="c1",
            repository="test-repo",
            file_path="auth.py",
            language="python",
            content="def validate_token(): pass",
            start_line=1,
            end_line=5,
            symbol="validate_token",
            score=0.9,
            source="graph",
        )
    ]
    mock_grounding.get_evidence.return_value = []
    mock_grounding.get_path.return_value = None

    dataset = [
        GraphEvaluationExample(
            query="Who calls validate_token?",
            query_type="callers",
            root_symbol="validate_token",
            expected_entities=["validate_token"],
        )
    ]

    bench = GraphBenchmark(mock_retriever, mock_grounding)
    results = bench.run_benchmark(dataset)

    assert results["total_queries"] == 1
    assert results["hit_rate_at_10"] == 1.0
    assert results["recall_at_10"] == 1.0
    assert results["mrr"] == 1.0