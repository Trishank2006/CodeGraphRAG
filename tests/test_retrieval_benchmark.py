from evaluation.dataset import EvaluationExample
from evaluation.retrieval_benchmark import (
    RetrievalBenchmark,
    results_to_dict,
)


def make_dataset():
    return [
        EvaluationExample(
            query="Where is the application service implemented?",
            query_type="implementation",
            expected_files=("application/service.py",),
            expected_symbols=(),
            ground_truth_answer="The application service is implemented in application/service.py.",
        ),
        EvaluationExample(
            query="Where is the API health endpoint defined?",
            query_type="implementation",
            expected_files=("api/main.py",),
            expected_symbols=(),
            ground_truth_answer="The API health endpoint is defined in api/main.py.",
        ),
        EvaluationExample(
            query="Where is the hybrid retrieval pipeline implemented?",
            query_type="implementation",
            expected_files=("retrieval/pipeline.py",),
            expected_symbols=(),
            ground_truth_answer="The hybrid retrieval pipeline is implemented in retrieval/pipeline.py.",
        ),
    ]


def test_benchmark_single_retriever():
    dataset = make_dataset()

    benchmark = RetrievalBenchmark(dataset, top_k=3)

    def retriever(example):
        if example.expected_files:
            return [
                "unrelated.py",
                example.expected_files[0],
                "another.py",
            ]
        return []

    result = benchmark.evaluate("Dense", retriever)

    assert result.name == "Dense"
    assert result.hit_rate == 1.0
    assert result.recall == 1.0
    assert result.precision == 1 / 3
    assert result.mrr == 0.5
    assert result.ndcg > 0.0


def test_benchmark_perfect_retriever():
    dataset = make_dataset()

    benchmark = RetrievalBenchmark(dataset, top_k=5)

    def retriever(example):
        return list(example.expected_files)

    result = benchmark.evaluate("Perfect", retriever)

    assert result.hit_rate == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.mrr == 1.0
    assert result.ndcg == 1.0


def test_benchmark_missing_results():
    dataset = make_dataset()

    benchmark = RetrievalBenchmark(dataset, top_k=5)

    result = benchmark.evaluate(
        "Empty",
        lambda example: [],
    )

    assert result.hit_rate == 0.0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.mrr == 0.0
    assert result.ndcg == 0.0


def test_compare_multiple_retrievers():
    dataset = make_dataset()

    benchmark = RetrievalBenchmark(dataset, top_k=3)

    retrievers = {
        "Dense": lambda example: [
            "unrelated.py",
            *example.expected_files,
        ],
        "BM25": lambda example: list(example.expected_files),
        "Hybrid": lambda example: [
            *example.expected_files,
            "unrelated.py",
        ],
    }

    results = benchmark.compare(retrievers)

    assert len(results) == 3

    assert [result.name for result in results] == [
        "Dense",
        "BM25",
        "Hybrid",
    ]

    assert results[1].mrr == 1.0
    assert results[1].recall == 1.0


def test_results_to_dict():
    dataset = make_dataset()

    benchmark = RetrievalBenchmark(dataset, top_k=3)

    result = benchmark.evaluate(
        "Dense",
        lambda example: list(example.expected_files),
    )

    converted = results_to_dict([result])

    assert len(converted) == 1
    assert converted[0]["name"] == "Dense"
    assert converted[0]["hit_rate"] == 1.0
    assert converted[0]["precision"] == 1.0
    assert converted[0]["recall"] == 1.0
    assert converted[0]["mrr"] == 1.0
    assert converted[0]["ndcg"] == 1.0


def test_empty_dataset():
    benchmark = RetrievalBenchmark([], top_k=5)

    result = benchmark.evaluate(
        "Empty Dataset",
        lambda example: [],
    )

    assert result.name == "Empty Dataset"
    assert result.hit_rate == 0.0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.mrr == 0.0
    assert result.ndcg == 0.0


def test_invalid_top_k():
    try:
        RetrievalBenchmark([], top_k=0)
        assert False
    except ValueError:
        pass


def test_invalid_benchmark_name():
    benchmark = RetrievalBenchmark([], top_k=5)

    try:
        benchmark.evaluate(
            "",
            lambda example: [],
        )
        assert False
    except ValueError:
        pass


def test_benchmark_with_retrieval_result_object():
    from retrieval.models import RetrievalResult

    dataset = [
        EvaluationExample(
            query="Find service",
            query_type="implementation",
            expected_files=("application/service.py",),
            expected_symbols=(),
            ground_truth_answer="Service implementation.",
        )
    ]

    benchmark = RetrievalBenchmark(dataset, top_k=3)

    def retriever(example):
        return [
            RetrievalResult(
                id="chunk-1",
                source="dense",
                score=0.9,
                repository="CodeGraphRAG",
                file_path="application/service.py",
                language="python",
                symbol="CodeGraphRAGService",
                start_line=1,
                end_line=20,
                content="class CodeGraphRAGService:",
            )
        ]

    result = benchmark.evaluate("Dense", retriever)

    assert result.hit_rate == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.mrr == 1.0
    assert result.ndcg == 1.0


def test_benchmark_with_dictionary_result():
    dataset = [
        EvaluationExample(
            query="Find pipeline",
            query_type="implementation",
            expected_files=("retrieval/pipeline.py",),
            expected_symbols=(),
            ground_truth_answer="Pipeline implementation.",
        )
    ]

    benchmark = RetrievalBenchmark(dataset, top_k=3)

    def retriever(example):
        return [
            {
                "file_path": "retrieval/pipeline.py",
                "score": 0.95,
            }
        ]

    result = benchmark.evaluate("Dense", retriever)

    assert result.hit_rate == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.mrr == 1.0
    assert result.ndcg == 1.0


def test_invalid_retrieval_result_type():
    dataset = [
        EvaluationExample(
            query="Find service",
            query_type="implementation",
            expected_files=("application/service.py",),
            expected_symbols=(),
            ground_truth_answer="Service implementation.",
        )
    ]

    benchmark = RetrievalBenchmark(dataset, top_k=3)

    def retriever(example):
        return [12345]

    try:
        benchmark.evaluate("Invalid", retriever)
        assert False
    except TypeError:
        pass