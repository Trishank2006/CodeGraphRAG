from pathlib import Path

from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.config import EvaluationConfig
from evaluation.dataset import (
    EvaluationExample,
    EvaluationDataset,
)


class FakeService:
    pass


def make_dataset():
    return EvaluationDataset(
        [
            EvaluationExample(
                query="Find service",
                query_type="implementation",
                expected_files=(
                    "application/service.py",
                ),
                expected_symbols=(),
                ground_truth_answer=(
                    "Service implementation."
                ),
            ),
            EvaluationExample(
                query="Find API",
                query_type="implementation",
                expected_files=(
                    "api/main.py",
                ),
                expected_symbols=(),
                ground_truth_answer=(
                    "API implementation."
                ),
            ),
        ]
    )


def make_runner():
    return BenchmarkRunner(
        dataset=make_dataset(),
        config=EvaluationConfig.create(),
    )


def test_runner_initialization():
    runner = BenchmarkRunner()

    assert runner.dataset is not None
    assert runner.config is not None


def test_runner_with_custom_dataset():
    runner = make_runner()

    assert len(runner.dataset) == 2


def test_build_report_structure(monkeypatch):
    runner = make_runner()

    fake_retrieval_report = {
        "metrics": [
            {
                "name": "Dense",
                "hit_rate": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "mrr": 1.0,
                "ndcg": 1.0,
            }
        ],
        "query_count": 2,
        "top_k": 5,
    }

    monkeypatch.setattr(
        runner,
        "run_retrieval_benchmark",
        lambda service: fake_retrieval_report,
    )

    report = runner.build_report(
        FakeService(),
        repository_version="test-version",
    )

    assert "benchmark" in report
    assert "configuration" in report
    assert "retrieval" in report
    assert "generation" in report
    assert "citation" in report
    assert "latency" in report
    assert "context_efficiency" in report
    assert "ablation" in report

    assert (
        report["benchmark"]["query_count"]
        == 2
    )

    assert (
        report["benchmark"][
            "repository_version"
        ]
        == "test-version"
    )


def test_generation_benchmark():
    runner = make_runner()

    answers = {
        "Find service":
            "Service implementation.",
        "Find API":
            "API implementation.",
    }

    contexts = {
        "Find service":
            "Service implementation.",
        "Find API":
            "API implementation.",
    }

    result = runner.run_generation_benchmark(
        answers=answers,
        contexts=contexts,
    )

    assert result["query_count"] == 2
    assert "average" in result
    assert "per_query" in result
    assert len(result["per_query"]) == 2


def test_generation_benchmark_without_answers():
    runner = make_runner()

    result = runner.run_generation_benchmark()

    assert result["query_count"] == 2
    assert len(result["per_query"]) == 2


def test_citation_benchmark():
    runner = make_runner()

    result = runner.run_citation_benchmark()

    assert result["query_count"] == 2
    assert "average" in result
    assert "per_query" in result
    assert len(result["per_query"]) == 2


def test_latency_benchmark_without_stages():
    runner = make_runner()

    result = runner.run_latency_benchmark()

    assert result["iterations"] == 3
    assert result["stages"] == {}


def test_context_efficiency_without_context():
    runner = make_runner()

    result = runner.run_context_efficiency()

    assert result is not None


def test_ablation_metadata():
    runner = make_runner()

    results = [
        {
            "name": "Dense",
            "hit_rate": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "mrr": 0.5,
            "ndcg": 0.5,
        }
    ]

    result = runner.build_ablation_metadata(
        results
    )

    assert "stages" in result
    assert "table" in result
    assert len(result["stages"]) == 1


def test_save_report(tmp_path):
    runner = make_runner()

    report = {
        "test": True,
        "metrics": [1, 2, 3],
    }

    output = (
        tmp_path / "benchmark.json"
    )

    saved = runner.save_report(
        report,
        output,
    )

    assert saved == Path(output)
    assert saved.exists()

    content = saved.read_text(
        encoding="utf-8"
    )

    assert '"test": true' in content
    assert '"metrics"' in content


def test_save_report_creates_parent_directory(
    tmp_path,
):
    runner = make_runner()

    output = (
        tmp_path
        / "nested"
        / "results"
        / "benchmark.json"
    )

    runner.save_report(
        {"status": "ok"},
        output,
    )

    assert output.exists()


def test_retrieval_config_mapping():
    runner = make_runner()

    retrieval_config = (
        runner._retrieval_config()
    )

    assert (
        retrieval_config.dense_top_k
        == runner.config.top_k
    )

    assert (
        retrieval_config.bm25_top_k
        == runner.config.top_k
    )

    assert (
        retrieval_config.graph_top_k
        == runner.config.top_k
    )

    assert (
        retrieval_config.rrf_top_k
        == runner.config.top_k
    )

    assert (
        retrieval_config.rerank_top_k
        == runner.config.top_k
    )

    assert (
        retrieval_config.rrf_k
        == runner.config.rrf_k
    )