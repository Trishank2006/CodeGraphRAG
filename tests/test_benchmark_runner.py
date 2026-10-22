from __future__ import annotations

from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.config import EvaluationConfig
from evaluation.dataset import EvaluationDataset, EvaluationExample


def make_runner() -> BenchmarkRunner:
    dataset = EvaluationDataset(
        examples=[
            EvaluationExample(
                query="Find service",
                query_type="implementation_lookup",
                expected_files=("application/service.py",),
                ground_truth_answer="Service implementation.",
            ),
            EvaluationExample(
                query="Find API",
                query_type="implementation_lookup",
                expected_files=("api/main.py",),
                ground_truth_answer="API implementation.",
            ),
        ]
    )

    return BenchmarkRunner(
        dataset=dataset,
        config=EvaluationConfig.create(),
    )


def test_runner_initializes():
    runner = make_runner()

    assert len(runner.dataset) == 2
    assert runner.config is not None


def test_retrieval_config_mapping():
    runner = make_runner()
    config = runner._retrieval_config()

    assert config.dense_top_k == 20
    assert config.bm25_top_k == 20
    assert config.graph_top_k == 20
    assert config.rrf_top_k == 30
    assert config.rerank_top_k == max(
        runner.config.top_k,
        runner.config.rerank_top_k,
    )
    assert config.rrf_k == runner.config.rrf_k


def test_generation_benchmark_without_answers():
    runner = make_runner()

    result = runner.run_generation_benchmark()

    assert result["query_count"] == 2
    assert result["evaluated_count"] == 0
    assert result["status"] == "not_evaluated"
    assert len(result["per_query"]) == 2
    assert "average" in result
    assert all(
        item["status"] == "not_evaluated"
        for item in result["per_query"]
    )


def test_generation_benchmark_with_complete_inputs():
    runner = make_runner()

    answers = {
        "Find service": "Service implementation.",
        "Find API": "API implementation.",
    }
    contexts = {
        "Find service": (
            "application/service.py contains the service implementation."
        ),
        "Find API": "api/main.py contains the API implementation.",
    }

    result = runner.run_generation_benchmark(
        answers=answers,
        contexts=contexts,
    )

    assert result["status"] == "evaluated"
    assert result["query_count"] == 2
    assert result["evaluated_count"] == 2
    assert len(result["per_query"]) == 2
    assert all(
        item["status"] == "evaluated"
        for item in result["per_query"]
    )
    assert set(result["average"]) == {
        "faithfulness",
        "answer_relevance",
        "context_relevance",
    }


def test_generation_benchmark_counts_partial_inputs():
    runner = make_runner()

    result = runner.run_generation_benchmark(
        answers={
            "Find service": "Service implementation.",
        },
        contexts={
            "Find service": "application/service.py implements the service.",
            "Find API": "api/main.py contains the API implementation.",
        },
    )

    assert result["status"] == "partially_evaluated"
    assert result["query_count"] == 2
    assert result["evaluated_count"] == 1
    assert result["per_query"][0]["status"] == "evaluated"
    assert result["per_query"][1]["status"] == "not_evaluated"


def test_generation_benchmark_rejects_whitespace_only_inputs_as_missing():
    runner = make_runner()

    result = runner.run_generation_benchmark(
        answers={
            "Find service": "   ",
            "Find API": "API implementation.",
        },
        contexts={
            "Find service": "application/service.py",
            "Find API": "   ",
        },
    )

    assert result["status"] == "not_evaluated"
    assert result["evaluated_count"] == 0


def test_citation_benchmark_without_citations():
    runner = make_runner()

    result = runner.run_citation_benchmark()

    assert result["query_count"] == 2
    assert len(result["per_query"]) == 2
    assert result["status"] == "not_evaluated"
    assert result["validity_status"] == "not_evaluated"
    assert result["coverage_status"] == "not_evaluated"
    assert result["evaluated_count"] == 0
    assert result["coverage_evaluated_count"] == 0
    assert result["average"]["citation_validity_rate"] is None
    assert result["average"]["citation_coverage"] is None


def test_citation_benchmark_marks_coverage_as_evaluated_when_supplied():
    runner = make_runner()

    result = runner.run_citation_benchmark(
        citations_by_query={
            "Find service": [],
            "Find API": [],
        }
    )

    assert result["status"] == "not_evaluated"
    assert result["validity_status"] == "not_evaluated"
    assert result["coverage_status"] == "evaluated"
    assert result["evaluated_count"] == 0
    assert result["coverage_evaluated_count"] == 2
    assert result["average"]["citation_coverage"] == 0.0


def test_citation_benchmark_marks_partial_coverage():
    runner = make_runner()

    result = runner.run_citation_benchmark(
        citations_by_query={
            "Find service": [],
        }
    )

    assert result["coverage_status"] == "partially_evaluated"
    assert result["coverage_evaluated_count"] == 1


def test_latency_benchmark_without_stages():
    runner = make_runner()

    result = runner.run_latency_benchmark()

    assert result["iterations"] == 3
    assert result["stages"] == {}


def test_latency_benchmark_with_samples():
    runner = make_runner()

    result = runner.run_latency_benchmark(
        stages={"retrieval": [10.0, 20.0, 30.0]}
    )

    assert result["stages"]["retrieval"]["iterations"] == 3
    assert result["stages"]["retrieval"]["average_ms"] == 20.0
    assert result["stages"]["retrieval"]["min_ms"] == 10.0
    assert result["stages"]["retrieval"]["max_ms"] == 30.0


def test_latency_benchmark_rejects_nonpositive_iterations():
    runner = make_runner()

    try:
        runner.run_latency_benchmark(iterations=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero iterations")


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

    result = runner.build_ablation_metadata(results)

    assert result["stages"][0]["name"] == "Dense"
    assert result["stages"][0]["recall"] == 0.5


def test_average_numeric_metrics_ignores_booleans():
    result = BenchmarkRunner._average_numeric_metrics(
        [
            {"score": 1.0, "valid": True},
            {"score": 3.0, "valid": False},
        ]
    )

    assert result["score"] == 2.0
    assert "valid" not in result


def test_save_report(tmp_path):
    report = {"benchmark": {"query_count": 2}}
    output = tmp_path / "results" / "benchmark.json"

    saved_path = BenchmarkRunner.save_report(report, output)

    assert saved_path.exists()
    assert '"query_count": 2' in saved_path.read_text(encoding="utf-8")


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
        service=object(),
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

    assert report["benchmark"]["query_count"] == 2
    assert (
        report["benchmark"]["repository_version"]
        == "test-version"
    )

    # Generation and citation inputs are not supplied by this
    # report-building call, so they must not claim evaluation.
    assert report["generation"]["status"] == "not_evaluated"
    assert report["citation"]["validity_status"] == "not_evaluated"


def test_runner_with_custom_dataset():
    runner = make_runner()
    assert len(runner.dataset) == 2


def test_save_report_creates_parent_directory(tmp_path):
    runner = make_runner()

    output = (
        tmp_path
        / "nested"
        / "results"
        / "benchmark.json"
    )

    runner.save_report({"status": "ok"}, output)

    assert output.exists()


def test_build_report_passes_repository_files_to_citation_evaluator(
    tmp_path, monkeypatch
):
    runner = make_runner()

    # Create a minimal repository with one source file.
    source_dir = tmp_path / "application"
    source_dir.mkdir()
    (source_dir / "service.py").write_text(
        "def index_repository():\n    pass\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runner,
        "run_retrieval_benchmark",
        lambda service: {
            "metrics": [],
            "query_count": len(runner.dataset),
            "top_k": 5,
        },
    )

    captured = {}

    def fake_citation_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "status": "not_evaluated",
            "validity_status": "not_evaluated",
            "coverage_status": "not_evaluated",
            "query_count": len(runner.dataset),
            "evaluated_count": 0,
            "coverage_evaluated_count": 0,
            "average": {
                "citation_validity_rate": 0.0,
                "citation_coverage": 0.0,
            },
            "per_query": [],
        }

    monkeypatch.setattr(
        runner,
        "run_citation_benchmark",
        fake_citation_benchmark,
    )

    runner.build_report(
        service=object(),
        repository_root=str(tmp_path),
    )

    assert "application/service.py" in captured["available_files"]


def test_retrieval_benchmark_includes_per_query_diagnostics(monkeypatch):
    from retrieval.models import RetrievalResult

    class FakeServiceRetriever:
        def __init__(self, service, config=None):
            pass

        def as_retrievers(self):
            def dense(example):
                return [
                    RetrievalResult(
                        id=f"chunk-{example.query}",
                        source="dense",
                        score=0.9,
                        repository="CodeGraphRAG",
                        file_path=example.expected_files[0],
                        language="python",
                        symbol=None,
                        start_line=1,
                        end_line=10,
                        content="Test result",
                    )
                ]

            return {"Dense": dense}

    monkeypatch.setattr(
        "evaluation.benchmark_runner.ServiceRetriever",
        FakeServiceRetriever,
    )

    runner = make_runner()
    report = runner.run_retrieval_benchmark(service=object())

    assert report["query_count"] == 2
    assert report["top_k"] == runner.config.top_k
    assert len(report["metrics"]) == 1
    assert report["metrics"][0]["name"] == "Dense"

    assert len(report["diagnostics"]) == 1
    dense_diagnostics = report["diagnostics"][0]
    assert dense_diagnostics["name"] == "Dense"
    assert len(dense_diagnostics["per_query"]) == 2

    first = dense_diagnostics["per_query"][0]
    assert first["query"] == "Find service"
    assert first["expected_targets"] == ["application/service.py"]
    assert first["hit"] is True
    assert first["first_relevant_rank"] == 1
