from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.ablation import (
    create_ablation_result,
    results_to_table,
)

from evaluation.citation_metrics import (
    calculate_citation_metrics,
)

from evaluation.config import EvaluationConfig

from evaluation.context_efficiency import (
    compare_context_budgets,
)

from evaluation.dataset import (
    EvaluationDataset,
    create_default_dataset,
)

from evaluation.generation_metrics import (
    average_generation_metrics,
    generation_metrics,
)

from evaluation.latency import (
    benchmark_pipeline_stages,
)

from evaluation.retrieval_benchmark import (
    RetrievalBenchmark,
    results_to_dict,
)

from evaluation.service_retriever import ServiceRetriever


class BenchmarkRunner:
    """
    Phase 7 evaluation and benchmarking runner.

    The runner combines:

    - retrieval evaluation
    - generation evaluation
    - citation evaluation
    - latency measurement
    - context efficiency
    - ablation metadata
    """

    def __init__(
        self,
        dataset: Optional[EvaluationDataset] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        self.dataset = dataset or create_default_dataset()
        self.config = config or EvaluationConfig.create()

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def run_retrieval_benchmark(
        self,
        service: Any,
    ) -> Dict[str, Any]:
        """
        Run all configured retrieval stages.
        """

        service_retriever = ServiceRetriever(
            service,
            config=self._retrieval_config(),
        )

        retrievers = service_retriever.as_retrievers()

        benchmark = RetrievalBenchmark(
            self.dataset,
            top_k=self.config.top_k,
        )

        results = benchmark.compare(retrievers)

        return {
            "metrics": results_to_dict(results),
            "query_count": len(self.dataset),
            "top_k": self.config.top_k,
        }

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------

    def run_generation_benchmark(
        self,
        answers: Optional[Dict[str, str]] = None,
        contexts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate generated answers.

        `answers` maps query text -> generated answer.

        `contexts` maps query text -> retrieved context.

        When no answers are supplied, deterministic placeholder
        evaluation is used. This keeps unit tests offline and avoids
        accidentally calling an external LLM.
        """

        per_query = []

        for example in self.dataset:
            answer = ""

            if answers is not None:
                answer = answers.get(
                    example.query,
                    "",
                )

            context = ""

            if contexts is not None:
                context = contexts.get(
                    example.query,
                    "",
                )

            # generation_metrics expects:
            # generation_metrics(query, answer, context)
            #
            # The context parameter expects a sequence of strings,
            # so convert the single context string into a one-item list.
            context_items = (
                [context]
                if context
                else []
            )

            metrics = generation_metrics(
                query=example.query,
                answer=answer,
                context=context_items,
            )

            per_query.append(
                {
                    "query": example.query,
                    "metrics": metrics,
                }
            )

        metric_values = [
            item["metrics"]
            for item in per_query
        ]

        averages = average_generation_metrics(
            metric_values
        )

        return {
            "query_count": len(self.dataset),
            "average": averages,
            "per_query": per_query,
        }

    # ------------------------------------------------------------------
    # CITATIONS
    # ------------------------------------------------------------------

    def run_citation_benchmark(
        self,
        citations_by_query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate citation quality.

        `citations_by_query` maps query text -> citation sequence.

        The lower-level citation evaluator is intentionally kept
        separate from this runner. This runner accepts already prepared
        citation evaluations when they are supplied.

        For the current offline benchmark, no external LLM or live
        repository is required.
        """

        per_query = []

        for example in self.dataset:
            citations = []

            if citations_by_query is not None:
                citations = citations_by_query.get(
                    example.query,
                    [],
                )

            # calculate_citation_metrics expects:
            #
            #   evaluations
            #   citations
            #   answer
            #
            # At this stage the runner does not yet construct live
            # CitationEvaluation objects from GeneratedAnswer output.
            # Therefore an empty evaluation sequence is used for the
            # deterministic offline benchmark.
            citation_metrics = calculate_citation_metrics(
                evaluations=[],
                citations=citations,
                answer=example.ground_truth_answer,
            )

            per_query.append(
                {
                    "query": example.query,
                    "metrics": citation_metrics,
                }
            )

        if not per_query:
            return {
                "query_count": 0,
                "average": {},
                "per_query": [],
            }

        metric_names = per_query[0]["metrics"].keys()

        average = {}

        for name in metric_names:
            values = [
                float(
                    item["metrics"].get(
                        name,
                        0.0,
                    )
                )
                for item in per_query
            ]

            average[name] = (
                sum(values) / len(values)
                if values
                else 0.0
            )

        return {
            "query_count": len(per_query),
            "average": average,
            "per_query": per_query,
        }

    # ------------------------------------------------------------------
    # LATENCY
    # ------------------------------------------------------------------

    def run_latency_benchmark(
        self,
        stages: Optional[Dict[str, Any]] = None,
        iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Benchmark callable pipeline stages.

        The actual production pipeline can provide its functions later.
        """

        if not stages:
            return {
                "iterations": iterations,
                "stages": {},
            }

        measured = benchmark_pipeline_stages(
            stages,
            iterations=iterations,
        )

        return {
            "iterations": iterations,
            "stages": measured,
        }

    # ------------------------------------------------------------------
    # CONTEXT EFFICIENCY
    # ------------------------------------------------------------------

    def run_context_efficiency(
        self,
        contexts: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Compare context budgets of 10, 20 and 30 chunks.
        """

        if contexts is None:
            contexts = []

        comparisons = compare_context_budgets(
            contexts,
            (10, 20, 30),
        )

        return comparisons

    # ------------------------------------------------------------------
    # ABLATION
    # ------------------------------------------------------------------

    def build_ablation_metadata(
        self,
        retrieval_results: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Build the ablation structure used in Phase 7.

        Actual metric values are inserted when the corresponding
        benchmark has been executed.
        """

        if retrieval_results is None:
            retrieval_results = []

        ablation_results = []

        for result in retrieval_results:
            ablation_results.append(
                create_ablation_result(
                    name=result["name"],
                    recall=result.get(
                        "recall",
                        0.0,
                    ),
                    mrr=result.get(
                        "mrr",
                        0.0,
                    ),
                    ndcg=result.get(
                        "ndcg",
                        0.0,
                    ),
                    faithfulness=result.get(
                        "faithfulness",
                        0.0,
                    ),
                    citation_accuracy=result.get(
                        "citation_accuracy",
                        0.0,
                    ),
                    latency_ms=result.get(
                        "latency_ms",
                        0.0,
                    ),
                )
            )

        return {
            "stages": [
                asdict(result)
                for result in ablation_results
            ],
            "table": results_to_table(
                ablation_results
            ),
        }

    # ------------------------------------------------------------------
    # COMPLETE REPORT
    # ------------------------------------------------------------------

    def build_report(
        self,
        service: Any,
        repository_version: Optional[str] = None,
        answers: Optional[Dict[str, str]] = None,
        contexts: Optional[Dict[str, str]] = None,
        citations_by_query: Optional[Dict[str, Any]] = None,
        latency_stages: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the complete Phase 7 report.
        """

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        # --------------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------------

        retrieval_report = (
            self.run_retrieval_benchmark(
                service
            )
        )

        # --------------------------------------------------------------
        # Generation
        # --------------------------------------------------------------

        generation_report = (
            self.run_generation_benchmark(
                answers=answers,
                contexts=contexts,
            )
        )

        # --------------------------------------------------------------
        # Citation
        # --------------------------------------------------------------

        citation_report = (
            self.run_citation_benchmark(
                citations_by_query=citations_by_query,
            )
        )

        # --------------------------------------------------------------
        # Latency
        # --------------------------------------------------------------

        latency_report = (
            self.run_latency_benchmark(
                stages=latency_stages,
            )
        )

        # --------------------------------------------------------------
        # Context efficiency
        # --------------------------------------------------------------

        context_values = []

        if contexts is not None:
            context_values = list(
                contexts.values()
            )

        context_report = (
            self.run_context_efficiency(
                contexts=context_values,
            )
        )

        # --------------------------------------------------------------
        # Ablation
        # --------------------------------------------------------------

        ablation_report = (
            self.build_ablation_metadata(
                retrieval_report["metrics"]
            )
        )

        # --------------------------------------------------------------
        # Final report
        # --------------------------------------------------------------

        return {
            "benchmark": {
                "dataset_version": (
                    self.config.dataset_version
                ),
                "query_count": len(self.dataset),
                "timestamp": timestamp,
                "repository_version": (
                    repository_version
                    or self.config.repository_version
                ),
            },
            "configuration": asdict(
                self.config
            ),
            "retrieval": retrieval_report,
            "generation": generation_report,
            "citation": citation_report,
            "latency": latency_report,
            "context_efficiency": context_report,
            "ablation": ablation_report,
        }

    # ------------------------------------------------------------------
# PERSISTENCE
# ------------------------------------------------------------------

    def save_report(
        self,
        report: Dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """
        Save benchmark report as formatted JSON.
        """

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        def make_json_serializable(value: Any) -> Any:
            if is_dataclass(value):
                return {
                    key: make_json_serializable(val)
                    for key, val in asdict(value).items()
                }

            if isinstance(value, dict):
                return {
                    str(key): make_json_serializable(val)
                    for key, val in value.items()
                }

            if isinstance(value, (list, tuple)):
                return [
                    make_json_serializable(item)
                    for item in value
                ]

            if isinstance(value, Path):
                return str(value)

            return value

        serializable_report = make_json_serializable(report)

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                serializable_report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return path

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    def _retrieval_config(self):
        """
        Convert evaluation configuration into the retrieval
        configuration expected by the production pipeline.
        """

        from retrieval.config import RetrievalConfig

        return RetrievalConfig(
            dense_top_k=max(
                self.config.top_k,
                1,
            ),
            bm25_top_k=max(
                self.config.top_k,
                1,
            ),
            graph_top_k=max(
                self.config.top_k,
                1,
            ),
            rrf_top_k=max(
                self.config.top_k,
                1,
            ),
            rerank_top_k=max(
                self.config.top_k,
                1,
            ),
            rrf_k=self.config.rrf_k,
        )


def main() -> None:
    """
    Command-line entry point for the real benchmark.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run CodeGraphRAG Phase 7 "
            "evaluation benchmark."
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "evaluation/results/benchmark.json"
        ),
        help="Output JSON path.",
    )

    parser.add_argument(
        "--repository",
        default=".",
        help="Repository path to benchmark.",
    )

    parser.add_argument(
        "--repository-name",
        default=None,
        help="Optional repository name.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of retrieved results "
            "evaluated per query."
        ),
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit(
            "--top-k must be greater than zero."
        )

    from application.service import (
        CodeGraphRAGService,
    )
    from retrieval.reranker import (
        CrossEncoderReranker,
    )

    config = EvaluationConfig.create()

    config = EvaluationConfig(
        dataset_version=(
            config.dataset_version
        ),
        top_k=args.top_k,
        rrf_k=config.rrf_k,
        rerank_top_k=config.rerank_top_k,
        embedding_model=(
            config.embedding_model
        ),
        reranker_model=(
            config.reranker_model
        ),
        llm_model=config.llm_model,
        max_context_items=(
            config.max_context_items
        ),
        max_context_characters=(
            config.max_context_characters
        ),
        repository_version=(
            config.repository_version
        ),
        benchmark_timestamp=(
            config.benchmark_timestamp
        ),
    )

    runner = BenchmarkRunner(
        dataset=create_default_dataset(),
        config=config,
    )

    reranker = CrossEncoderReranker(
        model_name=config.reranker_model,
    )

    service = CodeGraphRAGService(
        reranker=reranker,
    )

    print("=" * 70)
    print("CodeGraphRAG Phase 7 Evaluation")
    print("=" * 70)

    print(
        f"Repository : "
        f"{Path(args.repository).resolve()}"
    )

    print(
        f"Dataset    : "
        f"{len(runner.dataset)} queries"
    )

    print(
        f"Top-K      : "
        f"{config.top_k}"
    )

    print()

    print("Indexing repository...")

    summary = service.index_repository(
        args.repository,
        repository_name=args.repository_name,
    )

    print()
    print("Indexing complete.")

    print(
        f"Files      : "
        f"{summary.files}"
    )

    print(
        f"Chunks     : "
        f"{summary.chunks}"
    )

    print(
        f"Graph nodes: "
        f"{summary.graph_nodes}"
    )

    print(
        f"Graph edges: "
        f"{summary.graph_edges}"
    )

    print()

    print(
        "Running retrieval benchmark..."
    )

    report = runner.build_report(
        service,
        repository_version=(
            config.repository_version
        ),
    )

    output = runner.save_report(
        report,
        args.output,
    )

    print()
    print("Benchmark complete.")

    print(
        f"Results saved to: {output}"
    )

    print()

    for metric in report[
        "retrieval"
    ]["metrics"]:
        print(
            f"{metric['name']:<35} "
            f"Hit={metric['hit_rate']:.4f} "
            f"Precision={metric['precision']:.4f} "
            f"Recall={metric['recall']:.4f} "
            f"MRR={metric['mrr']:.4f} "
            f"nDCG={metric['ndcg']:.4f}"
        )


if __name__ == "__main__":
    main()