from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from evaluation.ablation import (
    create_ablation_result,
    results_to_table,
)

from evaluation.citation_metrics import (
    calculate_citation_metrics,
)

from evaluation.config import EvaluationConfig
from retrieval.config import RetrievalConfig

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
        Run retrieval benchmarks and retain per-query diagnostics.
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

        metrics = []
        diagnostics = []

        for name, retriever in retrievers.items():
            detailed = benchmark.evaluate_detailed(
                name,
                retriever,
            )

            overall = detailed["by_query_type"]["overall"]

            metrics.append({
                "name": name,
                "hit_rate": overall["hit_rate"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "mrr": overall["mrr"],
                "ndcg": overall["ndcg"],
            })

            diagnostics.append({
                "name": name,
                "by_query_type": detailed["by_query_type"],
                "per_query": detailed["per_query"],
            })

        return {
            "metrics": metrics,
            "query_count": len(self.dataset),
            "top_k": self.config.top_k,
            "diagnostics": diagnostics,
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
        Evaluate generated answers using the supplied answers and contexts.

        `answers` maps query text -> generated answer.
        `contexts` maps query text -> retrieved context.

        Examples without both a non-empty answer and context are marked
        as not evaluated and excluded from aggregate generation metrics.

        This benchmark is deterministic and does not call an external LLM.
        """
        per_query = []
        evaluated_metrics = []

        for example in self.dataset:
            answer = (
                answers.get(example.query, "")
                if answers is not None
                else ""
            )

            context = (
                contexts.get(example.query, "")
                if contexts is not None
                else ""
            )

            # Treat missing or whitespace-only inputs as unavailable.
            answer = answer if isinstance(answer, str) else ""
            context = context if isinstance(context, str) else ""

            has_answer = bool(answer.strip())
            has_context = bool(context.strip())
            evaluated = has_answer and has_context

            context_items = [context] if has_context else []

            metrics = generation_metrics(
                query=example.query,
                answer=answer,
                context=context_items,
            )

            if evaluated:
                evaluated_metrics.append(metrics)
            else:
                # None means unavailable; 0.0 remains a measured score.
                metrics = {key: None for key in metrics}

            per_query.append(
                {
                    "query": example.query,
                    "status": (
                        "evaluated"
                        if evaluated
                        else "not_evaluated"
                    ),
                    "has_answer": has_answer,
                    "has_context": has_context,
                    "metrics": metrics,
                }
            )

        evaluated_count = len(evaluated_metrics)
        query_count = len(self.dataset)

        if evaluated_count == 0:
            status = "not_evaluated"
        elif evaluated_count == query_count:
            status = "evaluated"
        else:
            status = "partially_evaluated"

        averages = average_generation_metrics(evaluated_metrics)
        if evaluated_count == 0:
            averages = {key: None for key in averages}

        return {
            "status": status,
            "query_count": query_count,
            "evaluated_count": evaluated_count,
            "average": averages,
            "per_query": per_query,
        }

    # ------------------------------------------------------------------
    # CITATIONS
    # ------------------------------------------------------------------

    
    def run_citation_benchmark(
        self,
        citations_by_query: Optional[Dict[str, Any]] = None,
        answers: Optional[Dict[str, str]] = None,
        contexts: Optional[Dict[str, str]] = None,
        available_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from evaluation.citation_metrics import (
            Citation as MetricCitation,
            calculate_citation_metrics,
            evaluate_citation,
        )

        per_query = []
        coverage_values = []
        coverage_evaluated_count = 0
        validity_evaluated_count = 0

        available_file_set = set(available_files or [])

        for example in self.dataset:
            query = example.query

            citations_supplied = (
                citations_by_query is not None
                and query in citations_by_query
            )

            raw_citations = (
                citations_by_query.get(query, [])
                if citations_by_query is not None
                else []
            )

            # Convert incoming dictionaries or citation objects into
            # the Citation type expected by citation_metrics.py.
            citations = []

            for raw in raw_citations or []:
                if isinstance(raw, dict):
                    file_path = raw.get("file_path")
                    start_line = raw.get("start_line")
                    end_line = raw.get("end_line")
                else:
                    file_path = getattr(raw, "file_path", None)
                    start_line = getattr(raw, "start_line", None)
                    end_line = getattr(raw, "end_line", None)

                if (
                    not file_path
                    or not isinstance(start_line, int)
                    or not isinstance(end_line, int)
                ):
                    continue

                citations.append(
                    MetricCitation(
                        file_path=str(file_path),
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

            # Prefer generated answers and contexts when available.
            # Fall back to the ground-truth answer for coverage-only tests.
            answer = (
                answers.get(query, "")
                if answers is not None
                else example.ground_truth_answer
            )
            if not answer:
                answer = example.ground_truth_answer or ""

            context = (
                contexts.get(query, "")
                if contexts is not None
                else ""
            )

            # Coverage is considered evaluated when the caller explicitly
            # supplied citations for this query, including an empty list.
            coverage_evaluated = citations_supplied

            evaluations = []
            validity_evaluated = False

            # Validity requires source-file information, answer/context data,
            # and at least one usable citation.
            if (
                citations
                and available_file_set
                and context.strip()
                and answer.strip()
            ):
                evaluations = [
                    evaluate_citation(
                        citation=citation,
                        available_files=available_file_set,
                        context=[context],
                        answer=answer,
                    )
                    for citation in citations
                ]
                validity_evaluated = True
                validity_evaluated_count += 1

            citation_metrics = calculate_citation_metrics(
                evaluations=evaluations,
                citations=citations,
                answer=answer,
            )

            # Preserve zero only when a metric was actually evaluated.
            if not validity_evaluated:
                citation_metrics["citation_validity_rate"] = None
            if not coverage_evaluated:
                citation_metrics["citation_coverage"] = None

            if coverage_evaluated:
                coverage_values.append(
                    citation_metrics["citation_coverage"]
                )
                coverage_evaluated_count += 1

            per_query.append(
                {
                    "query": query,
                    "status": (
                        "coverage_evaluated"
                        if coverage_evaluated
                        else "not_evaluated"
                    ),
                    "validity_status": (
                        "evaluated"
                        if validity_evaluated
                        else "not_evaluated"
                    ),
                    "metrics": citation_metrics,
                }
            )

        query_count = len(per_query)

        coverage_status = (
            "evaluated"
            if coverage_evaluated_count == query_count and query_count > 0
            else (
                "partially_evaluated"
                if coverage_evaluated_count > 0
                else "not_evaluated"
            )
        )

        validity_status = (
            "evaluated"
            if validity_evaluated_count == query_count and query_count > 0
            else (
                "partially_evaluated"
                if validity_evaluated_count > 0
                else "not_evaluated"
            )
        )

        average_coverage = (
            sum(coverage_values) / len(coverage_values)
            if coverage_values
            else None
        )

        return {
            "status": validity_status,
            "validity_status": validity_status,
            "coverage_status": coverage_status,
            "query_count": query_count,
            "evaluated_count": validity_evaluated_count,
            "coverage_evaluated_count": len(coverage_values),
            "average": {
                "citation_validity_rate": (
                    sum(
                        item["metrics"]["citation_validity_rate"]
                        for item in per_query
                        if item["validity_status"] == "evaluated"
                    )
                    / validity_evaluated_count
                    if validity_evaluated_count
                    else None
                ),
                "citation_coverage": average_coverage,
            },
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
        Benchmark callable pipeline stages or summarize pre-recorded samples.

        Callable stages are measured by ``benchmark_pipeline_stages``.
        A list/tuple of numeric samples is treated as already-recorded
        latency measurements, which is useful for deterministic tests.
        """

        if iterations <= 0:
            raise ValueError("iterations must be greater than 0")

        if not stages:
            return {
                "iterations": iterations,
                "stages": {},
            }

        callable_stages: Dict[str, Any] = {}
        sample_stages: Dict[str, list[float]] = {}

        for name, stage in stages.items():
            if callable(stage):
                callable_stages[name] = stage
                continue

            if isinstance(stage, (list, tuple)):
                if not stage:
                    raise ValueError(
                        f"Latency samples for stage {name!r} cannot be empty"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    for value in stage
                ):
                    raise TypeError(
                        f"Latency samples for stage {name!r} must be numeric"
                    )
                if any(float(value) < 0 for value in stage):
                    raise ValueError(
                        f"Latency samples for stage {name!r} must be non-negative"
                    )
                sample_stages[name] = [float(value) for value in stage]
                continue

            raise TypeError(
                f"Stage {name!r} must be a callable or a sequence of numeric samples"
            )

        measured: Dict[str, Any] = {}

        if callable_stages:
            measured.update(
                benchmark_pipeline_stages(
                    callable_stages,
                    iterations=iterations,
                )
            )

        for name, samples in sample_stages.items():
            measured[name] = {
                "iterations": len(samples),
                "average_ms": sum(samples) / len(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }

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
        """Compare character budgets across query-specific retrieved chunks."""
        if contexts is None:
            contexts = []

        comparisons = compare_context_budgets(
            contexts,
            (1000, 5000, 12000),
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
    repository_root: Optional[str] = None,
    context_chunks_by_query=None,
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
        repository_files: List[str] = []

        if repository_root is not None:
            root = Path(repository_root).resolve()
            ignored_dirs = {
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                ".pytest_cache",
                "node_modules",
            }

            if not root.is_dir():
                raise ValueError(
                    f"Repository root is not a directory: {root}"
                )

            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue

                relative_path = file_path.relative_to(root)

                if any(part in ignored_dirs for part in relative_path.parts):
                    continue

                repository_files.append(relative_path.as_posix())

        citation_report = self.run_citation_benchmark(
            citations_by_query=citations_by_query,
            answers=answers,
            contexts=contexts,
            available_files=repository_files,
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
            for context in contexts.values():
                if isinstance(context, str) and context.strip():
                    # Generation stores chunks separated by two newlines.
                    context_values.append(context.split("\n\n"))
                elif isinstance(context, list):
                    context_values.append(context)

        context_chunks = []

        if context_chunks_by_query:
            context_chunks = list(context_chunks_by_query.values())
        elif contexts:
            for context in contexts.values():
                if isinstance(context, str) and context.strip():
                    context_chunks.append(context.split("\n\n"))
                elif isinstance(context, list):
                    context_chunks.append(context)

        context_report = self.run_context_efficiency(contexts=context_chunks)


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

    @staticmethod
    def save_report(
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

    def _retrieval_config(self) -> RetrievalConfig:
        """Build consistent candidate pools for every retrieval variant."""
        final_top_k = self.config.top_k
        rerank_top_k = max(final_top_k, self.config.rerank_top_k)

        return RetrievalConfig(
            dense_top_k=max(20, rerank_top_k),
            bm25_top_k=max(20, rerank_top_k),
            graph_top_k=max(20, rerank_top_k),
            rrf_top_k=max(30, rerank_top_k),
            rerank_top_k=rerank_top_k,
            rrf_k=self.config.rrf_k,
        )

    @staticmethod
    def _average_numeric_metrics(
        metric_rows: list[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Average numeric metrics while excluding booleans and non-numeric values."""
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for row in metric_rows:
            for name, value in row.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                totals[name] = totals.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1

        return {
            name: totals[name] / counts[name]
            for name in totals
            if counts[name] > 0
        }



def main() -> None:
    """Command-line entry point for the real benchmark."""
    parser = argparse.ArgumentParser(
        description="Run CodeGraphRAG Phase 7 evaluation benchmark."
    )

    parser.add_argument(
        "--output",
        default="evaluation/results/benchmark.json",
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
        help="Number of retrieved results evaluated per query.",
    )

    parser.add_argument(
        "--evaluate-generation",
        action="store_true",
        help=(
            "Generate answers with the configured LLM and "
            "evaluate them against retrieved context."
        ),
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero.")

    from application.service import CodeGraphRAGService
    from retrieval.reranker import CrossEncoderReranker

    base_config = EvaluationConfig.create()

    config = EvaluationConfig(
        dataset_version=base_config.dataset_version,
        top_k=args.top_k,
        rrf_k=base_config.rrf_k,
        rerank_top_k=base_config.rerank_top_k,
        embedding_model=base_config.embedding_model,
        reranker_model=base_config.reranker_model,
        llm_model=base_config.llm_model,
        max_context_items=base_config.max_context_items,
        max_context_characters=base_config.max_context_characters,
        repository_version=base_config.repository_version,
        benchmark_timestamp=base_config.benchmark_timestamp,
        git_working_tree_dirty=base_config.git_working_tree_dirty,
        tracked_diff_sha256=base_config.tracked_diff_sha256,
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

    try:
        print("=" * 70)
        print("CodeGraphRAG Phase 7 Evaluation")
        print("=" * 70)

        print(f"Repository : {Path(args.repository).resolve()}")
        print(f"Dataset    : {len(runner.dataset)} queries")
        print(f"Top-K      : {config.top_k}")
        print(
            "Generation : "
            + ("enabled" if args.evaluate_generation else "disabled")
        )
        print()

        print("Indexing repository...")

        summary = service.index_repository(
            args.repository,
            repository_name=args.repository_name,
        )

        print()
        print("Indexing complete.")
        print(f"Files      : {summary.files}")
        print(f"Chunks     : {summary.chunks}")
        print(f"Graph nodes: {summary.graph_nodes}")
        print(f"Graph edges: {summary.graph_edges}")
        print()

        print("Running retrieval benchmark...")

        
        answers = None
        contexts = None
        citations_by_query = None
        latency_stages = None
        context_chunks_by_query = None

        if args.evaluate_generation:
            answers = {}
            contexts = {}
            citations_by_query = {}
            latencies_by_query = {}
            context_chunks_by_query = {}

        if args.evaluate_generation:
            answers = {}
            contexts = {}
            citations_by_query = {}
            latencies_by_query = {}
            context_chunks_by_query = {}

            checkpoint_path = Path(args.output).with_suffix(
                ".checkpoint.json"
            )
            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Resume previously completed queries if a checkpoint exists.
            if checkpoint_path.exists():
                with checkpoint_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    checkpoint = json.load(file)

                answers = checkpoint.get("answers", {})
                contexts = checkpoint.get("contexts", {})
                citations_by_query = checkpoint.get(
                    "citations_by_query", {}
                )
                latencies_by_query = checkpoint.get(
                    "latencies_by_query", {}
                )
                context_chunks_by_query = checkpoint.get(
                    "context_chunks_by_query", {}
                )
                if not context_chunks_by_query:
                    context_chunks_by_query = {
                        query: context.split("\n\n")
                        for query, context in contexts.items()
                        if isinstance(context, str) and context.strip()
                    }

                print(
                    f"Resuming from checkpoint: "
                    f"{checkpoint_path}"
                )
                print(
                    f"Saved answers: {len(answers)}/"
                    f"{len(runner.dataset)}"
                )

            def save_checkpoint() -> None:
                checkpoint = {
                    "answers": answers,
                    "contexts": contexts,
                    "citations_by_query": citations_by_query,
                    "latencies_by_query": latencies_by_query,
                    "context_chunks_by_query": context_chunks_by_query,
                }

                temporary_path = checkpoint_path.with_suffix(
                    checkpoint_path.suffix + ".tmp"
                )

                with temporary_path.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        checkpoint,
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )

                temporary_path.replace(checkpoint_path)

            print("Running answer-generation benchmark...")

            for index, example in enumerate(
                runner.dataset,
                start=1,
            ):
                query = example.query

                # Skip queries that already have a saved answer,
                # context, and citation entry.
                already_saved = (
                    query in answers
                    and query in contexts
                    and query in citations_by_query
                )

                if already_saved:
                    print(
                        f"[{index}/{len(runner.dataset)}] "
                        "Using saved answer..."
                    )
                    continue

                print(
                    f"[{index}/{len(runner.dataset)}] "
                    "Generating answer..."
                )

                start = time.perf_counter()
                generated = service.answer(query)

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                answers[query] = generated.answer

                context_items = []
                for item in generated.context:
                    context_items.append(
                        f"File: {item.file_path}\n"
                        f"Symbol: {item.symbol}\n"
                        f"Lines: {item.start_line}-{item.end_line}\n"
                        f"Source: {item.source}\n"
                        f"Content:\n{item.content}"
                    )

                contexts[query] = "\n\n".join(context_items)
                context_chunks_by_query[query] = context_items
                

                citations_by_query[query] = [
                    {
                        "file_path": citation.file_path,
                        "symbol": citation.symbol,
                        "start_line": citation.start_line,
                        "end_line": citation.end_line,
                    }
                    for citation in generated.citations
                ]

                latencies_by_query[query] = elapsed_ms

                # Persist each successful answer immediately.
                save_checkpoint()

            generation_latencies = [
                latencies_by_query[example.query]
                for example in runner.dataset
                if example.query in latencies_by_query
            ]

            latency_stages = {
                "generation": generation_latencies,
            }

            print(f"Checkpoint saved: {checkpoint_path}")


        report = runner.build_report(
            service,
            repository_version=config.repository_version,
            answers=answers,
            contexts=contexts,
            citations_by_query=citations_by_query,
            latency_stages=latency_stages,
            repository_root=args.repository,
            context_chunks_by_query=context_chunks_by_query,
        )

        output = runner.save_report(
            report,
            args.output,
        )
        diagnostics_path = (
            Path(args.output).with_name(
                f"{Path(args.output).stem}_retrieval_diagnostics.json"
            )
        )

        BenchmarkRunner.save_report(
            {
                "query_count": report["retrieval"]["query_count"],
                "top_k": report["retrieval"]["top_k"],
                "diagnostics": report["retrieval"]["diagnostics"],
            },
            diagnostics_path,
        )

        print(
            f"Retrieval diagnostics saved to: {diagnostics_path}"
        )

        print()
        print("Benchmark complete.")
        print(f"Results saved to: {output}")
        print()

        for metric in report["retrieval"]["metrics"]:
            print(
                f"{metric['name']:<35} "
                f"Hit={metric['hit_rate']:.4f} "
                f"Precision={metric['precision']:.4f} "
                f"Recall={metric['recall']:.4f} "
                f"MRR={metric['mrr']:.4f} "
                f"nDCG={metric['ndcg']:.4f}"
            )

        print()
        print(
            "Generation status: "
            f"{report['generation']['status']}"
        )
        print(
            "Citation validity status: "
            f"{report['citation']['validity_status']}"
        )

    finally:
        service.close()


if __name__ == "__main__":
    main()
