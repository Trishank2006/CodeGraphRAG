from typing import Dict, Any, List
from graph.retrieval import GraphRetriever
from graph.grounding import GraphGroundingService
from evaluation.graph_dataset import create_graph_evaluation_dataset, GraphEvaluationExample
from evaluation.graph_metrics import (
    calculate_graph_hit_rate_at_k,
    calculate_graph_recall_at_k,
    calculate_graph_mrr,
    calculate_relationship_accuracy,
)
from evaluation.path_metrics import evaluate_path_reconstruction


class GraphBenchmark:
    """Runs structural evaluation queries and produces summary benchmark metrics."""

    def __init__(self, retriever: GraphRetriever, grounding: GraphGroundingService):
        self.retriever = retriever
        self.grounding = grounding

    def run_benchmark(
        self, dataset: List[GraphEvaluationExample] | None = None
    ) -> Dict[str, Any]:
        dataset = dataset or create_graph_evaluation_dataset()
        hit_rates: List[float] = []
        recalls: List[float] = []
        mrrs: List[float] = []
        rel_accuracies: List[float] = []
        path_exact_matches: List[float] = []

        for ex in dataset:
            if ex.query_type == "path" and ex.target_symbol:
                path_obj = self.grounding.get_path(ex.root_symbol, ex.target_symbol)
                retrieved_nodes = (
                    [step.source.name for step in path_obj.steps] + [path_obj.steps[-1].target.name]
                    if path_obj and getattr(path_obj, "steps", None)
                    else []
                )
                res = evaluate_path_reconstruction(retrieved_nodes, ex.expected_path)
                path_exact_matches.append(1.0 if res["exact_match"] else 0.0)
            else:
                results = self.retriever.search(ex.root_symbol, top_k=10)
                retrieved_symbols = [r.symbol for r in results]

                hit_rates.append(
                    calculate_graph_hit_rate_at_k(retrieved_symbols, ex.expected_entities, k=10)
                )
                recalls.append(
                    calculate_graph_recall_at_k(retrieved_symbols, ex.expected_entities, k=10)
                )
                mrrs.append(calculate_graph_mrr(retrieved_symbols, ex.expected_entities))

                if ex.expected_edges:
                    evidence = self.grounding.get_evidence(ex.root_symbol)
                    retrieved_edges = [
                        (
                            ev.source_entity,
                            getattr(ev, "relationship", getattr(ev, "relation_type", "")),
                            ev.target_entity,
                        )
                        for ev in evidence
                    ]
                    rel_accuracies.append(
                        calculate_relationship_accuracy(retrieved_edges, ex.expected_edges)
                    )

        def _mean(vals: List[float]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "total_queries": len(dataset),
            "hit_rate_at_10": _mean(hit_rates),
            "recall_at_10": _mean(recalls),
            "mrr": _mean(mrrs),
            "relationship_accuracy": _mean(rel_accuracies),
            "path_accuracy": _mean(path_exact_matches),
        }