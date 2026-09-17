from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for the hybrid retrieval pipeline."""

    dense_top_k: int = 20
    bm25_top_k: int = 20
    graph_top_k: int = 20
    rrf_top_k: int = 30
    rerank_top_k: int = 10
    rrf_k: int = 60

    def __post_init__(self) -> None:
        values = {
            "dense_top_k": self.dense_top_k,
            "bm25_top_k": self.bm25_top_k,
            "graph_top_k": self.graph_top_k,
            "rrf_top_k": self.rrf_top_k,
            "rerank_top_k": self.rerank_top_k,
            "rrf_k": self.rrf_k,
        }

        for name, value in values.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")