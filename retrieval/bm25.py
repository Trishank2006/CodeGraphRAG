import math
import re
import threading
from collections import Counter

from retrieval.models import RetrievalResult


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")
SEPARATOR_PATTERN = re.compile(r"[_./:-]+")

# Splits CamelCase and acronym boundaries:
# BM25Retriever -> BM25 | Retriever, LLMClient -> LLM | Client.
CAMEL_CASE_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)

# Higher weights prioritize exact identifiers and repository paths.
FIELD_WEIGHTS = {
    "symbol": 3.0,
    "file_path": 2.0,
    "content": 1.0,
    "language": 0.25,
}


def _tokenize(text: str) -> list[str]:
    """Keep whole identifiers, then add snake_case, path, and CamelCase parts.

    Each part is added once, so a plain word such as ``context`` is not
    double-counted.
    """
    tokens: list[str] = []

    for raw_token in TOKEN_PATTERN.findall(text):
        tokens.append(raw_token.lower())

        parts = [part for part in SEPARATOR_PATTERN.split(raw_token) if part]
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)

        for part in parts:
            pieces = [
                piece for piece in CAMEL_CASE_BOUNDARY.split(part) if piece
            ]
            if len(pieces) > 1:
                tokens.extend(piece.lower() for piece in pieces)

    return tokens


class BM25Retriever:
    """In-memory field-aware BM25 retriever for code chunks.

    ``index`` builds a complete new index before swapping it in under a
    lock, so concurrent searches never see a half-built index.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        field_weights: dict[str, float] | None = None,
    ) -> None:
        # Validate BM25 parameters, including NaN and infinity.
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and greater than 0")

        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between 0 and 1")

        self.k1 = k1
        self.b = b

        self.field_weights = dict(
            FIELD_WEIGHTS if field_weights is None else field_weights
        )

        if set(self.field_weights) != set(FIELD_WEIGHTS):
            raise ValueError(
                f"field_weights must contain exactly: {list(FIELD_WEIGHTS)}"
            )

        if any(
            not math.isfinite(weight) or weight < 0
            for weight in self.field_weights.values()
        ):
            raise ValueError("field weights must be finite and non-negative")

        if not any(self.field_weights.values()):
            raise ValueError("at least one field weight must be positive")

        self._lock = threading.RLock()

        self.chunks: list[dict] = []

        # Retain aggregate index attributes for compatibility.
        self.documents: list[list[str]] = []
        self.document_lengths: list[int] = []
        self.term_frequencies: list[dict[str, int]] = []
        self.document_frequencies: dict[str, int] = {}
        self.average_document_length = 0.0

        # Independent indexes prevent long source code from diluting
        # symbol and path matches.
        self.field_documents: dict[str, list[list[str]]] = {}
        self.field_lengths: dict[str, list[int]] = {}
        self.field_term_frequencies: dict[str, list[dict[str, int]]] = {}
        self.field_document_frequencies: dict[str, dict[str, int]] = {}
        self.field_average_lengths: dict[str, float] = {}

    def index(self, chunks: list[dict]) -> None:
        """Build aggregate and field-specific indexes from code chunks."""
        new_chunks = list(chunks)
        fields = tuple(FIELD_WEIGHTS)

        documents: list[list[str]] = []
        document_lengths: list[int] = []
        term_frequencies: list[dict[str, int]] = []
        document_frequencies: dict[str, int] = {}

        field_documents: dict[str, list[list[str]]] = {f: [] for f in fields}
        field_lengths: dict[str, list[int]] = {f: [] for f in fields}
        field_term_frequencies: dict[str, list[dict[str, int]]] = {
            f: [] for f in fields
        }
        field_document_frequencies: dict[str, dict[str, int]] = {
            f: {} for f in fields
        }

        for chunk in new_chunks:
            field_text = {
                "symbol": str(chunk.get("symbol") or ""),
                "file_path": str(chunk.get("file_path") or ""),
                "content": str(chunk.get("content") or ""),
                "language": str(chunk.get("language") or ""),
            }

            combined_tokens: list[str] = []

            for field_name, text in field_text.items():
                tokens = _tokenize(text)
                combined_tokens.extend(tokens)
                frequencies = Counter(tokens)

                field_documents[field_name].append(tokens)
                field_lengths[field_name].append(len(tokens))
                field_term_frequencies[field_name].append(dict(frequencies))

                df = field_document_frequencies[field_name]
                for token in frequencies:
                    df[token] = df.get(token, 0) + 1

            # Preserve aggregate index data for existing callers/tests.
            aggregate_frequencies = Counter(combined_tokens)
            documents.append(combined_tokens)
            document_lengths.append(len(combined_tokens))
            term_frequencies.append(dict(aggregate_frequencies))

            for token in aggregate_frequencies:
                document_frequencies[token] = (
                    document_frequencies.get(token, 0) + 1
                )

        average_document_length = (
            sum(document_lengths) / len(document_lengths)
            if document_lengths
            else 0.0
        )
        field_average_lengths = {
            f: (
                sum(field_lengths[f]) / len(field_lengths[f])
                if field_lengths[f]
                else 0.0
            )
            for f in fields
        }

        with self._lock:
            self.chunks = new_chunks
            self.documents = documents
            self.document_lengths = document_lengths
            self.term_frequencies = term_frequencies
            self.document_frequencies = document_frequencies
            self.average_document_length = average_document_length
            self.field_documents = field_documents
            self.field_lengths = field_lengths
            self.field_term_frequencies = field_term_frequencies
            self.field_document_frequencies = field_document_frequencies
            self.field_average_lengths = field_average_lengths

    def _idf(self, token: str) -> float:
        """Calculate aggregate BM25 inverse document frequency."""
        document_count = len(self.documents)
        document_frequency = self.document_frequencies.get(token, 0)

        if document_count == 0:
            return 0.0

        return math.log(
            1
            + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def _field_idf(self, field: str, token: str) -> float:
        """Calculate IDF using statistics from one field."""
        document_count = len(self.chunks)
        document_frequency = (
            self.field_document_frequencies[field].get(token, 0)
        )

        if document_count == 0:
            return 0.0

        return math.log(
            1
            + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def _score_field(
        self,
        query_tokens: list[str],
        field: str,
        document_index: int,
    ) -> float:
        """Score one document field with BM25."""
        frequencies = self.field_term_frequencies[field][document_index]
        document_length = self.field_lengths[field][document_index]
        average_length = self.field_average_lengths[field]

        score = 0.0

        for token in query_tokens:
            term_frequency = frequencies.get(token, 0)

            if term_frequency == 0:
                continue

            denominator = term_frequency + self.k1 * (
                1
                - self.b
                + self.b * document_length / max(average_length, 1.0)
            )

            score += self._field_idf(field, token) * (
                term_frequency * (self.k1 + 1) / denominator
            )

        return score

    def _score_document(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:
        """Combine field-specific BM25 scores using configured weights."""
        return sum(
            weight * self._score_field(query_tokens, field, document_index)
            for field, weight in self.field_weights.items()
            if weight > 0
        )

    def search_bm25(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """Search code chunks using weighted field-aware BM25."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        if not query or not query.strip() or top_k == 0:
            return []

        # Preserve order while preventing duplicate query terms from
        # being counted repeatedly.
        query_tokens = list(dict.fromkeys(_tokenize(query)))

        if not query_tokens:
            return []

        scored_results: list[RetrievalResult] = []

        with self._lock:
            for index, chunk in enumerate(self.chunks):
                score = self._score_document(query_tokens, index)

                if score <= 0:
                    continue

                scored_results.append(
                    RetrievalResult(
                        id=chunk["chunk_id"],
                        source="bm25",
                        score=score,
                        repository=chunk["repository"],
                        file_path=chunk["file_path"],
                        language=chunk["language"],
                        symbol=chunk["symbol"],
                        start_line=chunk["start_line"],
                        end_line=chunk["end_line"],
                        content=chunk["content"],
                    )
                )

        scored_results.sort(key=lambda result: (-result.score, result.id))

        return scored_results[:top_k]