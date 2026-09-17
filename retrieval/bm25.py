import math
import re

from retrieval.models import RetrievalResult


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")


def _tokenize(text: str) -> list[str]:
    """
    Tokenize code and metadata while preserving:
    - complete identifiers
    - file paths
    - individual identifier/path components
    """

    raw_tokens = TOKEN_PATTERN.findall(text.lower())

    tokens = []

    for token in raw_tokens:
        # Keep the complete token.
        tokens.append(token)

        # Also split common code/path separators so that
        # process_payment can match "payment", and
        # src/payment/service.py can match "payment" or "service".
        parts = re.split(r"[_./:-]+", token)

        for part in parts:
            if part:
                tokens.append(part)

    return tokens


class BM25Retriever:
    """
    In-memory BM25 retriever for code chunks.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        self.chunks: list[dict] = []
        self.documents: list[list[str]] = []
        self.document_lengths: list[int] = []

        self.term_frequencies: list[dict[str, int]] = []
        self.document_frequencies: dict[str, int] = {}

        self.average_document_length = 0.0

    def index(self, chunks: list[dict]) -> None:
        """
        Build the BM25 index from code chunks.

        Each searchable document contains:
        - symbol
        - file path
        - language
        - source code
        """

        self.chunks = list(chunks)
        self.documents = []
        self.document_lengths = []
        self.term_frequencies = []
        self.document_frequencies = {}

        for chunk in self.chunks:
            searchable_text = " ".join(
                [
                    chunk.get("symbol", ""),
                    chunk.get("file_path", ""),
                    chunk.get("language", ""),
                    chunk.get("content", ""),
                ]
            )

            tokens = _tokenize(searchable_text)

            self.documents.append(tokens)
            self.document_lengths.append(len(tokens))

            frequencies: dict[str, int] = {}

            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            self.term_frequencies.append(frequencies)

            # Count each term once per document.
            for token in frequencies:
                self.document_frequencies[token] = (
                    self.document_frequencies.get(token, 0) + 1
                )

        if self.document_lengths:
            self.average_document_length = (
                sum(self.document_lengths)
                / len(self.document_lengths)
            )
        else:
            self.average_document_length = 0.0

    def _idf(self, token: str) -> float:
        """
        Compute BM25 inverse document frequency.
        """

        document_count = len(self.documents)
        document_frequency = self.document_frequencies.get(token, 0)

        return math.log(
            1
            + (
                document_count
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )

    def _score_document(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:
        """
        Calculate the BM25 score for one document.
        """

        if not self.documents:
            return 0.0

        frequencies = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]

        score = 0.0

        for token in query_tokens:
            term_frequency = frequencies.get(token, 0)

            if term_frequency == 0:
                continue

            idf = self._idf(token)

            denominator = (
                term_frequency
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * document_length
                    / max(self.average_document_length, 1.0)
                )
            )

            score += (
                idf
                * (
                    term_frequency
                    * (self.k1 + 1)
                    / denominator
                )
            )

        return score

    def search_bm25(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Search indexed code chunks using BM25.
        """

        if not self.chunks:
            return []

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scored_results: list[RetrievalResult] = []

        for index, chunk in enumerate(self.chunks):
            score = self._score_document(
                query_tokens,
                index,
            )

            if score <= 0:
                continue

            result = RetrievalResult(
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

            scored_results.append(result)

        scored_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return scored_results[:top_k]