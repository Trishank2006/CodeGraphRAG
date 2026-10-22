import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)


DEFAULT_COLLECTION = "code_chunks"
UPSERT_BATCH_SIZE = 256
SCROLL_BATCH_SIZE = 256


def point_id_for(chunk: dict) -> str:
    """Deterministic point ID, scoped by repository so the same code indexed
    under two names never shares (and overwrites) points."""
    key = f"{chunk.get('repository', '')}:{chunk['chunk_id']}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _validate_repository(repository: str) -> None:
    if not repository or not repository.strip():
        raise ValueError("Repository name must not be empty")


def _repository_condition(repository: str) -> FieldCondition:
    return FieldCondition(key="repository", match=MatchValue(value=repository))


class QdrantStore:
    def __init__(
        self,
        path: str = "data/qdrant",
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = 384,
    ):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.vector_size = vector_size

    def _collection_exists(self) -> bool:
        return any(
            collection.name == self.collection_name
            for collection in self.client.get_collections().collections
        )

    def create_collection(self) -> None:
        """Create the vector collection if it does not exist."""
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def delete_repository(self, repository: str) -> None:
        """Delete all vectors belonging to one repository; others are kept."""
        _validate_repository(repository)

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[_repository_condition(repository)])
            ),
            wait=True,
        )

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Insert or update code chunks and their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            return

        points = [
            PointStruct(id=point_id_for(chunk), vector=embedding, payload=chunk)
            for chunk, embedding in zip(chunks, embeddings)
        ]

        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start:start + UPSERT_BATCH_SIZE],
                wait=True,
            )

    def replace_repository(
        self,
        repository: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Replace one repository's points without an empty window.

        New points are upserted first (unchanged chunks keep their IDs and are
        overwritten in place); stale points are deleted afterwards.
        """
        _validate_repository(repository)

        if not chunks:
            raise ValueError(
                "replace_repository needs at least one chunk; "
                "use delete_repository to clear a repository"
            )

        if {chunk.get("repository") for chunk in chunks} != {repository}:
            raise ValueError(f"All chunks must belong to repository '{repository}'")

        self.upsert_chunks(chunks, embeddings)

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[_repository_condition(repository)],
                    must_not=[
                        HasIdCondition(
                            has_id=[point_id_for(chunk) for chunk in chunks]
                        )
                    ],
                )
            ),
            wait=True,
        )

    def count_repository(self, repository: str) -> int:
        """Return how many points a repository has (0 if nothing is indexed)."""
        if not self._collection_exists():
            return 0

        return self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(must=[_repository_condition(repository)]),
            exact=True,
        ).count

    def load_repository_chunks(self, repository: str) -> list[dict]:
        """Return the stored chunk payloads of one repository."""
        if not self._collection_exists():
            return []

        chunks: list[dict] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[_repository_condition(repository)]),
                limit=SCROLL_BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(dict(point.payload or {}) for point in points)

            if offset is None:
                return chunks

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        repository: str | None = None,
        language: str | None = None,
        file_path: str | None = None,
        path_prefix: str | None = None,
    ) -> list[dict]:
        """
        Search for similar code chunks.

        Supports exact language filtering, exact file-path filtering,
        and path-prefix filtering.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        if top_k == 0:
            return []

        conditions = []
        if repository is not None:
            conditions.append(_repository_condition(repository))

        if language is not None:
            conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )

        if file_path is not None:
            conditions.append(
                FieldCondition(key="file_path", match=MatchValue(value=file_path))
            )

        query_filter = Filter(must=conditions) if conditions else None

        # Prefix filtering happens after vector retrieval because Qdrant
        # exact-match filters do not implement arbitrary string prefixes.
        candidate_limit = top_k
        if path_prefix is not None:
            candidate_limit = max(top_k * 10, 100)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=candidate_limit,
            with_payload=True,
        )

        filtered_results = []

        for result in results.points:
            payload = result.payload or {}
            result_file_path = payload.get("file_path", "")

            if path_prefix is not None and not result_file_path.startswith(path_prefix):
                continue

            filtered_results.append({"score": result.score, "payload": payload})

            if len(filtered_results) >= top_k:
                break

        return filtered_results

    def ping(self) -> None:
        """Raise if the local store can't be read."""
        self.client.get_collections()

    def close(self) -> None:
        """Close the Qdrant client."""
        self.client.close()
