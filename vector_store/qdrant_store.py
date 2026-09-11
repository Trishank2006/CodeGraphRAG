import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


DEFAULT_COLLECTION = "code_chunks"


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

    def create_collection(self) -> None:
        """
        Create the vector collection if it does not already exist.
        """
        collections = self.client.get_collections()

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """
        Insert or update code chunks and their embeddings.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk["chunk_id"],
                )
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=chunk,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        language: str | None = None,
        file_path: str | None = None,
    ) -> list[dict]:
        """
        Search for the most similar code chunks.
        """
        conditions = []

        if language is not None:
            conditions.append(
                FieldCondition(
                    key="language",
                    match=MatchValue(value=language),
                )
            )

        if file_path is not None:
            conditions.append(
                FieldCondition(
                    key="file_path",
                    match=MatchValue(value=file_path),
                )
            )

        query_filter = None

        if conditions:
            query_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "score": result.score,
                "payload": result.payload,
            }
            for result in results.points
        ]