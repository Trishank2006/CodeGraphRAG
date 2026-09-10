import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


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
            collection.name for collection in collections.collections
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