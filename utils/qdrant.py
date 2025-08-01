import os
from typing import Optional

from bytewax.outputs import DynamicSink, StatelessSinkPartition
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Batch

from models.settings import settings
from models.post import EmbeddedChunkedPost


class QdrantVectorOutput(DynamicSink):
    """A class representing a Qdrant vector output.

    This class is used to create a Qdrant vector output, which is a type of dynamic output that supports
    at-least-once processing. Messages from the resume epoch will be duplicated right after resume.

    Args:
        vector_size (int): The size of the vector.
        collection_name (str, optional): The name of the collection.
            Defaults to settings.VECTOR_DB_OUTPUT_COLLECTION_NAME.
        client (Optional[QdrantClient], optional): The Qdrant client. Defaults to None.
    """

    def __init__(
        self,
        vector_size: int,
        collection_name: str = settings.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        client: Optional[QdrantClient] = None,
    ):
        self._collection_name = collection_name
        self._vector_size = vector_size

        if client:
            self.client = client
        else:
            self.client = build_qdrant_client()

        try:
            self.client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def build(self, step_id:str, worker_index:int, worker_count:int) -> "QdrantVectorSink":
        """Builds a QdrantVectorSink object.

        Args:
            worker_index (int): The index of the worker.
            worker_count (int): The total number of workers.

        Returns:
            QdrantVectorSink: A QdrantVectorSink object.
        """

        return QdrantVectorSink(self.client, self._collection_name)


def build_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Builds a QdrantClient object with the given URL and API key.
    Falls back to in-memory mode if connection fails.

    Args:
        url (Optional[str]): The URL of the Qdrant server. If not provided,
            it will be read from the QDRANT_URL environment variable.
        api_key (Optional[str]): The API key to use for authentication. If not provided,
            it will be read from the QDRANT_API_KEY environment variable.

    Returns:
        QdrantClient: A QdrantClient object connected to the specified Qdrant server
                     or in-memory client as fallback.
    """

    # Try to connect to configured Qdrant server
    if url is None:
        url = os.environ.get("QDRANT_URL", "localhost:6333")
    
    if api_key is None:
        api_key = os.environ.get("QDRANT_API_KEY")
    
    # Build client kwargs
    client_kwargs = {"url": url}
    if api_key:
        client_kwargs["api_key"] = api_key

    try:
        # Try to connect to the configured server
        client = QdrantClient(**client_kwargs)
        # Test the connection
        client.get_collections()
        print(f"✅ Connected to Qdrant at {url}")
        return client
    except Exception as e:
        print(f"⚠️  Failed to connect to Qdrant at {url}: {e}")
        print("🔄 Falling back to in-memory Qdrant client")
        # Fallback to in-memory client
        client = QdrantClient(":memory:")
        return client


class QdrantVectorSink(StatelessSinkPartition):
    """
    A sink that writes document embeddings to a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client to use for writing.
        collection_name (str, optional): The name of the collection to write to.
            Defaults to settings.VECTOR_DB_OUTPUT_COLLECTION_NAME.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = settings.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        self._client = client
        self._collection_name = collection_name

    def write_batch(self, chunks: list[EmbeddedChunkedPost]):
        ids = []
        embeddings = []
        metadata = []
        for chunk in chunks:
            chunk_id, text_embedding, chunk_metadata = chunk.to_payload()

            ids.append(chunk_id)
            embeddings.append(text_embedding)
            metadata.append(chunk_metadata)

        self._client.upsert(
            collection_name=self._collection_name,
            points=Batch(
                ids=ids,
                vectors=embeddings,
                payloads=metadata,
            ),
        )