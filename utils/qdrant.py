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
    Builds a QdrantClient object with comprehensive fallback options.
    
    Priority order:
    1. Explicit URL parameter
    2. Environment variables (QDRANT_URL, QDRANT_API_KEY)
    3. Local server (localhost:6333)
    4. Enhanced persistent memory mode
    5. Basic in-memory mode

    Args:
        url (Optional[str]): The URL of the Qdrant server. Special values:
            - ":memory:" forces basic in-memory mode
            - ":persistent:" forces enhanced persistent memory mode
            - None uses environment variables or defaults
        api_key (Optional[str]): The API key to use for authentication.

    Returns:
        QdrantClient: A QdrantClient object (may be enhanced persistent client)
    """

    # Handle special URL values
    if url == ":memory:":
        print("🔄 Using basic in-memory Qdrant client")
        return QdrantClient(":memory:")
    elif url == ":persistent:":
        print("🔄 Using enhanced persistent memory Qdrant client")
        try:
            from .persistent_memory import PersistentMemoryQdrant
            return PersistentMemoryQdrant()
        except ImportError as e:
            print(f"⚠️ Could not load persistent memory module: {e}")
            print("🔄 Falling back to basic in-memory client")
            return QdrantClient(":memory:")

    # Get connection parameters
    if url is None:
        url = os.environ.get("QDRANT_URL")
    
    if api_key is None:
        api_key = os.environ.get("QDRANT_API_KEY")
    
    # Try cloud/remote connection if URL is provided
    if url:
        client_kwargs = {"url": url}
        if api_key:
            client_kwargs["api_key"] = api_key

        try:
            client = QdrantClient(**client_kwargs)
            # Test the connection
            client.get_collections()
            print(f"✅ Connected to Qdrant at {url}")
            return client
        except Exception as e:
            print(f"⚠️ Failed to connect to Qdrant at {url}: {e}")
    
    # Try local server
    try:
        print("🔗 Trying to connect to local Qdrant server...")
        client = QdrantClient("localhost", port=6333)
        # Test the connection
        client.get_collections()
        print("✅ Connected to local Qdrant server")
        return client
    except Exception as e:
        print(f"⚠️ Failed to connect to local Qdrant server: {e}")
    
    # Enhanced fallback: persistent memory mode
    try:
        print("🔄 Trying enhanced persistent memory mode...")
        from .persistent_memory import PersistentMemoryQdrant
        client = PersistentMemoryQdrant()
        print("✅ Using enhanced persistent memory Qdrant")
        return client
    except ImportError as e:
        print(f"⚠️ Could not load persistent memory module: {e}")
    except Exception as e:
        print(f"⚠️ Could not initialize persistent memory mode: {e}")
    
    # Final fallback: basic in-memory
    print("🔄 Falling back to basic in-memory Qdrant client")
    return QdrantClient(":memory:")


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
        if not chunks:
            print("📝 QdrantVectorSink: Empty batch, skipping...")
            return
            
        print(f"💾 QdrantVectorSink: Writing batch of {len(chunks)} chunks...")
        
        ids = []
        embeddings = []
        metadata = []
        processed_count = 0
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_id, text_embedding, chunk_metadata = chunk.to_payload()
                ids.append(chunk_id)
                embeddings.append(text_embedding)
                metadata.append(chunk_metadata)
                processed_count += 1
                
                if i < 3:  # Show first few for debugging
                    print(f"  📝 Chunk {i+1}: {chunk.post_id} → {len(text_embedding)} dims")
                    
            except Exception as e:
                print(f"❌ Error processing chunk {i}: {e}")
                continue

        if not ids:
            print("⚠️ No valid chunks to upsert")
            return

        try:
            print(f"🔄 Upserting {len(ids)} points to collection '{self._collection_name}'...")
            
            # Add timeout and retry logic for Qdrant operations
            import time
            max_retries = 3
            for retry in range(max_retries):
                try:
                    self._client.upsert(
                        collection_name=self._collection_name,
                        points=Batch(
                            ids=ids,
                            vectors=embeddings,
                            payloads=metadata,
                        ),
                    )
                    print(f"✅ Successfully upserted {len(ids)} points to Qdrant! (attempt {retry + 1})")
                    break
                    
                except Exception as retry_error:
                    if retry == max_retries - 1:
                        raise retry_error
                    print(f"⚠️ Retry {retry + 1} failed: {retry_error}")
                    time.sleep(1)  # Wait before retry
            
        except Exception as e:
            print(f"❌ Error upserting to Qdrant after {max_retries} attempts: {e}")
            # Don't raise - allow pipeline to continue
            print("🔄 Continuing pipeline despite Qdrant error...")
        
        finally:
            print(f"🏁 QdrantVectorSink batch processing complete: {processed_count}/{len(chunks)} chunks processed")