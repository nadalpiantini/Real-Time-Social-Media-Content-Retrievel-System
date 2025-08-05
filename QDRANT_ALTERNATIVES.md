# Qdrant Deployment Alternatives

## Problem Statement

Docker Hub authentication issues prevent downloading the official Qdrant image:
```
Error: failed to authorize: failed to fetch oauth token: 401 Unauthorized
```

This document outlines alternative solutions for deploying Qdrant vector database.

## Solution 1: Qdrant Cloud (Recommended)

### Advantages
- ✅ No local installation required
- ✅ Managed service with automatic scaling
- ✅ Built-in monitoring and backups
- ✅ Global distribution
- ✅ Free tier available

### Setup Steps

1. **Create Account**
   ```bash
   # Visit https://cloud.qdrant.io
   # Sign up for free account
   # Create new cluster
   ```

2. **Get Connection Details**
   ```bash
   # From Qdrant Cloud dashboard
   QDRANT_URL="https://your-cluster-id.qdrant.io"
   QDRANT_API_KEY="your-api-key"
   ```

3. **Update Configuration**
   ```python
   # In utils/qdrant.py
   import os
   from qdrant_client import QdrantClient
   
   def build_qdrant_client(url=None):
       if url and url == ':memory:':
           return QdrantClient(":memory:")
       
       # Try cloud first
       cloud_url = os.getenv('QDRANT_URL')
       cloud_key = os.getenv('QDRANT_API_KEY')
       
       if cloud_url and cloud_key:
           return QdrantClient(url=cloud_url, api_key=cloud_key)
       
       # Fallback to in-memory
       return QdrantClient(":memory:")
   ```

### Environment Configuration
```bash
# .env file
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

## Solution 2: Native Qdrant Installation

### Linux/macOS Installation
```bash
# Download binary
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz

# Make executable
chmod +x qdrant

# Start server
./qdrant --config-path ./config.yaml
```

### Configuration File (config.yaml)
```yaml
service:
  http_port: 6333
  grpc_port: 6334

storage:
  storage_path: ./qdrant_storage

cluster:
  enabled: false

telemetry:
  disabled: true
```

### Python Package Alternative
```bash
# Install Qdrant as Python package (experimental)
pip install qdrant-engine
```

## Solution 3: Enhanced In-Memory Mode

### Persistent In-Memory with Serialization

```python
# utils/persistent_memory.py
import os
import pickle
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

class PersistentMemoryQdrant:
    """Enhanced in-memory Qdrant with persistence simulation"""
    
    def __init__(self, storage_path="./qdrant_memory_backup"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.client = QdrantClient(":memory:")
        self._load_from_disk()
    
    def _get_backup_path(self, collection_name):
        return self.storage_path / f"{collection_name}.backup"
    
    def _save_to_disk(self, collection_name):
        """Save collection data to disk"""
        try:
            # Get all points from collection
            result = self.client.scroll(
                collection_name=collection_name,
                limit=10000  # Adjust based on your needs
            )
            
            backup_data = {
                'points': result[0],
                'collection_info': self.client.get_collection(collection_name)
            }
            
            with open(self._get_backup_path(collection_name), 'wb') as f:
                pickle.dump(backup_data, f)
                
        except Exception as e:
            print(f"Warning: Could not backup collection {collection_name}: {e}")
    
    def _load_from_disk(self):
        """Load collections from disk"""
        for backup_file in self.storage_path.glob("*.backup"):
            collection_name = backup_file.stem
            
            try:
                with open(backup_file, 'rb') as f:
                    backup_data = pickle.load(f)
                
                # Recreate collection
                collection_info = backup_data['collection_info']
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=collection_info.config.params.vectors
                )
                
                # Restore points
                if backup_data['points']:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=backup_data['points']
                    )
                
                print(f"✅ Restored collection: {collection_name}")
                
            except Exception as e:
                print(f"Warning: Could not restore collection {collection_name}: {e}")
    
    def upsert(self, collection_name, points):
        """Upsert with automatic backup"""
        result = self.client.upsert(collection_name, points)
        self._save_to_disk(collection_name)
        return result
    
    def create_collection(self, collection_name, vectors_config):
        """Create collection with backup setup"""
        result = self.client.create_collection(collection_name, vectors_config)
        self._save_to_disk(collection_name)
        return result
    
    def __getattr__(self, name):
        """Delegate all other methods to the underlying client"""
        return getattr(self.client, name)
```

### Integration in utils/qdrant.py
```python
def build_qdrant_client(url=None):
    """Build Qdrant client with fallback options"""
    
    # Option 1: Explicit URL (including :memory:)
    if url:
        if url == ':memory:':
            return QdrantClient(":memory:")
        return QdrantClient(url=url)
    
    # Option 2: Cloud configuration
    cloud_url = os.getenv('QDRANT_URL')
    cloud_key = os.getenv('QDRANT_API_KEY')
    
    if cloud_url and cloud_key:
        try:
            return QdrantClient(url=cloud_url, api_key=cloud_key)
        except Exception as e:
            print(f"⚠️ Cloud Qdrant connection failed: {e}")
    
    # Option 3: Local server
    try:
        client = QdrantClient("localhost", port=6333)
        # Test connection
        client.get_collections()
        return client
    except Exception:
        print("⚠️ Local Qdrant server not available")
    
    # Option 4: Enhanced in-memory with persistence
    try:
        from .persistent_memory import PersistentMemoryQdrant
        print("🔄 Using enhanced in-memory mode with persistence")
        return PersistentMemoryQdrant()
    except ImportError:
        pass
    
    # Option 5: Basic in-memory fallback
    print("🔄 Falling back to basic in-memory Qdrant client")
    return QdrantClient(":memory:")
```

## Solution 4: Alternative Vector Databases

### ChromaDB Alternative
```python
# Alternative implementation with ChromaDB
import chromadb
from chromadb.config import Settings

class ChromaDBRetriever:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
    
    def create_collection(self, name="posts"):
        self.collection = self.client.get_or_create_collection(name=name)
        return self.collection
    
    def add_documents(self, texts, metadatas, ids):
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query, n_results=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
```

### FAISS Alternative
```python
# utils/faiss_client.py
import faiss
import numpy as np
import pickle
import os

class FAISSVectorStore:
    def __init__(self, dimension=384, storage_path="./faiss_storage"):
        self.dimension = dimension
        self.storage_path = storage_path
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.metadata = {}
        self.counter = 0
        
        # Load existing index if available
        self._load_index()
    
    def add_vectors(self, vectors, metadata_list):
        """Add vectors with metadata"""
        vectors = np.array(vectors).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Add to index
        start_id = self.counter
        self.index.add(vectors)
        
        # Store metadata
        for i, metadata in enumerate(metadata_list):
            self.metadata[start_id + i] = metadata
        
        self.counter += len(vectors)
        self._save_index()
    
    def search(self, query_vector, k=5):
        """Search for similar vectors"""
        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append({
                    'score': float(score),
                    'metadata': self.metadata.get(idx, {})
                })
        
        return results
    
    def _save_index(self):
        """Save index and metadata to disk"""
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.storage_path, "index.faiss"))
        
        # Save metadata
        with open(os.path.join(self.storage_path, "metadata.pkl"), 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'counter': self.counter,
                'dimension': self.dimension
            }, f)
    
    def _load_index(self):
        """Load index and metadata from disk"""
        index_path = os.path.join(self.storage_path, "index.faiss")
        metadata_path = os.path.join(self.storage_path, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data['metadata']
                    self.counter = data['counter']
                    
                print(f"✅ Loaded FAISS index with {self.counter} vectors")
                
            except Exception as e:
                print(f"Warning: Could not load existing index: {e}")
```

## Solution 5: Docker Alternatives

### Podman (Docker Alternative)
```bash
# Install Podman
brew install podman  # macOS
# or
sudo apt-get install podman  # Ubuntu

# Run Qdrant with Podman
podman run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage docker.io/qdrant/qdrant
```

### Manual Docker Image Build
```dockerfile
# Build Qdrant from source
FROM rust:1.70 as builder

RUN git clone https://github.com/qdrant/qdrant.git
WORKDIR /qdrant
RUN cargo build --release

FROM debian:bullseye-slim
COPY --from=builder /qdrant/target/release/qdrant /usr/local/bin/qdrant
EXPOSE 6333 6334
CMD ["qdrant"]
```

## Recommendation Matrix

| Solution | Complexity | Reliability | Performance | Cost |
|----------|------------|-------------|-------------|------|
| Qdrant Cloud | Low | High | High | $$ |
| Native Install | Medium | Medium | High | Free |
| Enhanced Memory | Low | Medium | Medium | Free |
| ChromaDB | Low | Medium | Medium | Free |
| FAISS | Medium | High | High | Free |

## Implementation Priority

1. **Immediate (Today)**: Enhanced in-memory with persistence
2. **Short-term (This week)**: Qdrant Cloud setup  
3. **Long-term (Next month)**: Native installation or alternatives

## Next Steps

1. Implement enhanced in-memory solution
2. Create Qdrant Cloud account and test
3. Update deployment documentation
4. Add configuration examples for all options