"""
Enhanced in-memory Qdrant with persistence simulation
Provides a fallback solution when Docker/Cloud Qdrant is not available
"""

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class PersistentMemoryQdrant:
    """Enhanced in-memory Qdrant with persistence simulation"""
    
    def __init__(self, storage_path: str = "./qdrant_memory_backup"):
        """
        Initialize persistent memory Qdrant client
        
        Args:
            storage_path: Directory to store backup files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.client = QdrantClient(":memory:")
        self.collections = set()
        
        # Load existing collections from disk
        self._load_from_disk()
        
        print(f"🔄 Using enhanced in-memory Qdrant with persistence at {storage_path}")
    
    def _get_backup_path(self, collection_name: str) -> Path:
        """Get backup file path for collection"""
        return self.storage_path / f"{collection_name}.backup"
    
    def _get_config_path(self, collection_name: str) -> Path:
        """Get configuration file path for collection"""
        return self.storage_path / f"{collection_name}.config.json"
    
    def _save_to_disk(self, collection_name: str) -> bool:
        """
        Save collection data to disk
        
        Args:
            collection_name: Name of collection to backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            
            # Save collection configuration
            config_data = {
                'name': collection_name,
                'vectors_config': {
                    'size': collection_info.config.params.vectors.get('', VectorParams()).size,
                    'distance': collection_info.config.params.vectors.get('', VectorParams()).distance.value
                }
            }
            
            with open(self._get_config_path(collection_name), 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            # Get all points from collection
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=10000  # Adjust based on your needs
            )
            
            # Convert points to serializable format
            serializable_points = []
            for point in points:
                serializable_points.append({
                    'id': point.id,
                    'vector': point.vector,
                    'payload': point.payload
                })
            
            backup_data = {
                'collection_name': collection_name,
                'points': serializable_points,
                'point_count': len(serializable_points),
                'timestamp': str(Path().resolve())
            }
            
            with open(self._get_backup_path(collection_name), 'wb') as f:
                pickle.dump(backup_data, f)
            
            print(f"💾 Backed up collection '{collection_name}' ({len(serializable_points)} points)")
            return True
                
        except Exception as e:
            print(f"⚠️ Could not backup collection {collection_name}: {e}")
            return False
    
    def _load_from_disk(self) -> int:
        """
        Load collections from disk
        
        Returns:
            Number of collections loaded
        """
        loaded_count = 0
        
        for backup_file in self.storage_path.glob("*.backup"):
            collection_name = backup_file.stem
            config_file = self._get_config_path(collection_name)
            
            try:
                # Load configuration
                if not config_file.exists():
                    print(f"⚠️ No config file for {collection_name}, skipping")
                    continue
                
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load backup data
                with open(backup_file, 'rb') as f:
                    backup_data = pickle.load(f)
                
                # Recreate collection
                vectors_config = VectorParams(
                    size=config_data['vectors_config']['size'],
                    distance=Distance(config_data['vectors_config']['distance'])
                )
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config
                )
                
                # Restore points
                if backup_data['points']:
                    # Convert back to PointStruct format
                    points = []
                    for point_data in backup_data['points']:
                        points.append(PointStruct(
                            id=point_data['id'],
                            vector=point_data['vector'], 
                            payload=point_data['payload']
                        ))
                    
                    # Batch upsert
                    batch_size = 100
                    for i in range(0, len(points), batch_size):
                        batch = points[i:i + batch_size]
                        self.client.upsert(
                            collection_name=collection_name,
                            points=batch
                        )
                
                self.collections.add(collection_name)
                loaded_count += 1
                
                print(f"✅ Restored collection '{collection_name}' ({len(backup_data['points'])} points)")
                
            except Exception as e:
                print(f"⚠️ Could not restore collection {collection_name}: {e}")
        
        if loaded_count > 0:
            print(f"📁 Loaded {loaded_count} collections from persistent storage")
        
        return loaded_count
    
    def create_collection(self, collection_name: str, vectors_config: VectorParams, **kwargs):
        """Create collection with backup setup"""
        result = self.client.create_collection(
            collection_name=collection_name, 
            vectors_config=vectors_config,
            **kwargs
        )
        
        self.collections.add(collection_name)
        self._save_to_disk(collection_name)
        return result
    
    def upsert(self, collection_name: str, points: List[PointStruct], **kwargs):
        """Upsert with automatic backup"""
        result = self.client.upsert(
            collection_name=collection_name, 
            points=points,
            **kwargs
        )
        
        # Auto-save after upsert operations
        self._save_to_disk(collection_name)
        return result
    
    def delete_collection(self, collection_name: str):
        """Delete collection and remove backups"""
        result = self.client.delete_collection(collection_name)
        
        # Remove from tracking
        self.collections.discard(collection_name)
        
        # Remove backup files
        backup_path = self._get_backup_path(collection_name)
        config_path = self._get_config_path(collection_name)
        
        try:
            if backup_path.exists():
                backup_path.unlink()
            if config_path.exists():
                config_path.unlink()
            print(f"🗑️ Removed backups for collection '{collection_name}'")
        except Exception as e:
            print(f"⚠️ Could not remove backup files: {e}")
        
        return result
    
    def get_collections(self):
        """Get collections with enhanced info"""
        result = self.client.get_collections()
        
        # Add backup info
        for collection in result.collections:
            backup_path = self._get_backup_path(collection.name)
            if backup_path.exists():
                collection.backup_available = True
                collection.backup_size = backup_path.stat().st_size
            else:
                collection.backup_available = False
        
        return result
    
    def cleanup_old_backups(self, max_backups: int = 5):
        """Clean up old backup files, keeping only the most recent ones"""
        backup_files = list(self.storage_path.glob("*.backup"))
        
        if len(backup_files) > max_backups:
            # Sort by modification time
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old files
            for old_file in backup_files[max_backups:]:
                try:
                    old_file.unlink()
                    # Also remove corresponding config
                    config_file = self._get_config_path(old_file.stem)
                    if config_file.exists():
                        config_file.unlink()
                    print(f"🧹 Removed old backup: {old_file.name}")
                except Exception as e:
                    print(f"⚠️ Could not remove old backup {old_file}: {e}")
    
    def get_backup_info(self) -> Dict[str, Any]:
        """Get information about backup storage"""
        backup_files = list(self.storage_path.glob("*.backup"))
        total_size = sum(f.stat().st_size for f in backup_files)
        
        return {
            'storage_path': str(self.storage_path),
            'total_collections': len(backup_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'collections': [f.stem for f in backup_files]
        }
    
    def __getattr__(self, name):
        """Delegate all other methods to the underlying client"""
        return getattr(self.client, name)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save all collections"""
        for collection_name in self.collections:
            self._save_to_disk(collection_name)


def create_persistent_qdrant_client(storage_path: str = "./qdrant_memory_backup") -> PersistentMemoryQdrant:
    """
    Factory function to create a persistent memory Qdrant client
    
    Args:
        storage_path: Directory to store backup files
        
    Returns:
        PersistentMemoryQdrant instance
    """
    return PersistentMemoryQdrant(storage_path=storage_path)