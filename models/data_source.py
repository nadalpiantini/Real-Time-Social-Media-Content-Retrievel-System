"""
Unified data source that can read from JSON files or Supabase
Maintains compatibility with existing bytewax flow
"""
import datetime
import json
import os
from pathlib import Path
from typing import Iterable, List

from bytewax.inputs import DynamicSource, StatelessSourcePartition
from utils.supabase_client import supabase_client


def json_generator_from_files(json_files: List[Path]):
    """Original JSON file generator"""
    print(f"📁 Starting to process {len(json_files)} JSON files")
    
    for json_file in json_files:
        print(f"📖 Reading file: {json_file}")
        
        if not json_file.exists():
            print(f"⚠️ File not found: {json_file}")
            continue
            
        try:
            with json_file.open() as f:
                data = json.load(f)
            
            posts = data.get("Posts", {})
            print(f"📊 File {json_file.name}: {len(posts)} posts found")
            
            if len(posts) == 0:
                print(f"⚠️ File {json_file.name} contains no posts - skipping")
                continue
                
            print(f"✅ Yielding {len(posts)} posts from {json_file.name}")
            yield list(posts.items())
            
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")


def json_generator_from_supabase():
    """Generate data from Supabase in same format as JSON files"""
    try:
        all_posts = supabase_client.get_all_posts()
        
        if not all_posts:
            return
        
        # Group posts by source_name to maintain original structure
        sources_data = {}
        for post in all_posts:
            source_name = post.get('source_name', 'unknown')
            if source_name not in sources_data:
                sources_data[source_name] = {}
            
            sources_data[source_name][post['post_id']] = {
                'text': post['text'],
                'post_owner': post['post_owner'],
                'source': post['platform']
            }
        
        # Yield each source's posts
        for source_name, posts in sources_data.items():
            yield list(posts.items())
    
    except Exception as e:
        print(f"Error loading from Supabase: {e}")
        return


class UnifiedDataPartition(StatelessSourcePartition):
    """Partition that can handle both JSON files and Supabase data"""
    
    def __init__(self, json_files: List[str] = None, use_supabase: bool = False):
        self.use_supabase = use_supabase and supabase_client.is_available()
        
        if self.use_supabase:
            self._generator = json_generator_from_supabase()
        else:
            # Fallback to JSON files
            if json_files:
                json_files_paths = [Path(json_file) for json_file in json_files]
                self._generator = json_generator_from_files(json_files=json_files_paths)
            else:
                # Auto-discover JSON files in data folder
                data_folder = Path("data")
                if data_folder.exists():
                    json_files_paths = list(data_folder.glob("*_data.json"))
                    self._generator = json_generator_from_files(json_files=json_files_paths)
                else:
                    self._generator = iter([])
    
    def next_batch(self) -> Iterable[dict]:
        try:
            return next(self._generator)
        except StopIteration:
            return []


class UnifiedDataSource(DynamicSource):
    """
    Unified data source that can read from JSON files or Supabase
    Maintains backward compatibility with JSONSource
    """
    
    def __init__(self, json_files: List[str] = None, use_supabase: bool = None):
        self._json_files = json_files or []
        
        # Auto-detect if we should use Supabase
        if use_supabase is None:
            self.use_supabase = supabase_client.is_available()
        else:
            self.use_supabase = use_supabase
    
    def build(
        self, now: datetime.datetime, worker_index: int, worker_count: int
    ) -> UnifiedDataPartition:
        
        if self.use_supabase:
            # For Supabase, we don't need to split across workers for now
            # since we're loading all data in memory
            return UnifiedDataPartition(use_supabase=True)
        else:
            # Original JSON file logic with worker distribution
            if not self._json_files:
                return UnifiedDataPartition(use_supabase=False)
            
            num_files_per_worker = len(self._json_files) // worker_count
            num_leftover_files = len(self._json_files) % worker_count

            start_index = worker_index * num_files_per_worker
            end_index = start_index + num_files_per_worker
            if worker_index == worker_count - 1:
                end_index += num_leftover_files

            return UnifiedDataPartition(
                json_files=self._json_files[start_index:end_index],
                use_supabase=False
            )


# Backward compatibility alias
JSONSource = UnifiedDataSource