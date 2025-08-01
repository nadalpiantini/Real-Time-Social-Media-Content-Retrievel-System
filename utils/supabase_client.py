"""
Supabase client for SMContent system
Handles connection and data operations with Supabase
"""
import json
import os
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from models.settings import settings
import logging

logger = logging.getLogger(__name__)

class SupabaseClient:
    """Client for Supabase operations"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._connect()
    
    def _connect(self):
        """Initialize Supabase client"""
        try:
            if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                logger.warning("Supabase credentials not found, using local JSON files")
                return
                
            self.client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY
            )
            logger.info("Connected to Supabase successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Supabase is available"""
        return self.client is not None and settings.USE_SUPABASE
    
    def save_scraped_data(self, source_name: str, posts_data: Dict) -> bool:
        """
        Save scraped posts data to Supabase
        
        Args:
            source_name: Name of the source (e.g., 'manthanbhikadiya')
            posts_data: Posts data in original JSON format
            
        Returns:
            bool: Success status
        """
        if not self.is_available():
            return self._save_to_json_fallback(source_name, posts_data)
        
        try:
            # Insert source if not exists
            source_result = self.client.table('smc_sources').upsert({
                'name': source_name,
                'platform': 'Linkedin',
                'last_scraped': 'now()'
            }).execute()
            
            # Prepare posts for batch insert
            posts_to_insert = []
            for post_id, post_data in posts_data.get('Posts', {}).items():
                posts_to_insert.append({
                    'post_id': post_id,
                    'source_name': source_name,
                    'platform': post_data.get('source', 'Linkedin'),
                    'text': post_data.get('text', ''),
                    'full_raw_text': post_data.get('text', ''),
                    'post_owner': post_data.get('post_owner', post_data.get('name', source_name))
                })
            
            # Batch insert posts (upsert to avoid duplicates)
            if posts_to_insert:
                result = self.client.table('smc_posts').upsert(
                    posts_to_insert,
                    on_conflict='post_id,source_name'
                ).execute()
                
                logger.info(f"Saved {len(posts_to_insert)} posts for {source_name} to Supabase")
                return True
                
        except Exception as e:
            logger.error(f"Error saving to Supabase: {e}")
            return self._save_to_json_fallback(source_name, posts_data)
        
        return False
    
    def get_posts_by_source(self, source_name: str) -> Optional[Dict]:
        """
        Get posts by source name in original JSON format
        
        Args:
            source_name: Name of the source
            
        Returns:
            Dict: Posts data in original format or None
        """
        if not self.is_available():
            return self._load_from_json_fallback(source_name)
        
        try:
            result = self.client.table('smc_posts').select('*').eq('source_name', source_name).execute()
            
            if not result.data:
                return None
            
            # Convert to original JSON format
            posts_dict = {}
            for post in result.data:
                posts_dict[post['post_id']] = {
                    'text': post['text'],
                    'post_owner': post['post_owner'],
                    'source': post['platform']
                }
            
            return {
                'Name': source_name,
                'Posts': posts_dict
            }
            
        except Exception as e:
            logger.error(f"Error loading from Supabase: {e}")
            return self._load_from_json_fallback(source_name)
    
    def get_all_posts(self) -> List[Dict]:
        """
        Get all posts from all sources
        
        Returns:
            List[Dict]: List of all posts
        """
        if not self.is_available():
            return self._load_all_json_files()
        
        try:
            result = self.client.table('smc_posts').select('*').execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Error loading all posts from Supabase: {e}")
            return self._load_all_json_files()
    
    def log_search(self, query: str, results_count: int, execution_time_ms: int = 0):
        """Log search query for analytics"""
        if not self.is_available():
            return
        
        try:
            self.client.table('smc_searches').insert({
                'query': query,
                'results_count': results_count,
                'execution_time_ms': execution_time_ms
            }).execute()
        except Exception as e:
            logger.error(f"Error logging search: {e}")
    
    def _save_to_json_fallback(self, source_name: str, posts_data: Dict) -> bool:
        """Fallback to save data as JSON file"""
        try:
            os.makedirs(settings.DATA_FOLDER, exist_ok=True)
            file_path = os.path.join(settings.DATA_FOLDER, f"{source_name}_data.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(posts_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {source_name} data to JSON file as fallback")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON fallback: {e}")
            return False
    
    def _load_from_json_fallback(self, source_name: str) -> Optional[Dict]:
        """Fallback to load data from JSON file"""
        try:
            file_path = os.path.join(settings.DATA_FOLDER, f"{source_name}_data.json")
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading JSON fallback: {e}")
            return None
    
    def _load_all_json_files(self) -> List[Dict]:
        """Load all JSON files from data folder"""
        posts = []
        try:
            if not os.path.exists(settings.DATA_FOLDER):
                return posts
            
            for filename in os.listdir(settings.DATA_FOLDER):
                if filename.endswith('_data.json'):
                    file_path = os.path.join(settings.DATA_FOLDER, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Convert to flat post format
                    for post_id, post_data in data.get('Posts', {}).items():
                        posts.append({
                            'post_id': post_id,
                            'source_name': data.get('Name', ''),
                            'text': post_data.get('text', ''),
                            'post_owner': post_data.get('post_owner', post_data.get('name', '')),
                            'platform': post_data.get('source', 'Linkedin')
                        })
                        
        except Exception as e:
            logger.error(f"Error loading JSON files: {e}")
        
        return posts

# Singleton instance
supabase_client = SupabaseClient()