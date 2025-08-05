"""
Enhanced Retriever with Analytics Integration
Wrapper around the existing QdrantVectorDBRetriever with metrics collection and advanced features
"""
import time
import uuid
from typing import List, Dict, Optional, Union, Any
from datetime import datetime

from utils.retriever import QdrantVectorDBRetriever
from utils.embedding import EmbeddingModelSingleton, CrossEncoderModelSingleton
from qdrant_client import QdrantClient
from models.post import EmbeddedChunkedPost
from analytics.metrics_collector import get_metrics_collector


class AnalyticsEnabledRetriever:
    """Enhanced retriever with analytics collection and advanced search features"""
    
    def __init__(self,
                 embedding_model: EmbeddingModelSingleton,
                 vector_db_client: QdrantClient,
                 cross_encoder_model: Optional[CrossEncoderModelSingleton] = None,
                 vector_db_collection: str = "posts"):
        
        # Initialize base retriever
        self.base_retriever = QdrantVectorDBRetriever(
            embedding_model=embedding_model,
            vector_db_client=vector_db_client,
            cross_encoder_model=cross_encoder_model,
            vector_db_collection=vector_db_collection
        )
        
        # Analytics
        self.metrics_collector = get_metrics_collector()
        
        # Session management
        self.session_id = str(uuid.uuid4())
    
    def search(self, 
              query: str, 
              limit: int = 3,
              filters: Optional[Dict[str, Any]] = None,
              return_all: bool = False) -> Union[List[EmbeddedChunkedPost], Dict[str, Any]]:
        """Enhanced search with analytics collection and optional filters"""
        
        start_time = time.time()
        
        try:
            # Apply filters if provided
            if filters:
                results = self._search_with_filters(query, limit, filters, return_all)
            else:
                results = self.base_retriever.search(query, limit, return_all)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Extract posts for analytics
            if return_all and isinstance(results, dict):
                posts = results.get("posts", [])
            else:
                posts = results if isinstance(results, list) else []
            
            # Record analytics
            self.metrics_collector.record_search(
                query=query,
                results=posts,
                processing_time_ms=processing_time_ms,
                session_id=self.session_id
            )
            
            return results
            
        except Exception as e:
            # Record failed search
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics_collector.record_search(
                query=query,
                results=[],
                processing_time_ms=processing_time_ms,
                session_id=self.session_id
            )
            raise e
    
    def _search_with_filters(self, 
                           query: str, 
                           limit: int,
                           filters: Dict[str, Any],
                           return_all: bool = False) -> Union[List[EmbeddedChunkedPost], Dict[str, Any]]:
        """Apply filters to search results"""
        
        # Get larger result set to apply filters
        extended_limit = min(limit * 5, 100)  # Get more results to filter from
        
        results = self.base_retriever.search(query, extended_limit, return_all=True)
        posts = results.get("posts", [])
        
        # Apply filters
        filtered_posts = self._apply_filters(posts, filters)
        
        # Limit results
        filtered_posts = filtered_posts[:limit]
        
        if return_all:
            return {
                **results,
                "posts": filtered_posts,
                "filters_applied": filters,
                "original_count": len(posts),
                "filtered_count": len(filtered_posts)
            }
        
        return filtered_posts
    
    def _apply_filters(self, posts: List[EmbeddedChunkedPost], filters: Dict[str, Any]) -> List[EmbeddedChunkedPost]:
        """Apply various filters to search results"""
        
        filtered_posts = posts
        
        # Author filter
        if "authors" in filters and filters["authors"]:
            author_list = filters["authors"] if isinstance(filters["authors"], list) else [filters["authors"]]
            filtered_posts = [p for p in filtered_posts if p.post_owner in author_list]
        
        # Source filter
        if "sources" in filters and filters["sources"]:
            source_list = filters["sources"] if isinstance(filters["sources"], list) else [filters["sources"]]
            filtered_posts = [p for p in filtered_posts if p.source in source_list]
        
        # Minimum score filter
        if "min_score" in filters and filters["min_score"] is not None:
            min_score = float(filters["min_score"])
            filtered_posts = [p for p in filtered_posts if p.score and p.score >= min_score]
        
        # Text length filter
        if "min_text_length" in filters and filters["min_text_length"]:
            min_length = int(filters["min_text_length"])
            filtered_posts = [p for p in filtered_posts if len(p.text) >= min_length]
        
        if "max_text_length" in filters and filters["max_text_length"]:
            max_length = int(filters["max_text_length"])
            filtered_posts = [p for p in filtered_posts if len(p.text) <= max_length]
        
        # Content type filter (based on keywords)
        if "keywords" in filters and filters["keywords"]:
            keywords = filters["keywords"] if isinstance(filters["keywords"], list) else [filters["keywords"]]
            keywords_lower = [k.lower() for k in keywords]
            
            def contains_keywords(post):
                text_lower = post.text.lower()
                return any(keyword in text_lower for keyword in keywords_lower)
            
            filtered_posts = [p for p in filtered_posts if contains_keywords(p)]
        
        # Exclude keywords filter
        if "exclude_keywords" in filters and filters["exclude_keywords"]:
            exclude_keywords = filters["exclude_keywords"] if isinstance(filters["exclude_keywords"], list) else [filters["exclude_keywords"]]
            exclude_keywords_lower = [k.lower() for k in exclude_keywords]
            
            def excludes_keywords(post):
                text_lower = post.text.lower()
                return not any(keyword in text_lower for keyword in exclude_keywords_lower)
            
            filtered_posts = [p for p in filtered_posts if excludes_keywords(p)]
        
        return filtered_posts
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on popular queries"""
        
        # Get popular queries from analytics
        search_analytics = self.metrics_collector.get_search_analytics(hours_back=168)  # Last week
        top_queries = search_analytics.get("top_queries", [])
        
        # Filter queries that contain the partial query
        suggestions = []
        partial_lower = partial_query.lower()
        
        for query, count in top_queries:
            if partial_lower in query.lower() and query.lower() != partial_lower:
                suggestions.append(query)
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    def get_related_content(self, post: EmbeddedChunkedPost, limit: int = 5) -> List[EmbeddedChunkedPost]:
        """Find content related to a specific post using its embedding"""
        
        if not post.text_embedding:
            return []
        
        try:
            # Use the post's embedding as query vector
            from qdrant_client.http import models
            
            search_request = models.SearchRequest(
                vector=post.text_embedding,
                limit=limit + 1,  # +1 to exclude the original post
                with_payload=True,
                with_vector=True
            )
            
            retrieved_points = self.base_retriever._vector_db_client.search(
                collection_name=self.base_retriever._vector_db_collection,
                **search_request.dict()
            )
            
            # Convert to posts and exclude the original
            related_posts = []
            for point in retrieved_points:
                related_post = EmbeddedChunkedPost.from_retrieved_point(point)
                if related_post.chunk_id != post.chunk_id:  # Exclude the original post
                    related_posts.append(related_post)
            
            return related_posts[:limit]
            
        except Exception as e:
            print(f"Error finding related content: {e}")
            return []
    
    def get_content_by_author(self, author: str, limit: int = 10) -> List[EmbeddedChunkedPost]:
        """Get all content by a specific author"""
        
        try:
            from qdrant_client.http import models
            
            # Create filter for author
            author_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="post_owner",
                        match=models.MatchValue(value=author)
                    )
                ]
            )
            
            # Search with filter
            search_request = models.SearchRequest(
                vector=[0.0] * 384,  # Dummy vector, we're filtering by metadata
                limit=limit,
                with_payload=True,
                with_vector=True,
                filter=author_filter
            )
            
            retrieved_points = self.base_retriever._vector_db_client.search(
                collection_name=self.base_retriever._vector_db_collection,
                **search_request.dict()
            )
            
            return [EmbeddedChunkedPost.from_retrieved_point(point) for point in retrieved_points]
            
        except Exception as e:
            print(f"Error getting content by author: {e}")
            return []
    
    def get_trending_content(self, hours_back: int = 24, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending content based on search hits"""
        
        content_analytics = self.metrics_collector.get_content_analytics()
        top_content = content_analytics.get("top_content", [])
        
        # Return trending content with additional metadata
        trending = []
        for content in top_content[:limit]:
            trending.append({
                "post_id": content["post_id"],
                "author": content["author"],
                "source": content["source"],
                "search_hits": content["search_hits"],
                "text_preview": content.get("text_preview", ""),
                "last_accessed": content.get("last_accessed"),
                "trend_score": content["search_hits"]  # Simple trend score based on hits
            })
        
        return trending
    
    def export_search_history(self, format: str = "json") -> str:
        """Export search history and analytics"""
        return self.metrics_collector.export_analytics(format)
    
    def get_search_analytics_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive search analytics summary"""
        
        search_analytics = self.metrics_collector.get_search_analytics(hours_back)
        content_analytics = self.metrics_collector.get_content_analytics()
        system_analytics = self.metrics_collector.get_system_analytics()
        
        return {
            "search_analytics": search_analytics,
            "content_analytics": content_analytics,
            "system_analytics": system_analytics,
            "session_id": self.session_id,
            "generated_at": datetime.now().isoformat()
        }