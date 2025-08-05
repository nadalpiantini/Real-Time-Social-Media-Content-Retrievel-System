"""
Analytics Metrics Collector
Collects and stores analytics data for the Real-Time Social Media Content Retrieval System
"""
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict, deque

from models.post import EmbeddedChunkedPost


@dataclass
class SearchMetric:
    """Individual search metric"""
    timestamp: datetime
    query: str
    query_hash: str
    results_count: int
    processing_time_ms: float
    user_session_id: str
    top_score: float
    avg_score: float
    sources: List[str]
    authors: List[str]


@dataclass 
class ContentMetric:
    """Content-related metrics"""
    timestamp: datetime
    post_id: str
    chunk_id: str
    author: str
    source: str
    text_length: int
    search_hits: int = 0
    last_accessed: Optional[datetime] = None
    similarity_scores: List[float] = None
    topics: List[str] = None


@dataclass
class SystemMetric:
    """System performance metrics"""
    timestamp: datetime
    memory_usage_mb: float
    cpu_usage_percent: float
    active_sessions: int
    total_searches: int
    total_content_items: int
    cache_hit_rate: float
    avg_response_time_ms: float


class MetricsCollector:
    """Advanced metrics collection system for analytics"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "analytics/data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics storage
        self.search_metrics: deque = deque(maxlen=10000)  # Last 10K searches
        self.content_metrics: Dict[str, ContentMetric] = {}  # chunk_id -> metric
        self.system_metrics: deque = deque(maxlen=1440)  # 24 hours of minute data
        
        # Real-time aggregations
        self.hourly_search_counts: defaultdict = defaultdict(int)
        self.popular_queries: defaultdict = defaultdict(int)
        self.top_authors: defaultdict = defaultdict(int)
        self.search_response_times: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Session tracking
        self.active_sessions: set = set()
        
        # Load existing data
        self._load_persistent_data()
    
    def record_search(self, 
                     query: str, 
                     results: List[EmbeddedChunkedPost],
                     processing_time_ms: float,
                     session_id: str) -> str:
        """Record a search event and return metric ID"""
        
        with self._lock:
            # Calculate aggregated metrics
            scores = [r.score for r in results if r.score is not None]
            top_score = max(scores) if scores else 0.0
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            sources = list(set(r.source for r in results))
            authors = list(set(r.post_owner for r in results))
            
            # Create search metric
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:12]
            metric = SearchMetric(
                timestamp=datetime.now(),
                query=query,
                query_hash=query_hash,
                results_count=len(results),
                processing_time_ms=processing_time_ms,
                user_session_id=session_id,
                top_score=top_score,
                avg_score=avg_score,
                sources=sources,
                authors=authors
            )
            
            # Store metric
            self.search_metrics.append(metric)
            
            # Update aggregations
            hour_key = metric.timestamp.strftime("%Y-%m-%d-%H")
            self.hourly_search_counts[hour_key] += 1
            self.popular_queries[query.lower()] += 1
            for author in authors:
                self.top_authors[author] += 1
            self.search_response_times.append(processing_time_ms)
            
            # Update content metrics
            for result in results:
                self._update_content_access(result)
            
            # Track session
            self.active_sessions.add(session_id)
            
            return query_hash
    
    def record_content_ingestion(self, posts: List[EmbeddedChunkedPost]):
        """Record content ingestion metrics"""
        
        with self._lock:
            for post in posts:
                if post.chunk_id not in self.content_metrics:
                    self.content_metrics[post.chunk_id] = ContentMetric(
                        timestamp=datetime.now(),
                        post_id=post.post_id,
                        chunk_id=post.chunk_id,
                        author=post.post_owner,
                        source=post.source,
                        text_length=len(post.text),
                        similarity_scores=[],
                        topics=[]
                    )
    
    def record_system_metrics(self, 
                            memory_mb: float,
                            cpu_percent: float,
                            cache_hit_rate: float):
        """Record system performance metrics"""
        
        with self._lock:
            # Calculate current stats
            avg_response_time = (
                sum(self.search_response_times) / len(self.search_response_times)
                if self.search_response_times else 0.0
            )
            
            metric = SystemMetric(
                timestamp=datetime.now(),
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                active_sessions=len(self.active_sessions),
                total_searches=len(self.search_metrics),
                total_content_items=len(self.content_metrics),
                cache_hit_rate=cache_hit_rate,
                avg_response_time_ms=avg_response_time
            )
            
            self.system_metrics.append(metric)
    
    def get_search_analytics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get search analytics for the specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            recent_searches = [
                s for s in self.search_metrics 
                if s.timestamp > cutoff_time
            ]
            
            if not recent_searches:
                return self._empty_analytics()
            
            # Calculate metrics
            total_searches = len(recent_searches)
            avg_response_time = sum(s.processing_time_ms for s in recent_searches) / total_searches
            avg_results_per_search = sum(s.results_count for s in recent_searches) / total_searches
            
            # Top queries (limit to reasonable display)
            query_counts = defaultdict(int)
            for search in recent_searches:
                query_counts[search.query] += 1
            top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Author popularity
            author_counts = defaultdict(int)
            for search in recent_searches:
                for author in search.authors:
                    author_counts[author] += 1
            top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Source distribution
            source_counts = defaultdict(int)
            for search in recent_searches:
                for source in search.sources:
                    source_counts[source] += 1
            
            # Hourly distribution
            hourly_counts = defaultdict(int)
            for search in recent_searches:
                hour_key = search.timestamp.strftime("%H:00")
                hourly_counts[hour_key] += 1
            
            return {
                "total_searches": total_searches,
                "avg_response_time_ms": round(avg_response_time, 2),
                "avg_results_per_search": round(avg_results_per_search, 1),
                "top_queries": top_queries,
                "top_authors": top_authors,
                "source_distribution": dict(source_counts),
                "hourly_distribution": dict(hourly_counts),
                "active_sessions": len(self.active_sessions),
                "time_period_hours": hours_back
            }
    
    def get_content_analytics(self) -> Dict[str, Any]:
        """Get content popularity and engagement analytics"""
        
        with self._lock:
            if not self.content_metrics:
                return {"total_content_items": 0, "top_content": [], "author_stats": {}}
            
            # Sort content by search hits
            popular_content = sorted(
                self.content_metrics.values(),
                key=lambda x: x.search_hits,
                reverse=True
            )[:20]  # Top 20
            
            # Author statistics
            author_stats = defaultdict(lambda: {"posts": 0, "total_hits": 0, "avg_text_length": 0})
            text_lengths = []
            
            for metric in self.content_metrics.values():
                author_stats[metric.author]["posts"] += 1
                author_stats[metric.author]["total_hits"] += metric.search_hits
                author_stats[metric.author]["avg_text_length"] += metric.text_length
                text_lengths.append(metric.text_length)
            
            # Calculate averages
            for author, stats in author_stats.items():
                if stats["posts"] > 0:
                    stats["avg_text_length"] = round(stats["avg_text_length"] / stats["posts"])
                    stats["avg_hits_per_post"] = round(stats["total_hits"] / stats["posts"], 1)
            
            return {
                "total_content_items": len(self.content_metrics),
                "top_content": [
                    {
                        "post_id": c.post_id,
                        "author": c.author,
                        "source": c.source,
                        "search_hits": c.search_hits,
                        "text_preview": c.text_length,
                        "last_accessed": c.last_accessed.isoformat() if c.last_accessed else None
                    }
                    for c in popular_content[:10]
                ],
                "author_stats": dict(author_stats),
                "avg_text_length": round(sum(text_lengths) / len(text_lengths)) if text_lengths else 0,
                "total_search_hits": sum(m.search_hits for m in self.content_metrics.values())
            }
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get system performance analytics"""
        
        with self._lock:
            if not self.system_metrics:
                return {"status": "No system metrics available"}
            
            recent_metrics = list(self.system_metrics)[-60:]  # Last hour
            
            avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
            avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            
            latest = recent_metrics[-1]
            
            return {
                "current_memory_mb": round(latest.memory_usage_mb, 1),
                "current_cpu_percent": round(latest.cpu_usage_percent, 1),
                "current_cache_hit_rate": round(latest.cache_hit_rate, 1),
                "avg_memory_mb": round(avg_memory, 1),
                "avg_cpu_percent": round(avg_cpu, 1),
                "avg_cache_hit_rate": round(avg_cache_hit_rate, 1),
                "active_sessions": latest.active_sessions,
                "total_searches": latest.total_searches,
                "total_content_items": latest.total_content_items,
                "avg_response_time_ms": round(latest.avg_response_time_ms, 2)
            }
    
    def export_analytics(self, format: str = "json") -> str:
        """Export analytics data to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = f"analytics_export_{timestamp}.json"
            filepath = self.storage_path / filename
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "search_analytics": self.get_search_analytics(),
                "content_analytics": self.get_content_analytics(),
                "system_analytics": self.get_system_analytics(),
                "raw_metrics": {
                    "total_searches": len(self.search_metrics),
                    "total_content_items": len(self.content_metrics),
                    "total_system_records": len(self.system_metrics)
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return str(filepath)
        
        raise ValueError(f"Unsupported export format: {format}")
    
    def _update_content_access(self, post: EmbeddedChunkedPost):
        """Update content access metrics"""
        
        if post.chunk_id in self.content_metrics:
            self.content_metrics[post.chunk_id].search_hits += 1
            self.content_metrics[post.chunk_id].last_accessed = datetime.now()
            
            # Track similarity score if available
            if post.score is not None:
                if self.content_metrics[post.chunk_id].similarity_scores is None:
                    self.content_metrics[post.chunk_id].similarity_scores = []
                self.content_metrics[post.chunk_id].similarity_scores.append(post.score)
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure"""
        return {
            "total_searches": 0,
            "avg_response_time_ms": 0,
            "avg_results_per_search": 0,
            "top_queries": [],
            "top_authors": [],
            "source_distribution": {},
            "hourly_distribution": {},
            "active_sessions": 0,
            "time_period_hours": 0
        }
    
    def _load_persistent_data(self):
        """Load persistent analytics data if available"""
        try:
            # Could implement persistent storage here
            # For now, we start fresh each session
            pass
        except Exception:
            # Graceful fallback if loading fails
            pass
    
    def cleanup_old_sessions(self, hours_old: int = 24):
        """Clean up old session tracking data"""
        # This would typically be called periodically
        # For simplicity, we keep sessions in memory for this implementation
        pass


# Global metrics collector instance
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector