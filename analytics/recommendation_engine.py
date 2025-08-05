"""
Content Recommendation Engine
Intelligent content recommendations based on user preferences, behavior, and content similarity
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import hashlib

from models.post import EmbeddedChunkedPost
from analytics.metrics_collector import get_metrics_collector
from analytics.clustering import get_clustering_engine


@dataclass
class UserProfile:
    """User profile for personalized recommendations"""
    user_id: str
    search_history: List[str]
    preferred_authors: List[str]
    preferred_sources: List[str]
    interest_keywords: List[str]
    interaction_patterns: Dict[str, float]
    created_at: datetime
    updated_at: datetime


@dataclass
class RecommendationItem:
    """Individual recommendation item"""
    post: EmbeddedChunkedPost
    score: float
    reason: str
    recommendation_type: str
    confidence: float


class ContentRecommendationEngine:
    """Advanced content recommendation system"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.metrics_collector = get_metrics_collector()
        self.clustering_engine = get_clustering_engine()
        
        # User profiles storage (in-memory for this implementation)
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Recommendation weights
        self.weights = {
            "content_similarity": 0.3,
            "author_preference": 0.2,
            "source_preference": 0.15,
            "popularity": 0.15,
            "keyword_match": 0.1,
            "recency": 0.1
        }
    
    def build_user_profile(self, user_id: str, search_history: List[str] = None) -> UserProfile:
        """Build or update user profile based on interaction history"""
        
        if search_history is None:
            # Use analytics data if available
            search_analytics = self.metrics_collector.get_search_analytics(hours_back=168)
            search_history = [query for query, count in search_analytics.get("top_queries", [])]
        
        # Extract preferences from search history
        preferred_authors = self._extract_author_preferences(search_history)
        preferred_sources = self._extract_source_preferences(search_history)
        interest_keywords = self._extract_keyword_interests(search_history)
        interaction_patterns = self._analyze_interaction_patterns(user_id)
        
        # Create or update profile
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.search_history.extend(search_history)
            profile.preferred_authors = preferred_authors
            profile.preferred_sources = preferred_sources
            profile.interest_keywords = interest_keywords
            profile.interaction_patterns = interaction_patterns
            profile.updated_at = datetime.now()
        else:
            profile = UserProfile(
                user_id=user_id,
                search_history=search_history,
                preferred_authors=preferred_authors,
                preferred_sources=preferred_sources,
                interest_keywords=interest_keywords,
                interaction_patterns=interaction_patterns,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        self.user_profiles[user_id] = profile
        return profile
    
    def get_recommendations(self, 
                          user_id: str,
                          content_pool: List[EmbeddedChunkedPost],
                          num_recommendations: int = 10,
                          recommendation_types: List[str] = None) -> List[RecommendationItem]:
        """Generate personalized content recommendations"""
        
        if recommendation_types is None:
            recommendation_types = ["similar_content", "trending", "author_based", "keyword_based"]
        
        # Build/update user profile
        user_profile = self.build_user_profile(user_id)
        
        all_recommendations = []
        
        # Generate different types of recommendations
        for rec_type in recommendation_types:
            if rec_type == "similar_content":
                recs = self._get_similar_content_recommendations(user_profile, content_pool)
            elif rec_type == "trending":
                recs = self._get_trending_recommendations(content_pool)
            elif rec_type == "author_based":
                recs = self._get_author_based_recommendations(user_profile, content_pool)
            elif rec_type == "keyword_based":
                recs = self._get_keyword_based_recommendations(user_profile, content_pool)
            elif rec_type == "cluster_based":
                recs = self._get_cluster_based_recommendations(user_profile, content_pool)
            else:
                recs = []
            
            all_recommendations.extend(recs)
        
        # Remove duplicates and sort by score
        seen_chunks = set()
        unique_recommendations = []
        
        for rec in all_recommendations:
            if rec.post.chunk_id not in seen_chunks:
                seen_chunks.add(rec.post.chunk_id)
                unique_recommendations.append(rec)
        
        # Sort by score and return top N
        unique_recommendations.sort(key=lambda x: x.score, reverse=True)
        return unique_recommendations[:num_recommendations]
    
    def _get_similar_content_recommendations(self, 
                                           user_profile: UserProfile,
                                           content_pool: List[EmbeddedChunkedPost]) -> List[RecommendationItem]:
        """Get recommendations based on content similarity to user interests"""
        
        recommendations = []
        
        if not user_profile.search_history:
            return recommendations
        
        # Calculate user interest vector (average of search query embeddings)
        try:
            # This would ideally use embeddings of search queries
            # For now, use keyword matching and content analysis
            
            user_keywords = set()
            for query in user_profile.search_history[-10:]:  # Last 10 searches
                user_keywords.update(query.lower().split())
            
            for post in content_pool:
                # Calculate similarity based on keyword overlap
                post_keywords = set(post.text.lower().split())
                keyword_overlap = len(user_keywords.intersection(post_keywords))
                
                if keyword_overlap > 0:
                    max_keywords = max(len(user_keywords), len(post_keywords))
                    similarity_score = keyword_overlap / max_keywords if max_keywords > 0 else 0
                    
                    if similarity_score > 0.1:  # Minimum threshold
                        rec = RecommendationItem(
                            post=post,
                            score=similarity_score * self.weights["content_similarity"],
                            reason=f"Similar to your interests ({keyword_overlap} matching keywords)",
                            recommendation_type="similar_content",
                            confidence=similarity_score
                        )
                        recommendations.append(rec)
        
        except Exception as e:
            print(f"Error in similar content recommendations: {e}")
        
        return recommendations
    
    def _get_trending_recommendations(self, content_pool: List[EmbeddedChunkedPost]) -> List[RecommendationItem]:
        """Get trending content recommendations"""
        
        recommendations = []
        
        # Get trending content from analytics
        content_analytics = self.metrics_collector.get_content_analytics()
        top_content = content_analytics.get("top_content", [])
        
        # Create mapping of post_id to search hits
        trending_posts = {item["post_id"]: item["search_hits"] for item in top_content}
        
        for post in content_pool:
            if post.post_id in trending_posts:
                search_hits = trending_posts[post.post_id]
                
                # Normalize trending score
                max_hits = max(trending_posts.values()) if trending_posts else 1
                trending_score = search_hits / max_hits if max_hits > 0 else 0
                
                rec = RecommendationItem(
                    post=post,
                    score=trending_score * self.weights["popularity"],
                    reason=f"Trending content ({search_hits} recent searches)",
                    recommendation_type="trending",
                    confidence=trending_score
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _get_author_based_recommendations(self, 
                                        user_profile: UserProfile,
                                        content_pool: List[EmbeddedChunkedPost]) -> List[RecommendationItem]:
        """Get recommendations based on preferred authors"""
        
        recommendations = []
        
        if not user_profile.preferred_authors:
            return recommendations
        
        # Score posts by author preference
        author_scores = {author: score for author, score in user_profile.preferred_authors}
        
        for post in content_pool:
            if post.post_owner in author_scores:
                author_score = author_scores[post.post_owner]
                
                rec = RecommendationItem(
                    post=post,
                    score=author_score * self.weights["author_preference"],
                    reason=f"Content by {post.post_owner} (preferred author)",
                    recommendation_type="author_based",
                    confidence=author_score
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _get_keyword_based_recommendations(self, 
                                         user_profile: UserProfile,
                                         content_pool: List[EmbeddedChunkedPost]) -> List[RecommendationItem]:
        """Get recommendations based on user's interest keywords"""
        
        recommendations = []
        
        if not user_profile.interest_keywords:
            return recommendations
        
        # Create keyword weights
        keyword_weights = {}
        for keyword, weight in user_profile.interest_keywords:
            keyword_weights[keyword.lower()] = weight
        
        for post in content_pool:
            post_text_lower = post.text.lower()
            keyword_score = 0
            matching_keywords = []
            
            for keyword, weight in keyword_weights.items():
                if keyword in post_text_lower:
                    keyword_score += weight
                    matching_keywords.append(keyword)
            
            if keyword_score > 0:
                # Normalize by post length to avoid bias toward longer posts
                normalized_score = keyword_score / (len(post.text) / 1000)  # Per 1000 chars
                
                rec = RecommendationItem(
                    post=post,
                    score=normalized_score * self.weights["keyword_match"],
                    reason=f"Matches your interests: {', '.join(matching_keywords[:3])}",
                    recommendation_type="keyword_based",
                    confidence=min(1.0, normalized_score)
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _get_cluster_based_recommendations(self, 
                                         user_profile: UserProfile,
                                         content_pool: List[EmbeddedChunkedPost]) -> List[RecommendationItem]:
        """Get recommendations based on content clustering"""
        
        recommendations = []
        
        try:
            # Analyze content clusters
            clustering_results = self.clustering_engine.analyze_content(content_pool)
            
            if not clustering_results.get("clustering_success"):
                return recommendations
            
            clusters = clustering_results.get("clusters", {}).get("clusters", {})
            
            # Find clusters that match user interests
            user_keywords = set()
            for query in user_profile.search_history[-5:]:  # Recent searches
                user_keywords.update(query.lower().split())
            
            # Score clusters by keyword overlap
            cluster_scores = {}
            for cluster_id, cluster_info in clusters.items():
                cluster_keywords = set(kw.lower() for kw in cluster_info.get("keywords", []))
                overlap = len(user_keywords.intersection(cluster_keywords))
                
                if overlap > 0:
                    cluster_scores[cluster_id] = overlap / len(cluster_keywords) if cluster_keywords else 0
            
            # Recommend posts from high-scoring clusters
            for post in content_pool:
                # This is simplified - in a full implementation, we'd track which cluster each post belongs to
                for cluster_id, score in cluster_scores.items():
                    if score > 0.3:  # Threshold for cluster relevance
                        rec = RecommendationItem(
                            post=post,
                            score=score * 0.2,  # Cluster-based weight
                            reason=f"From relevant content cluster",
                            recommendation_type="cluster_based",
                            confidence=score
                        )
                        recommendations.append(rec)
                        break  # Only add once per post
        
        except Exception as e:
            print(f"Error in cluster-based recommendations: {e}")
        
        return recommendations
    
    def _extract_author_preferences(self, search_history: List[str]) -> List[Tuple[str, float]]:
        """Extract author preferences from search analytics"""
        
        search_analytics = self.metrics_collector.get_search_analytics(hours_back=168)
        top_authors = search_analytics.get("top_authors", [])
        
        # Convert to (author, score) tuples with normalized scores
        if top_authors:
            max_count = max(count for _, count in top_authors) if top_authors else 1
            return [(author, count / max_count if max_count > 0 else 0) for author, count in top_authors[:10]]
        
        return []
    
    def _extract_source_preferences(self, search_history: List[str]) -> List[Tuple[str, float]]:
        """Extract source preferences from search analytics"""
        
        search_analytics = self.metrics_collector.get_search_analytics(hours_back=168)
        source_dist = search_analytics.get("source_distribution", {})
        
        if source_dist:
            max_count = max(source_dist.values()) if source_dist else 1
            return [(source, count / max_count if max_count > 0 else 0) for source, count in source_dist.items()]
        
        return []
    
    def _extract_keyword_interests(self, search_history: List[str]) -> List[Tuple[str, float]]:
        """Extract keyword interests from search history"""
        
        # Analyze search queries to extract keywords
        all_words = []
        for query in search_history:
            words = query.lower().split()
            all_words.extend(words)
        
        # Count word frequency
        word_counts = Counter(all_words)
        
        # Filter out common words (simple stopword removal)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_words = {word: count for word, count in word_counts.items() 
                         if word not in stopwords and len(word) > 2}
        
        # Normalize scores
        if filtered_words:
            max_count = max(filtered_words.values())
            return [(word, count / max_count if max_count > 0 else 0) for word, count in 
                   sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20]]
        
        return []
    
    def _analyze_interaction_patterns(self, user_id: str) -> Dict[str, float]:
        """Analyze user interaction patterns"""
        
        # This would analyze click patterns, dwell time, etc.
        # For now, return basic patterns
        return {
            "avg_session_length": 1.0,
            "search_frequency": 1.0,
            "result_engagement": 1.0
        }
    
    def get_recommendation_explanation(self, recommendation: RecommendationItem) -> Dict[str, Any]:
        """Get detailed explanation for a recommendation"""
        
        return {
            "post_id": recommendation.post.post_id,
            "author": recommendation.post.post_owner,
            "source": recommendation.post.source,
            "score": recommendation.score,
            "reason": recommendation.reason,
            "type": recommendation.recommendation_type,
            "confidence": recommendation.confidence,
            "text_preview": recommendation.post.text[:200] + "..." if len(recommendation.post.text) > 200 else recommendation.post.text
        }
    
    def get_diversified_recommendations(self, 
                                     recommendations: List[RecommendationItem],
                                     diversity_factor: float = 0.3) -> List[RecommendationItem]:
        """Apply diversity to recommendations to avoid over-concentration"""
        
        if not recommendations:
            return []
        
        diversified = []
        used_authors = set()
        used_sources = set()
        type_counts = defaultdict(int)
        
        # Sort by score first
        sorted_recs = sorted(recommendations, key=lambda x: x.score, reverse=True)
        
        for rec in sorted_recs:
            # Apply diversity constraints
            author_penalty = 0.2 if rec.post.post_owner in used_authors else 0
            source_penalty = 0.1 if rec.post.source in used_sources else 0
            type_penalty = 0.1 * type_counts[rec.recommendation_type]
            
            # Calculate diversified score
            diversity_penalty = (author_penalty + source_penalty + type_penalty) * diversity_factor
            diversified_score = rec.score * (1 - diversity_penalty)
            
            # Create new recommendation with adjusted score
            diversified_rec = RecommendationItem(
                post=rec.post,
                score=diversified_score,
                reason=rec.reason,
                recommendation_type=rec.recommendation_type,
                confidence=rec.confidence
            )
            
            diversified.append(diversified_rec)
            
            # Update tracking sets
            used_authors.add(rec.post.post_owner)
            used_sources.add(rec.post.source)
            type_counts[rec.recommendation_type] += 1
        
        # Re-sort by diversified score
        return sorted(diversified, key=lambda x: x.score, reverse=True)
    
    def update_user_feedback(self, user_id: str, post_id: str, feedback_type: str, feedback_value: float):
        """Update user profile based on feedback (clicks, likes, etc.)"""
        
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Update interaction patterns based on feedback
        if feedback_type == "click":
            profile.interaction_patterns["result_engagement"] = (
                profile.interaction_patterns.get("result_engagement", 0.5) * 0.9 + feedback_value * 0.1
            )
        elif feedback_type == "dwell_time":
            profile.interaction_patterns["avg_session_length"] = (
                profile.interaction_patterns.get("avg_session_length", 1.0) * 0.9 + feedback_value * 0.1
            )
        
        profile.updated_at = datetime.now()
    
    def export_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export user profile for analysis or backup"""
        
        if user_id not in self.user_profiles:
            return None
        
        profile = self.user_profiles[user_id]
        
        return {
            "user_id": profile.user_id,
            "search_history": profile.search_history[-50:],  # Last 50 searches
            "preferred_authors": profile.preferred_authors,
            "preferred_sources": profile.preferred_sources,
            "interest_keywords": profile.interest_keywords,
            "interaction_patterns": profile.interaction_patterns,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
            "profile_strength": self._calculate_profile_strength(profile)
        }
    
    def _calculate_profile_strength(self, profile: UserProfile) -> float:
        """Calculate strength of user profile for recommendation quality"""
        
        strength = 0.0
        
        # Search history contribution
        strength += min(1.0, len(profile.search_history) / 20) * 0.3
        
        # Author preferences contribution
        strength += min(1.0, len(profile.preferred_authors) / 10) * 0.25
        
        # Keyword interests contribution  
        strength += min(1.0, len(profile.interest_keywords) / 15) * 0.25
        
        # Source preferences contribution
        strength += min(1.0, len(profile.preferred_sources) / 5) * 0.2
        
        return strength


# Global recommendation engine instance
_recommendation_engine = None

def get_recommendation_engine() -> ContentRecommendationEngine:
    """Get global recommendation engine instance"""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = ContentRecommendationEngine()
    return _recommendation_engine