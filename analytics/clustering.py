"""
Semantic Clustering and Topic Modeling
Advanced content organization using machine learning techniques
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import hashlib

# Import clustering and topic modeling libraries
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from models.post import EmbeddedChunkedPost


@dataclass
class ContentCluster:
    """Represents a semantic content cluster"""
    cluster_id: str
    label: str
    description: str
    keywords: List[str]
    post_count: int
    chunk_ids: List[str]
    centroid: Optional[List[float]]
    coherence_score: float
    created_at: datetime
    updated_at: datetime


@dataclass
class TopicModel:
    """Topic modeling results"""
    topic_id: str
    label: str
    keywords: List[Tuple[str, float]]  # (word, weight)
    posts: List[str]  # post_ids
    coherence: float
    prevalence: float  # percentage of total content


class SemanticClustering:
    """Advanced semantic clustering and topic modeling system"""
    
    def __init__(self, 
                 min_cluster_size: int = 3,
                 max_clusters: int = 20,
                 random_state: int = 42):
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for clustering. Install with: pip install scikit-learn")
        
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.random_state = random_state
        
        # Storage
        self.clusters: Dict[str, ContentCluster] = {}
        self.topics: Dict[str, TopicModel] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Models
        self.tfidf_vectorizer = None
        self.cluster_model = None
        self.dimensionality_reducer = None
        
    def analyze_content(self, posts: List[EmbeddedChunkedPost]) -> Dict[str, Any]:
        """Perform comprehensive content analysis with clustering and topic modeling"""
        
        if len(posts) < self.min_cluster_size:
            return {"error": f"Need at least {self.min_cluster_size} posts for analysis"}
        
        # Prepare data
        embeddings = np.array([post.text_embedding for post in posts])
        texts = [post.text for post in posts]
        post_ids = [post.post_id for post in posts]
        chunk_ids = [post.chunk_id for post in posts]
        
        # Cache embeddings
        for i, post in enumerate(posts):
            self.embeddings_cache[post.chunk_id] = embeddings[i]
        
        results = {}
        
        # 1. Perform clustering
        try:
            clusters = self._perform_clustering(embeddings, texts, chunk_ids, post_ids)
            results["clusters"] = clusters
            results["clustering_success"] = True
        except Exception as e:
            results["clustering_error"] = str(e)
            results["clustering_success"] = False
        
        # 2. Topic modeling
        try:
            topics = self._perform_topic_modeling(texts, post_ids)
            results["topics"] = topics
            results["topic_modeling_success"] = True
        except Exception as e:
            results["topic_modeling_error"] = str(e)
            results["topic_modeling_success"] = False
        
        # 3. Content insights
        try:
            insights = self._generate_content_insights(posts, embeddings)
            results["insights"] = insights
        except Exception as e:
            results["insights_error"] = str(e)
        
        return results
    
    def _perform_clustering(self, 
                          embeddings: np.ndarray, 
                          texts: List[str],
                          chunk_ids: List[str],
                          post_ids: List[str]) -> Dict[str, Any]:
        """Perform semantic clustering on embeddings"""
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(embeddings)
        
        # Perform multiple clustering algorithms and pick best
        clustering_results = {}
        
        # 1. K-Means clustering
        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            kmeans_labels = kmeans.fit_predict(embeddings)
            kmeans_score = silhouette_score(embeddings, kmeans_labels) if len(set(kmeans_labels)) > 1 else 0
            
            clustering_results["kmeans"] = {
                "labels": kmeans_labels,
                "score": kmeans_score,
                "centroids": kmeans.cluster_centers_
            }
        except Exception as e:
            clustering_results["kmeans_error"] = str(e)
        
        # 2. DBSCAN clustering (density-based)
        try:
            # Estimate eps parameter
            eps = self._estimate_dbscan_eps(embeddings)
            dbscan = DBSCAN(eps=eps, min_samples=self.min_cluster_size)
            dbscan_labels = dbscan.fit_predict(embeddings)
            
            if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
                dbscan_score = silhouette_score(embeddings, dbscan_labels)
            else:
                dbscan_score = -1  # Penalize if all noise or single cluster
            
            clustering_results["dbscan"] = {
                "labels": dbscan_labels,
                "score": dbscan_score,
                "n_clusters": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            }
        except Exception as e:
            clustering_results["dbscan_error"] = str(e)
        
        # 3. Hierarchical clustering
        try:
            hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
            hierarchical_labels = hierarchical.fit_predict(embeddings)
            hierarchical_score = silhouette_score(embeddings, hierarchical_labels) if len(set(hierarchical_labels)) > 1 else 0
            
            clustering_results["hierarchical"] = {
                "labels": hierarchical_labels,
                "score": hierarchical_score
            }
        except Exception as e:
            clustering_results["hierarchical_error"] = str(e)
        
        # Select best clustering method
        best_method = "kmeans"  # default
        best_score = -1
        
        for method, result in clustering_results.items():
            if isinstance(result, dict) and "score" in result and result["score"] > best_score:
                best_method = method
                best_score = result["score"]
        
        best_labels = clustering_results[best_method]["labels"]
        
        # Generate cluster descriptions
        clusters = self._generate_cluster_descriptions(
            best_labels, texts, chunk_ids, post_ids, best_method, embeddings
        )
        
        return {
            "method_used": best_method,
            "silhouette_score": best_score,
            "n_clusters": len(set(best_labels)),
            "clusters": clusters,
            "all_results": clustering_results
        }
    
    def _perform_topic_modeling(self, texts: List[str], post_ids: List[str]) -> Dict[str, Any]:
        """Perform topic modeling using TF-IDF and clustering"""
        
        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,  # Limit features for efficiency
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.8  # Exclude words that appear in >80% of documents
        )
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        except ValueError:
            # Not enough vocabulary
            return {"error": "Insufficient vocabulary for topic modeling"}
        
        # Reduce dimensionality for topic clustering
        if tfidf_matrix.shape[1] > 50:
            svd = TruncatedSVD(n_components=min(50, tfidf_matrix.shape[0]-1), random_state=self.random_state)
            tfidf_reduced = svd.fit_transform(tfidf_matrix)
        else:
            tfidf_reduced = tfidf_matrix.toarray()
        
        # Cluster TF-IDF vectors to create topics
        n_topics = min(8, max(2, len(texts) // 5))  # Reasonable topic count
        
        kmeans_topics = KMeans(n_clusters=n_topics, random_state=self.random_state, n_init=10)
        topic_labels = kmeans_topics.fit_predict(tfidf_reduced)
        
        # Generate topic descriptions
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_id in range(n_topics):
            # Get posts in this topic
            topic_posts = [post_ids[i] for i, label in enumerate(topic_labels) if label == topic_id]
            topic_texts = [texts[i] for i, label in enumerate(topic_labels) if label == topic_id]
            
            if not topic_posts:
                continue
            
            # Calculate topic keywords from centroid
            centroid = kmeans_topics.cluster_centers_[topic_id]
            
            # Get top keywords
            if len(centroid) == len(feature_names):
                top_indices = centroid.argsort()[-10:][::-1]  # Top 10
                keywords = [(feature_names[i], float(centroid[i])) for i in top_indices if centroid[i] > 0]
            else:
                # Fallback: use most common words in topic
                topic_text = " ".join(topic_texts)
                words = topic_text.lower().split()
                word_counts = Counter(words)
                keywords = [(word, count) for word, count in word_counts.most_common(10)]
            
            # Generate topic label
            if keywords:
                topic_label = f"Topic {topic_id + 1}: {', '.join([kw[0] for kw in keywords[:3]])}"
            else:
                topic_label = f"Topic {topic_id + 1}"
            
            # Calculate coherence (simplified)
            coherence = len(set(topic_texts)) / len(topic_texts) if topic_texts else 0
            prevalence = len(topic_posts) / len(post_ids)
            
            topic_model = TopicModel(
                topic_id=f"topic_{topic_id}",
                label=topic_label,
                keywords=keywords,
                posts=topic_posts,
                coherence=coherence,
                prevalence=prevalence
            )
            
            topics[f"topic_{topic_id}"] = topic_model
        
        return {
            "n_topics": n_topics,
            "topics": {tid: {
                "label": topic.label,
                "keywords": topic.keywords[:5],  # Top 5 for display
                "post_count": len(topic.posts),
                "coherence": round(topic.coherence, 3),
                "prevalence": round(topic.prevalence * 100, 1)
            } for tid, topic in topics.items()},
            "topic_assignments": {post_ids[i]: f"topic_{label}" for i, label in enumerate(topic_labels)}
        }
    
    def _generate_content_insights(self, posts: List[EmbeddedChunkedPost], embeddings: np.ndarray) -> Dict[str, Any]:
        """Generate high-level content insights"""
        
        # Calculate content diversity using embedding distances
        if len(embeddings) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(embeddings, metric='cosine')
            avg_distance = np.mean(distances)
            diversity_score = min(1.0, avg_distance * 2)  # Normalize to 0-1
        else:
            diversity_score = 0.0
        
        # Author analysis
        authors = [post.post_owner for post in posts]
        author_counts = Counter(authors)
        top_authors = author_counts.most_common(5)
        
        # Source analysis
        sources = [post.source for post in posts]
        source_counts = Counter(sources)
        
        # Text length analysis
        text_lengths = [len(post.text) for post in posts]
        avg_length = sum(text_lengths) / len(text_lengths)
        
        # Content quality indicators
        quality_indicators = {
            "avg_text_length": round(avg_length),
            "content_diversity": round(diversity_score, 3),
            "author_diversity": len(author_counts),
            "source_diversity": len(source_counts),
            "total_posts": len(posts)
        }
        
        return {
            "quality_indicators": quality_indicators,
            "top_authors": top_authors,
            "source_distribution": dict(source_counts),
            "text_length_stats": {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "avg": round(avg_length),
                "median": round(np.median(text_lengths))
            }
        }
    
    def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method"""
        
        max_k = min(self.max_clusters, len(embeddings) - 1, 15)  # Reasonable upper bound
        min_k = max(2, self.min_cluster_size)
        
        if max_k <= min_k:
            return min_k
        
        inertias = []
        k_range = range(min_k, max_k + 1)
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
            except:
                inertias.append(float('inf'))
        
        # Simple elbow detection
        if len(inertias) >= 3:
            # Find the point with maximum rate of change decrease
            diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            if diffs:
                max_diff_idx = diffs.index(max(diffs))
                optimal_k = k_range[max_diff_idx]
            else:
                optimal_k = min_k
        else:
            optimal_k = min_k
        
        return min(optimal_k, max_k)
    
    def _estimate_dbscan_eps(self, embeddings: np.ndarray) -> float:
        """Estimate epsilon parameter for DBSCAN"""
        from sklearn.neighbors import NearestNeighbors
        
        # Use k = min_samples for k-distance graph
        k = self.min_cluster_size
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        
        # Sort k-distances
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)
        
        # Use knee point (simplified)
        # Take the distance at 90th percentile as a heuristic
        eps = np.percentile(k_distances, 90)
        
        return max(0.1, min(2.0, eps))  # Reasonable bounds
    
    def _generate_cluster_descriptions(self, 
                                     labels: np.ndarray,
                                     texts: List[str],
                                     chunk_ids: List[str],
                                     post_ids: List[str],
                                     method: str,
                                     embeddings: np.ndarray) -> Dict[str, Dict]:
        """Generate descriptions for each cluster"""
        
        clusters = {}
        unique_labels = set(labels)
        
        # Remove noise label if present
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for cluster_id in unique_labels:
            # Get items in this cluster
            cluster_mask = labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            cluster_chunk_ids = [chunk_ids[i] for i in range(len(chunk_ids)) if cluster_mask[i]]
            cluster_post_ids = [post_ids[i] for i in range(len(post_ids)) if cluster_mask[i]]
            
            if not cluster_texts:
                continue
            
            # Generate keywords using TF-IDF on cluster texts
            try:
                cluster_vectorizer = TfidfVectorizer(
                    max_features=10,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )
                cluster_tfidf = cluster_vectorizer.fit_transform(cluster_texts)
                feature_names = cluster_vectorizer.get_feature_names_out()
                
                # Get average TF-IDF scores
                mean_scores = np.mean(cluster_tfidf.toarray(), axis=0)
                top_indices = mean_scores.argsort()[-5:][::-1]  # Top 5
                keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            except:
                # Fallback: use most common words
                all_words = " ".join(cluster_texts).lower().split()
                word_counts = Counter(all_words)
                keywords = [word for word, _ in word_counts.most_common(5)]
            
            # Generate cluster label
            if keywords:
                label = f"Cluster {cluster_id}: {', '.join(keywords[:2])}"
            else:
                label = f"Cluster {cluster_id}"
            
            # Calculate centroid if available
            centroid = None
            if method == "kmeans" and hasattr(self, 'cluster_model'):
                try:
                    cluster_embeddings = embeddings[cluster_mask]
                    centroid = np.mean(cluster_embeddings, axis=0).tolist()
                except:
                    pass
            
            # Calculate coherence (simplified as average text similarity)
            try:
                if len(cluster_texts) > 1:
                    cluster_embeddings = embeddings[cluster_mask]
                    from scipy.spatial.distance import pdist
                    distances = pdist(cluster_embeddings, metric='cosine')
                    coherence_score = 1 - np.mean(distances)  # Higher is better
                else:
                    coherence_score = 1.0
            except:
                coherence_score = 0.5
            
            cluster_info = ContentCluster(
                cluster_id=f"cluster_{cluster_id}",
                label=label,
                description=f"Content cluster with {len(cluster_texts)} items focusing on {', '.join(keywords[:3])}",
                keywords=keywords,
                post_count=len(cluster_texts),
                chunk_ids=cluster_chunk_ids,
                centroid=centroid,
                coherence_score=round(coherence_score, 3),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            clusters[f"cluster_{cluster_id}"] = {
                "id": cluster_info.cluster_id,
                "label": cluster_info.label,
                "description": cluster_info.description,
                "keywords": cluster_info.keywords,
                "post_count": cluster_info.post_count,
                "coherence_score": cluster_info.coherence_score,
                "sample_posts": cluster_post_ids[:3]  # Show first 3 as samples
            }
        
        return clusters
    
    def get_cluster_visualization_data(self, posts: List[EmbeddedChunkedPost]) -> Optional[Dict[str, Any]]:
        """Prepare data for cluster visualization"""
        
        if len(posts) < 3:
            return None
        
        embeddings = np.array([post.text_embedding for post in posts])
        
        # Reduce dimensionality for visualization
        try:
            if UMAP_AVAILABLE and len(posts) > 10:
                # UMAP for better visualization
                reducer = umap.UMAP(n_components=2, random_state=self.random_state)
                reduced_embeddings = reducer.fit_transform(embeddings)
            else:
                # Fallback to t-SNE or PCA
                if len(posts) > 50:
                    tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=min(30, len(posts)-1))
                    reduced_embeddings = tsne.fit_transform(embeddings)
                else:
                    pca = PCA(n_components=2, random_state=self.random_state)
                    reduced_embeddings = pca.fit_transform(embeddings)
            
            # Prepare visualization data
            viz_data = {
                "points": [
                    {
                        "x": float(reduced_embeddings[i, 0]),
                        "y": float(reduced_embeddings[i, 1]),
                        "post_id": posts[i].post_id,
                        "author": posts[i].post_owner,
                        "source": posts[i].source,
                        "text_preview": posts[i].text[:100] + "..." if len(posts[i].text) > 100 else posts[i].text
                    }
                    for i in range(len(posts))
                ],
                "method": "UMAP" if UMAP_AVAILABLE and len(posts) > 10 else ("t-SNE" if len(posts) > 50 else "PCA")
            }
            
            return viz_data
            
        except Exception as e:
            return {"error": f"Visualization failed: {str(e)}"}


# Global clustering instance
_clustering_engine = None

def get_clustering_engine() -> SemanticClustering:
    """Get global clustering engine instance"""
    global _clustering_engine
    if _clustering_engine is None:
        _clustering_engine = SemanticClustering()
    return _clustering_engine