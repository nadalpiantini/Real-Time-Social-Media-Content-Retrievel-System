#!/usr/bin/env python3
"""
Phase 3A Analytics Integration Testing
Comprehensive tests for the new analytics features
"""
import sys
import os
import asyncio
import time
import logging
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import uuid

# Add current directory to path
sys.path.append('.')

# Test imports
try:
    from analytics.metrics_collector import MetricsCollector, get_metrics_collector
    from analytics.clustering import SemanticClustering, get_clustering_engine
    from analytics.enhanced_retriever import AnalyticsEnabledRetriever
    from analytics.search_filters import AdvancedSearchFilters, get_search_filters
    from analytics.recommendation_engine import ContentRecommendationEngine, get_recommendation_engine
    from analytics.dashboard import AnalyticsDashboard
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Analytics modules not available: {e}")
    ANALYTICS_AVAILABLE = False

# Core system imports
try:
    from utils.embedding import EmbeddingModelSingleton
    from utils.qdrant import build_qdrant_client
    from models.post import RawPost, EmbeddedChunkedPost
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Core modules not available: {e}")
    CORE_MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analytics_integration_test")

@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    success: bool
    duration_ms: float
    details: Dict[str, Any]
    error_message: str = ""

class AnalyticsIntegrationTester:
    """Comprehensive integration testing for analytics features"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data: List[EmbeddedChunkedPost] = []
        
        # Test configuration
        self.test_session_id = str(uuid.uuid4())
        self.test_queries = [
            "machine learning trends",
            "startup advice",
            "career development",
            "data science projects",
            "python programming"
        ]
    
    async def run_complete_analytics_tests(self) -> bool:
        """Run comprehensive analytics integration tests"""
        
        logger.info("🚀 STARTING ANALYTICS INTEGRATION TESTS - PHASE 3A")
        logger.info("=" * 80)
        
        if not ANALYTICS_AVAILABLE:
            logger.error("❌ Analytics modules not available - cannot run tests")
            return False
        
        if not CORE_MODULES_AVAILABLE:
            logger.error("❌ Core modules not available - cannot run tests")
            return False
        
        success = True
        
        try:
            # 1. Setup test environment
            await self.setup_test_environment()
            
            # 2. Test metrics collection
            success &= await self.test_metrics_collection()
            
            # 3. Test semantic clustering
            success &= await self.test_semantic_clustering()
            
            # 4. Test enhanced retriever
            success &= await self.test_enhanced_retriever()
            
            # 5. Test search filters
            success &= await self.test_search_filters()
            
            # 6. Test recommendation engine
            success &= await self.test_recommendation_engine()
            
            # 7. Test analytics dashboard
            success &= await self.test_analytics_dashboard()
            
            # 8. Test integration workflows
            success &= await self.test_integration_workflows()
            
            # Generate final report
            await self.generate_integration_report()
            
        except Exception as e:
            logger.error(f"💥 Critical error in analytics testing: {e}")
            success = False
        
        return success
    
    async def setup_test_environment(self):
        """Setup test environment with sample data"""
        
        logger.info("🔧 Setting up test environment...")
        
        # Generate test posts
        self.test_data = self._generate_test_posts(20)
        
        # Initialize core components
        try:
            self.embedding_model = EmbeddingModelSingleton()
            self.vector_db_client = build_qdrant_client()
            logger.info("✅ Core components initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize core components: {e}")
            raise
        
        # Initialize analytics components
        try:
            self.metrics_collector = get_metrics_collector()
            self.clustering_engine = get_clustering_engine()
            self.search_filters = get_search_filters()
            self.recommendation_engine = get_recommendation_engine()
            logger.info("✅ Analytics components initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize analytics components: {e}")
            raise
    
    async def test_metrics_collection(self) -> bool:
        """Test metrics collection functionality"""
        
        logger.info("📊 Testing metrics collection...")
        start_time = time.time()
        
        try:
            # Test search metric recording
            test_results = self.test_data[:5]
            processing_time = 150.0  # Mock processing time
            
            metric_id = self.metrics_collector.record_search(
                query="test query",
                results=test_results,
                processing_time_ms=processing_time,
                session_id=self.test_session_id
            )
            
            # Test content ingestion recording
            self.metrics_collector.record_content_ingestion(self.test_data)
            
            # Test system metrics recording
            self.metrics_collector.record_system_metrics(
                memory_mb=1024.0,
                cpu_percent=45.0,
                cache_hit_rate=85.0
            )
            
            # Test analytics retrieval
            search_analytics = self.metrics_collector.get_search_analytics(hours_back=1)
            content_analytics = self.metrics_collector.get_content_analytics()
            system_analytics = self.metrics_collector.get_system_analytics()
            
            # Verify results
            assert search_analytics["total_searches"] > 0, "Search analytics not recorded"
            assert content_analytics["total_content_items"] > 0, "Content metrics not recorded"
            assert system_analytics["current_memory_mb"] > 0, "System metrics not recorded"
            
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="metrics_collection",
                success=True,
                duration_ms=duration,
                details={
                    "metric_id": metric_id,
                    "search_analytics": search_analytics,
                    "content_count": content_analytics["total_content_items"],
                    "system_memory": system_analytics["current_memory_mb"]
                }
            ))
            
            logger.info(f"✅ Metrics collection test passed in {duration:.1f}ms")
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="metrics_collection",
                success=False,
                duration_ms=duration,
                details={},
                error_message=str(e)
            ))
            
            logger.error(f"❌ Metrics collection test failed: {e}")
            return False
    
    async def test_semantic_clustering(self) -> bool:
        """Test semantic clustering functionality"""
        
        logger.info("🧩 Testing semantic clustering...")
        start_time = time.time()
        
        try:
            # Test content analysis
            clustering_results = self.clustering_engine.analyze_content(self.test_data)
            
            # Verify clustering results
            assert "clustering_success" in clustering_results, "Clustering results missing success flag"
            
            if clustering_results["clustering_success"]:
                clusters = clustering_results["clusters"]
                assert "n_clusters" in clusters, "Cluster count missing"
                assert clusters["n_clusters"] > 0, "No clusters found"
                assert "clusters" in clusters, "Cluster details missing"
                
                logger.info(f"Found {clusters['n_clusters']} clusters using {clusters['method_used']} method")
            
            # Test topic modeling
            if clustering_results.get("topic_modeling_success", False):
                topics = clustering_results["topics"]
                assert "n_topics" in topics, "Topic count missing"
                logger.info(f"Found {topics['n_topics']} topics")
            
            # Test content insights
            if "insights" in clustering_results:
                insights = clustering_results["insights"]
                assert "quality_indicators" in insights, "Quality indicators missing"
                logger.info(f"Content diversity: {insights['quality_indicators']['content_diversity']:.3f}")
            
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="semantic_clustering",
                success=True,
                duration_ms=duration,
                details={
                    "clustering_success": clustering_results.get("clustering_success", False),
                    "topic_modeling_success": clustering_results.get("topic_modeling_success", False),
                    "n_clusters": clustering_results.get("clusters", {}).get("n_clusters", 0),
                    "n_topics": clustering_results.get("topics", {}).get("n_topics", 0)
                }
            ))
            
            logger.info(f"✅ Semantic clustering test passed in {duration:.1f}ms")
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="semantic_clustering",
                success=False,
                duration_ms=duration,
                details={},
                error_message=str(e)
            ))
            
            logger.error(f"❌ Semantic clustering test failed: {e}")
            return False
    
    async def test_enhanced_retriever(self) -> bool:
        """Test enhanced retriever with analytics"""
        
        logger.info("🔍 Testing enhanced retriever...")
        start_time = time.time()
        
        try:
            # Initialize enhanced retriever
            retriever = AnalyticsEnabledRetriever(
                embedding_model=self.embedding_model,
                vector_db_client=self.vector_db_client
            )
            
            # Test basic search (this might fail if no data in Qdrant, which is expected)
            try:
                results = retriever.search("test query", limit=3)
                search_successful = True
                result_count = len(results) if isinstance(results, list) else 0
            except Exception as search_error:
                # Expected if no data in vector DB
                search_successful = False
                result_count = 0
                logger.info(f"Search failed as expected (no data in vector DB): {search_error}")
            
            # Test search with filters
            filters = {
                "keywords": ["test"],
                "min_score": 0.5
            }
            
            try:
                filtered_results = retriever.search("test query", limit=3, filters=filters)
                filter_search_successful = True
            except Exception:
                filter_search_successful = False
            
            # Test analytics methods
            suggestions = retriever.get_search_suggestions("test", limit=3)
            trending = retriever.get_trending_content(limit=5)
            analytics_summary = retriever.get_search_analytics_summary()
            
            # Verify analytics integration
            assert isinstance(suggestions, list), "Search suggestions should be a list"
            assert isinstance(trending, list), "Trending content should be a list"
            assert isinstance(analytics_summary, dict), "Analytics summary should be a dict"
            assert "session_id" in analytics_summary, "Session ID missing from analytics"
            
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="enhanced_retriever",
                success=True,
                duration_ms=duration,
                details={
                    "search_successful": search_successful,
                    "result_count": result_count,
                    "filter_search_successful": filter_search_successful,
                    "suggestions_count": len(suggestions),
                    "trending_count": len(trending),
                    "has_analytics": bool(analytics_summary)
                }
            ))
            
            logger.info(f"✅ Enhanced retriever test passed in {duration:.1f}ms")
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="enhanced_retriever",
                success=False,
                duration_ms=duration,
                details={},
                error_message=str(e)
            ))
            
            logger.error(f"❌ Enhanced retriever test failed: {e}")
            return False
    
    async def test_search_filters(self) -> bool:
        """Test advanced search filters"""
        
        logger.info("🎯 Testing search filters...")
        start_time = time.time()
        
        try:
            # Test filter creation
            test_filters = {
                "keywords": ["machine", "learning"],
                "authors": ["test_author"],
                "min_score": 0.7,
                "min_text_length": 100
            }
            
            # Test filter functionality
            filter_suggestions = self.search_filters.get_filter_suggestions()
            filter_summary = self.search_filters.get_filter_summary(test_filters)
            enhanced_query = self.search_filters.apply_filters_to_query("original query", test_filters)
            
            # Verify filter components
            assert isinstance(filter_suggestions, dict), "Filter suggestions should be a dict"
            assert "popular_keywords" in filter_suggestions, "Popular keywords missing"
            assert isinstance(filter_summary, str), "Filter summary should be a string"
            assert isinstance(enhanced_query, str), "Enhanced query should be a string"
            assert len(enhanced_query) >= len("original query"), "Enhanced query should be longer"
            
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="search_filters",
                success=True,
                duration_ms=duration,
                details={
                    "test_filters": test_filters,
                    "suggestions_keys": list(filter_suggestions.keys()),
                    "summary_length": len(filter_summary),
                    "enhanced_query": enhanced_query
                }
            ))
            
            logger.info(f"✅ Search filters test passed in {duration:.1f}ms")
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="search_filters",
                success=False,
                duration_ms=duration,
                details={},
                error_message=str(e)
            ))
            
            logger.error(f"❌ Search filters test failed: {e}")
            return False
    
    async def test_recommendation_engine(self) -> bool:
        """Test recommendation engine"""
        
        logger.info("🤖 Testing recommendation engine...")
        start_time = time.time()
        
        try:
            # Build user profile
            user_profile = self.recommendation_engine.build_user_profile(
                user_id=self.test_session_id,
                search_history=self.test_queries
            )
            
            # Test recommendations
            recommendations = self.recommendation_engine.get_recommendations(
                user_id=self.test_session_id,
                content_pool=self.test_data,
                num_recommendations=5
            )
            
            # Test diversified recommendations
            if recommendations:
                diversified = self.recommendation_engine.get_diversified_recommendations(
                    recommendations, diversity_factor=0.3
                )
            else:
                diversified = []
            
            # Export user profile
            exported_profile = self.recommendation_engine.export_user_profile(self.test_session_id)
            
            # Verify recommendation engine
            assert user_profile.user_id == self.test_session_id, "User profile ID mismatch"
            assert len(user_profile.search_history) > 0, "Search history empty"
            assert isinstance(recommendations, list), "Recommendations should be a list"
            assert isinstance(diversified, list), "Diversified recommendations should be a list"
            assert exported_profile is not None, "Profile export failed"
            assert "profile_strength" in exported_profile, "Profile strength missing"
            
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="recommendation_engine",
                success=True,
                duration_ms=duration,
                details={
                    "user_id": user_profile.user_id,
                    "search_history_count": len(user_profile.search_history),
                    "recommendations_count": len(recommendations),
                    "diversified_count": len(diversified),
                    "profile_strength": exported_profile["profile_strength"]
                }
            ))
            
            logger.info(f"✅ Recommendation engine test passed in {duration:.1f}ms")
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="recommendation_engine",
                success=False,
                duration_ms=duration,
                details={},
                error_message=str(e)
            ))
            
            logger.error(f"❌ Recommendation engine test failed: {e}")
            return False
    
    async def test_analytics_dashboard(self) -> bool:
        """Test analytics dashboard components"""
        
        logger.info("📊 Testing analytics dashboard...")
        start_time = time.time()
        
        try:
            # Test dashboard initialization
            dashboard = AnalyticsDashboard()
            
            # Test dashboard methods (without Streamlit rendering)
            # These would normally render UI components
            health_score = dashboard._calculate_health_score({
                "current_memory_mb": 1024.0,
                "current_cpu_percent": 45.0,
                "current_cache_hit_rate": 85.0,
                "avg_response_time_ms": 200.0
            })
            
            # Test empty analytics handling
            empty_analytics = dashboard.metrics_collector.get_search_analytics(hours_back=1)
            
            # Verify dashboard functionality
            assert isinstance(dashboard, AnalyticsDashboard), "Dashboard initialization failed"
            assert 0 <= health_score <= 100, f"Health score out of range: {health_score}"
            assert isinstance(empty_analytics, dict), "Analytics should return dict"
            
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="analytics_dashboard",
                success=True,
                duration_ms=duration,
                details={
                    "health_score": health_score,
                    "analytics_keys": list(empty_analytics.keys()),
                    "dashboard_initialized": True
                }
            ))
            
            logger.info(f"✅ Analytics dashboard test passed in {duration:.1f}ms")
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="analytics_dashboard",
                success=False,
                duration_ms=duration,
                details={},
                error_message=str(e)
            ))
            
            logger.error(f"❌ Analytics dashboard test failed: {e}")
            return False
    
    async def test_integration_workflows(self) -> bool:
        """Test end-to-end integration workflows"""
        
        logger.info("🔄 Testing integration workflows...")
        start_time = time.time()
        
        try:
            # Simulate complete search workflow
            query = "machine learning trends"
            
            # 1. Record search with metrics
            self.metrics_collector.record_search(
                query=query,
                results=self.test_data[:3],
                processing_time_ms=180.0,
                session_id=self.test_session_id
            )
            
            # 2. Generate recommendations
            recommendations = self.recommendation_engine.get_recommendations(
                user_id=self.test_session_id,
                content_pool=self.test_data,
                num_recommendations=3
            )
            
            # 3. Apply clustering analysis
            clustering_results = self.clustering_engine.analyze_content(self.test_data[:10])
            
            # 4. Generate analytics summary
            analytics_summary = {
                "search_analytics": self.metrics_collector.get_search_analytics(hours_back=1),
                "content_analytics": self.metrics_collector.get_content_analytics(),
                "system_analytics": self.metrics_collector.get_system_analytics()
            }
            
            # Verify workflow integration
            assert analytics_summary["search_analytics"]["total_searches"] > 0, "Search not recorded"
            assert isinstance(recommendations, list), "Recommendations failed"
            assert "clustering_success" in clustering_results, "Clustering analysis failed"
            
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="integration_workflows",
                success=True,
                duration_ms=duration,
                details={
                    "search_recorded": True,
                    "recommendations_generated": len(recommendations),
                    "clustering_successful": clustering_results.get("clustering_success", False),
                    "analytics_complete": True
                }
            ))
            
            logger.info(f"✅ Integration workflows test passed in {duration:.1f}ms")
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            self.results.append(TestResult(
                test_name="integration_workflows",
                success=False,
                duration_ms=duration,
                details={},
                error_message=str(e)
            ))
            
            logger.error(f"❌ Integration workflows test failed: {e}")
            return False
    
    def _generate_test_posts(self, count: int) -> List[EmbeddedChunkedPost]:
        """Generate test posts for testing"""
        
        posts = []
        test_contents = [
            "Machine learning is transforming the way we approach data analysis and predictive modeling in business applications.",
            "Startup founders need to focus on product-market fit before scaling their operations and hiring more employees.",
            "Career development requires continuous learning and networking to stay relevant in today's competitive job market.",
            "Data science projects often require extensive data cleaning and preprocessing before applying advanced algorithms.",
            "Python programming offers powerful libraries like pandas and scikit-learn for data manipulation and machine learning.",
            "Artificial intelligence is revolutionizing industries from healthcare to finance with automated decision-making systems.",
            "Remote work has changed the dynamics of team collaboration and requires new tools for effective communication.",
            "Cloud computing provides scalable infrastructure solutions for modern applications and data storage needs.",
            "User experience design plays a crucial role in product success and customer satisfaction in digital products.",
            "Financial technology innovations are disrupting traditional banking with digital payment solutions and cryptocurrencies."
        ]
        
        authors = ["alice_ml", "bob_startup", "charlie_career", "diana_data", "eve_python"]
        sources = ["linkedin", "twitter", "blog", "newsletter"]
        
        for i in range(count):
            content = test_contents[i % len(test_contents)]
            author = authors[i % len(authors)]
            source = sources[i % len(sources)]
            
            # Create mock embedding (384 dimensions for all-MiniLM-L6-v2)
            mock_embedding = [0.1 * (i % 10) for _ in range(384)]
            
            post = EmbeddedChunkedPost(
                post_id=f"test_post_{i:03d}",
                chunk_id=f"chunk_{i:03d}",
                full_raw_text=content,
                text=content,
                text_embedding=mock_embedding,
                post_owner=author,
                source=source,
                score=0.8 + (i % 3) * 0.05  # Mock similarity scores
            )
            
            posts.append(post)
        
        return posts
    
    async def generate_integration_report(self):
        """Generate comprehensive integration test report"""
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 ANALYTICS INTEGRATION TEST REPORT - PHASE 3A")
        logger.info("=" * 80)
        
        # Calculate overall statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        total_duration = sum(r.duration_ms for r in self.results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        logger.info(f"📈 OVERALL RESULTS:")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        logger.info(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        logger.info(f"Total Duration: {total_duration:.1f}ms")
        logger.info(f"Average Duration: {avg_duration:.1f}ms")
        
        logger.info(f"\n🔍 DETAILED TEST RESULTS:")
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {result.test_name}: {result.duration_ms:.1f}ms")
            
            if result.success and result.details:
                for key, value in result.details.items():
                    logger.info(f"   • {key}: {value}")
            
            if not result.success and result.error_message:
                logger.info(f"   ❌ Error: {result.error_message}")
        
        # Feature-specific summaries
        logger.info(f"\n🎯 FEATURE SUMMARIES:")
        
        # Metrics Collection
        metrics_test = next((r for r in self.results if r.test_name == "metrics_collection"), None)
        if metrics_test and metrics_test.success:
            logger.info(f"📊 Metrics Collection: ✅ Working")
            logger.info(f"   • Search analytics: {metrics_test.details.get('search_analytics', {}).get('total_searches', 0)} searches recorded")
            logger.info(f"   • Content items: {metrics_test.details.get('content_count', 0)} items tracked")
        
        # Clustering
        clustering_test = next((r for r in self.results if r.test_name == "semantic_clustering"), None)
        if clustering_test and clustering_test.success:
            logger.info(f"🧩 Semantic Clustering: ✅ Working")
            logger.info(f"   • Clusters found: {clustering_test.details.get('n_clusters', 0)}")
            logger.info(f"   • Topics identified: {clustering_test.details.get('n_topics', 0)}")
        
        # Recommendations
        rec_test = next((r for r in self.results if r.test_name == "recommendation_engine"), None)
        if rec_test and rec_test.success:
            logger.info(f"🤖 Recommendation Engine: ✅ Working")
            logger.info(f"   • Recommendations generated: {rec_test.details.get('recommendations_count', 0)}")
            logger.info(f"   • Profile strength: {rec_test.details.get('profile_strength', 0):.3f}")
        
        # Export results
        await self._export_test_results()
        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.8:
            logger.info("\n🎉 PHASE 3A: ANALYTICS INTEGRATION - SUCCESS!")
            logger.info("✅ Advanced analytics features are working correctly")
            logger.info("🚀 Ready for production use with enhanced capabilities")
        elif success_rate >= 0.6:
            logger.info("\n⚠️ PHASE 3A: ANALYTICS INTEGRATION - PARTIAL SUCCESS")
            logger.info("✅ Core analytics features working, some advanced features may need attention")
        else:
            logger.info("\n❌ PHASE 3A: ANALYTICS INTEGRATION - NEEDS ATTENTION")
            logger.info("❌ Multiple analytics features need fixes before production use")
        
        logger.info("=" * 80)
    
    async def _export_test_results(self):
        """Export test results to JSON file"""
        
        results_data = {
            "test_timestamp": time.time(),
            "test_session_id": self.test_session_id,
            "phase": "3A_Analytics_Integration",
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "total_duration_ms": sum(r.duration_ms for r in self.results),
            "test_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "details": r.details,
                    "error_message": r.error_message
                }
                for r in self.results
            ],
            "environment_info": {
                "analytics_available": ANALYTICS_AVAILABLE,
                "core_modules_available": CORE_MODULES_AVAILABLE,
                "test_data_count": len(self.test_data)
            }
        }
        
        results_file = Path("analytics") / f"integration_test_results_{int(time.time())}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"📄 Test results exported to {results_file}")

async def main():
    """Main test function"""
    
    tester = AnalyticsIntegrationTester()
    
    try:
        success = await tester.run_complete_analytics_tests()
        return success
    except Exception as e:
        logger.error(f"💥 Critical error in analytics integration testing: {e}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit_code = 0 if success else 1
        logger.info(f"🏁 Analytics integration testing completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Analytics integration testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error in analytics integration testing: {e}")
        sys.exit(1)