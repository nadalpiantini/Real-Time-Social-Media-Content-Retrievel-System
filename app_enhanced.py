"""
Enhanced App with Analytics Integration
Real-Time Social Media Content Retrieval System with Advanced Analytics
"""
import streamlit as st
import time
import os
import json
import signal
import uuid

# FIRST STREAMLIT COMMAND - Must be first!
st.set_page_config(
    page_title="🎯 Real-Time LinkedIn Content Quest",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for analytics
if 'analytics_session_id' not in st.session_state:
    st.session_state.analytics_session_id = str(uuid.uuid4())

# Performance optimizations - import and setup early
try:
    from utils.performance import setup_performance_monitoring, StreamlitContextManager
    perf_optimizer = setup_performance_monitoring()
except ImportError:
    perf_optimizer = None
    StreamlitContextManager = None

# Core imports
try:
    from utils.embedding import EmbeddingModelSingleton, CrossEncoderModelSingleton
    from utils.qdrant import build_qdrant_client
    from utils.retriever import QdrantVectorDBRetriever
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Core modules not available: {e}")
    CORE_MODULES_AVAILABLE = False

# Analytics imports
try:
    from analytics.dashboard import render_analytics_dashboard
    from analytics.enhanced_retriever import AnalyticsEnabledRetriever
    from analytics.search_filters import get_search_filters
    from analytics.recommendation_engine import get_recommendation_engine
    from analytics.metrics_collector import get_metrics_collector
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Analytics features not available: {e}")
    ANALYTICS_AVAILABLE = False

# Optional imports - Bytewax pipeline
try:
    from bytewax.testing import run_main
    from flow import build as build_flow
    BYTEWAX_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Bytewax not available - using simplified mode")
    BYTEWAX_AVAILABLE = False

# Optional imports - LinkedIn scraper
try:
    from linkedin_posts_scrapper import fetch_posts, make_post_data
    SCRAPER_AVAILABLE = True
except ImportError:
    st.warning("⚠️ LinkedIn scraper not available - using existing data only")
    SCRAPER_AVAILABLE = False

# Import original functions from app.py
from app import (
    basic_prerequisites,
    migrate_data_to_vectordb,
    migrate_data_simplified,
    count_posts_in_files
)

def render_enhanced_header():
    """Render enhanced header with analytics toggle"""
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("🎯 Real-Time LinkedIn Content Quest")
        st.markdown("### 🔍 AI-Powered Semantic Search & Analytics")
        st.markdown("**Discover relevant LinkedIn posts with advanced analytics and recommendations**")
    
    with col2:
        if st.button("📊 Analytics", key="analytics_toggle"):
            st.session_state.show_analytics = not st.session_state.get('show_analytics', False)
    
    with col3:
        if ANALYTICS_AVAILABLE:
            st.success("✅ Analytics")
        else:
            st.error("❌ Analytics")

def render_system_info():
    """Render enhanced system information"""
    
    with st.expander("ℹ️ How it works", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔄 Core Process:**
            1. **📥 Fetch Data**: Import LinkedIn posts (via scraper or existing JSON files)
            2. **🔄 Process**: Convert posts to vector embeddings using AI models  
            3. **🔍 Search**: Enter queries to find semantically similar content
            4. **📊 Results**: Get ranked results with similarity scores
            """)
        
        with col2:
            st.markdown("""
            **📈 Analytics Features:**
            1. **📊 Dashboard**: Real-time analytics and visualizations
            2. **🎯 Smart Filters**: Advanced search filtering options
            3. **🤖 Recommendations**: Personalized content suggestions
            4. **🧩 Clustering**: Semantic content organization
            """)

def render_enhanced_sidebar():
    """Render enhanced sidebar with analytics controls"""
    
    st.sidebar.title("⚙️ Configuration")
    
    # Search Configuration
    st.sidebar.subheader("🔍 Search Settings")
    number_of_results = st.sidebar.slider("🎯 Number of results", 1, 20, 5, help="How many relevant posts to show")
    
    # Analytics Controls
    if ANALYTICS_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Analytics")
        
        show_recommendations = st.sidebar.checkbox("🤖 Show Recommendations", value=True)
        show_clustering = st.sidebar.checkbox("🧩 Enable Clustering", value=False)
        enable_filters = st.sidebar.checkbox("🔍 Advanced Filters", value=True)
        
        analytics_settings = {
            "show_recommendations": show_recommendations,
            "show_clustering": show_clustering,
            "enable_filters": enable_filters
        }
    else:
        analytics_settings = {
            "show_recommendations": False,
            "show_clustering": False,
            "enable_filters": False
        }
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ System Status")
    
    status_items = [
        ("📊 Core Modules", CORE_MODULES_AVAILABLE),
        ("📈 Analytics", ANALYTICS_AVAILABLE),
        ("⚡ Bytewax", BYTEWAX_AVAILABLE),
        ("🔍 Scraper", SCRAPER_AVAILABLE)
    ]
    
    for name, available in status_items:
        status = "✅" if available else "❌"
        st.sidebar.write(f"{status} {name}")
    
    return number_of_results, analytics_settings

def get_enhanced_insights_from_posts():
    """Enhanced search interface with analytics integration"""
    
    st.markdown("---")
    st.header("🔍 AI-Powered Content Search")
    
    if not CORE_MODULES_AVAILABLE:
        st.error("❌ Core modules not available. Please check your installation.")
        return
    
    # Initialize models and retriever
    try:
        embedding_model = EmbeddingModelSingleton()
        vector_db_client = build_qdrant_client()
        cross_encoder_model = CrossEncoderModelSingleton()
        
        if ANALYTICS_AVAILABLE:
            retriever = AnalyticsEnabledRetriever(
                embedding_model=embedding_model,
                vector_db_client=vector_db_client,
                cross_encoder_model=cross_encoder_model
            )
            metrics_collector = get_metrics_collector()
            recommendation_engine = get_recommendation_engine()
        else:
            retriever = QdrantVectorDBRetriever(
                embedding_model=embedding_model,
                vector_db_client=vector_db_client,
                cross_encoder_model=cross_encoder_model
            )
        
    except Exception as e:
        st.error(f"❌ Error initializing search components: {str(e)}")
        return
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Enter your search query:",
            placeholder="e.g., machine learning trends, startup advice, career development...",
            key="main_search_query"
        )
    
    with col2:
        search_button = st.button("🔍 **Search**", type="primary", use_container_width=True)
    
    # Advanced filters (if analytics available)
    filters = {}
    if ANALYTICS_AVAILABLE and st.session_state.get('analytics_settings', {}).get('enable_filters', False):
        search_filters = get_search_filters()
        filters = search_filters.render_filters_expander()
        
        if filters:
            st.info(f"🎯 Active filters: {search_filters.get_filter_summary(filters)}")
    
    # Perform search
    if search_button and search_query.strip():
        with st.spinner("🔍 Searching for relevant content..."):
            try:
                start_time = time.time()
                
                # Get number of results from sidebar
                num_results = st.session_state.get('number_of_results', 5)
                
                # Perform search with or without filters
                if ANALYTICS_AVAILABLE and filters:
                    results = retriever.search(search_query, limit=num_results, filters=filters)
                else:
                    results = retriever.search(search_query, limit=num_results)
                
                search_time = (time.time() - start_time) * 1000
                
                # Store results in session state for analytics
                if isinstance(results, list):
                    st.session_state.current_search_results = results
                    st.session_state.current_search_query = search_query
                
                # Display results
                if results:
                    st.success(f"✅ Found {len(results)} relevant posts in {search_time:.0f}ms")
                    display_enhanced_search_results(results, search_query)
                    
                    # Show recommendations if analytics available
                    if ANALYTICS_AVAILABLE and st.session_state.get('analytics_settings', {}).get('show_recommendations', False):
                        show_recommendations(results, search_query)
                    
                else:
                    st.warning("🤔 No relevant posts found. Try different keywords or adjust your filters.")
                
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
    
    # Show recent searches and trending content
    if ANALYTICS_AVAILABLE:
        render_recent_activity()

def display_enhanced_search_results(results, query):
    """Display search results with enhanced analytics features"""
    
    st.subheader(f"📋 Search Results for: '{query}'")
    
    # Results summary
    if results:
        avg_score = sum(r.score for r in results if r.score) / len(results)
        st.info(f"📊 Average similarity score: {avg_score:.3f}")
    
    # Display each result
    for i, post in enumerate(results, 1):
        with st.expander(f"📄 Result {i}: {post.post_owner} ({post.score:.3f} similarity)", expanded=i <= 3):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Author:** {post.post_owner}")
                st.markdown(f"**Source:** {post.source}")
                if post.rerank_score:
                    st.markdown(f"**Rerank Score:** {post.rerank_score:.3f}")
                
                st.markdown("**Content:**")
                st.write(post.text)
                
                if post.image:
                    st.markdown(f"**Image:** {post.image}")
            
            with col2:
                st.metric("Similarity", f"{post.score:.3f}")
                
                # Analytics features
                if ANALYTICS_AVAILABLE:
                    if st.button(f"🔍 Find Similar", key=f"similar_{post.chunk_id}"):
                        show_similar_content(post)
                    
                    if st.button(f"👥 Author Posts", key=f"author_{post.chunk_id}"):
                        show_author_content(post.post_owner)

def show_recommendations(search_results, query):
    """Show personalized recommendations"""
    
    if not ANALYTICS_AVAILABLE:
        return
    
    st.markdown("---")
    st.subheader("🤖 Personalized Recommendations")
    
    try:
        recommendation_engine = get_recommendation_engine()
        user_id = st.session_state.analytics_session_id
        
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            user_id=user_id,
            content_pool=search_results + st.session_state.get('all_content', []),
            num_recommendations=5
        )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"💡 Recommendation {i}: {rec.post.post_owner} ({rec.score:.3f})", expanded=i == 1):
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Reason:** {rec.reason}")
                        st.markdown(f"**Type:** {rec.recommendation_type}")
                        st.markdown(f"**Author:** {rec.post.post_owner}")
                        st.write(rec.post.text[:300] + "..." if len(rec.post.text) > 300 else rec.post.text)
                    
                    with col2:
                        st.metric("Score", f"{rec.score:.3f}")
                        st.metric("Confidence", f"{rec.confidence:.3f}")
        else:
            st.info("🤖 Building your preference profile... Search more to get better recommendations!")
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

def show_similar_content(post):
    """Show content similar to the selected post"""
    
    if not ANALYTICS_AVAILABLE:
        return
    
    st.markdown("### 🔍 Similar Content")
    
    try:
        embedding_model = EmbeddingModelSingleton()
        vector_db_client = build_qdrant_client()
        retriever = AnalyticsEnabledRetriever(
            embedding_model=embedding_model,
            vector_db_client=vector_db_client
        )
        
        similar_posts = retriever.get_related_content(post, limit=5)
        
        if similar_posts:
            for similar_post in similar_posts:
                st.write(f"**{similar_post.post_owner}** ({similar_post.score:.3f})")
                st.write(similar_post.text[:200] + "...")
                st.markdown("---")
        else:
            st.info("No similar content found")
    
    except Exception as e:
        st.error(f"Error finding similar content: {str(e)}")

def show_author_content(author):
    """Show all content by a specific author"""
    
    if not ANALYTICS_AVAILABLE:
        return
    
    st.markdown(f"### 👥 Content by {author}")
    
    try:
        embedding_model = EmbeddingModelSingleton()
        vector_db_client = build_qdrant_client()
        retriever = AnalyticsEnabledRetriever(
            embedding_model=embedding_model,
            vector_db_client=vector_db_client
        )
        
        author_posts = retriever.get_content_by_author(author, limit=10)
        
        if author_posts:
            for post in author_posts:
                st.write(f"**Score:** {post.score:.3f}")
                st.write(post.text[:200] + "...")
                st.markdown("---")
        else:
            st.info(f"No additional content found for {author}")
    
    except Exception as e:
        st.error(f"Error fetching author content: {str(e)}")

def render_recent_activity():
    """Render recent search activity and trending content"""
    
    if not ANALYTICS_AVAILABLE:
        return
    
    st.markdown("---")
    st.subheader("🔥 Recent Activity & Trending")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🕐 Recent Searches**")
        try:
            metrics_collector = get_metrics_collector()
            search_analytics = metrics_collector.get_search_analytics(hours_back=24)
            
            recent_queries = search_analytics.get("top_queries", [])[:5]
            
            if recent_queries:
                for query, count in recent_queries:
                    if st.button(f"🔍 {query} ({count})", key=f"recent_query_{hash(query)}"):
                        st.session_state.main_search_query = query
                        st.rerun()
            else:
                st.info("No recent searches")
        
        except Exception as e:
            st.error(f"Error loading recent searches: {str(e)}")
    
    with col2:
        st.markdown("**📈 Trending Authors**")
        try:
            search_analytics = metrics_collector.get_search_analytics(hours_back=24)
            top_authors = search_analytics.get("top_authors", [])[:5]
            
            if top_authors:
                for author, count in top_authors:
                    st.write(f"👥 **{author}** ({count} searches)")
            else:
                st.info("No trending authors")
        
        except Exception as e:
            st.error(f"Error loading trending authors: {str(e)}")

def main():
    """Main application with enhanced analytics"""
    
    # Render enhanced header
    render_enhanced_header()
    
    # Show analytics dashboard if toggled
    if st.session_state.get('show_analytics', False) and ANALYTICS_AVAILABLE:
        st.markdown("---")
        render_analytics_dashboard()
        return
    
    # Render system info
    render_system_info()
    
    # Render enhanced sidebar and get settings
    number_of_results, analytics_settings = render_enhanced_sidebar()
    st.session_state.number_of_results = number_of_results
    st.session_state.analytics_settings = analytics_settings
    
    # Main application workflow
    username = None
    
    # Step 1: Data Source Configuration
    with st.expander("📥 **Step 1: Configure Data Source**", expanded=True):
        username = basic_prerequisites()
    
    # Step 2: Data Processing 
    with st.expander("🔄 **Step 2: Process Data for AI Search**", expanded=False):
        st.markdown("Convert your posts into AI-searchable format")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 **Required**: Process data before searching to enable semantic search capabilities")
        with col2:
            migrate_data = st.button("🚀 **Start Processing**", type="primary", use_container_width=True)
        
        if migrate_data:
            # Initialize analytics if available
            if ANALYTICS_AVAILABLE:
                metrics_collector = get_metrics_collector()
            
            migrate_data_to_vectordb(username, st.sidebar.checkbox("🔍 Debug Mode", value=False))
    
    # Step 3: Enhanced Search Interface
    st.markdown("---")
    get_enhanced_insights_from_posts()
    
    # Footer with system information
    render_enhanced_footer()

def render_enhanced_footer():
    """Render enhanced footer with analytics information"""
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("ℹ️ **About This System**", expanded=False):
            st.markdown("""
            ### 🎯 **Real-Time LinkedIn Content Quest**
            
            Advanced AI-powered system for discovering and analyzing LinkedIn content:
            
            **🔍 Core Features:**
            - **🤖 AI Models**: Sentence-transformers + cross-encoders for semantic search
            - **⚡ Real-time Processing**: Bytewax pipeline for efficient data processing
            - **🗄️ Vector Storage**: Qdrant database for fast similarity search
            - **🔍 Smart Search**: Context-aware semantic understanding
            
            **📊 Analytics Features:**
            - **📈 Real-time Dashboard**: Search patterns and content insights
            - **🤖 Recommendations**: Personalized content suggestions
            - **🧩 Clustering**: Automatic content organization
            - **🎯 Advanced Filters**: Author, source, and content-based filtering
            
            **Built with**: Python, Streamlit, PyTorch, Qdrant, Bytewax, Plotly
            """)
    
    with col2:
        # Analytics session info
        if ANALYTICS_AVAILABLE:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 **Analytics Session**")
            st.sidebar.info(f"Session ID: {st.session_state.analytics_session_id[:8]}...")
            
            if 'current_search_results' in st.session_state:
                st.sidebar.success(f"✅ {len(st.session_state['current_search_results'])} results loaded")
            
            try:
                metrics_collector = get_metrics_collector()
                analytics = metrics_collector.get_search_analytics(hours_back=1)
                if analytics.get('total_searches', 0) > 0:
                    st.sidebar.metric("Searches This Hour", analytics['total_searches'])
            except:
                pass
        
        # Current session status
        st.sidebar.markdown("### 📊 **Current Session**")
        if 'posts_data' in st.session_state:
            st.sidebar.success(f"✅ {len(st.session_state['posts_data'])} posts loaded")
        else:
            st.sidebar.info("⏳ No posts loaded yet")

if __name__ == "__main__":
    main()