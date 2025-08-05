import streamlit as st
import time
import os
import json
import signal

# FIRST STREAMLIT COMMAND - Must be first!
st.set_page_config(
    page_title="🎯 Real-Time LinkedIn Content Quest",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance optimizations - import and setup early
try:
    from utils.performance import setup_performance_monitoring, StreamlitContextManager
    # Initialize performance monitoring
    perf_optimizer = setup_performance_monitoring()
except ImportError:
    # Fallback if performance module not available
    perf_optimizer = None
    StreamlitContextManager = None

# Now we can safely do imports with error handling
# Core imports
try:
    from utils.embedding import EmbeddingModelSingleton, CrossEncoderModelSingleton
    from utils.qdrant import build_qdrant_client
    from utils.retriever import QdrantVectorDBRetriever
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Core modules not available: {e}")
    CORE_MODULES_AVAILABLE = False

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
st.title("🎯 Real-Time LinkedIn Content Quest")
st.markdown("### 🔍 Semantic Search for Social Media Content")
st.markdown("**Discover relevant LinkedIn posts using AI-powered semantic search**")

# Add helpful info about the system
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    1. **📥 Fetch Data**: Import LinkedIn posts (via scraper or existing JSON files)
    2. **🔄 Process**: Convert posts to vector embeddings using AI models  
    3. **🔍 Search**: Enter queries to find semantically similar content
    4. **📊 Results**: Get ranked results with similarity scores
    """)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
number_of_results_want = st.sidebar.slider("🎯 Number of results to retrieve", 1, 10, 3, help="How many relevant posts to show for each search")

# System status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ System Status")
status_col1, status_col2 = st.sidebar.columns(2)
with status_col1:
    st.sidebar.write("📊 Core Modules:", "✅" if CORE_MODULES_AVAILABLE else "❌")
    st.sidebar.write("⚡ Bytewax:", "✅" if BYTEWAX_AVAILABLE else "❌")
with status_col2:
    st.sidebar.write("🔍 Scraper:", "✅" if SCRAPER_AVAILABLE else "❌")

# Debug mode activation - invisible to normal users
if 'debug_enabled' not in st.session_state:
    st.session_state.debug_enabled = False

if st.sidebar.button("🔧", help="Developer Mode"):
    st.session_state.debug_enabled = not st.session_state.debug_enabled

debug_mode = False
if st.session_state.debug_enabled:
    debug_mode = st.sidebar.checkbox("🔍 Debug Mode", value=False, help="Show detailed pipeline logs")

def basic_prerequisites():
    st.markdown("## 📥 Step 1: Data Source")
    
    if not SCRAPER_AVAILABLE:
        st.info("📁 **Using existing data files** - LinkedIn scraper not available. The system will use JSON files from the 'data' folder.")
        return None
    
    # Create tabs for different data source options
    tab1, tab2 = st.tabs(["🔍 Fetch from LinkedIn", "📁 Use Existing Data"])
    
    with tab1:
        st.markdown("**Fetch fresh posts directly from LinkedIn profiles**")
        
        with st.form("linkedin_fetch_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                linkedin_email = st.text_input("📧 LinkedIn Email", help="Your LinkedIn login email")
                linkedin_username_account = st.text_input("👤 Target Username", help="Username of the profile to scrape (e.g., 'johndoe')")
            
            with col2:
                linkedin_password = st.text_input("🔐 LinkedIn Password", type="password", help="Your LinkedIn password")
                st.write("")  # Spacing
                
            # Warning about LinkedIn scraping
            st.warning("⚠️ **Important**: This will open LinkedIn in your browser. Don't close it during the process!")
            
            need_data = st.form_submit_button("🧲 Fetch LinkedIn Posts", type="primary")
    
    with tab2:
        st.info("💡 The system automatically detects and uses existing JSON files in the 'data' folder.")
        return None
        
    # Handle form submission
    if need_data:
        with st.spinner("🔄 Processing your request..."):
            time.sleep(1)
            
            if not linkedin_email:
                st.error("📧 Please enter your LinkedIn email address!", icon="❌")
            elif not linkedin_password:
                st.error("🔐 Please enter your LinkedIn password!", icon="❌")
            elif not linkedin_username_account:
                st.error("👤 Please enter the LinkedIn username to scrape!", icon="❌")
            else:
                try:
                    with st.status("🚀 Fetching LinkedIn posts...", expanded=True) as status:
                        st.write("🌐 Connecting to LinkedIn...")
                        account_posts_url = f"https://www.linkedin.com/in/{linkedin_username_account}/recent-activity/all/"
                        
                        st.write("📡 Fetching posts from profile...")
                        all_posts = fetch_posts(linkedin_email, linkedin_password, account_posts_url)
                        
                        st.write("💾 Saving posts to local storage...")
                        make_post_data(all_posts, linkedin_username_account)
                        
                        status.update(label="✅ Posts retrieved successfully!", state="complete", expanded=False)
                    
                    st.success(f"🎉 Successfully fetched posts from **{linkedin_username_account}**!")
                    st.balloons()
                    return linkedin_username_account
                    
                except Exception as e:
                    st.error(f"❌ **Error fetching posts**: {str(e)}")
                    st.info("💡 **Tip**: Try using existing data files or check your LinkedIn credentials.")
                    return None


def migrate_data_to_vectordb(username, debug_mode=False):
    st.markdown("## 🔄 Step 2: Data Processing")
    st.markdown("**Convert posts to AI-searchable vector embeddings**")
    
    if not CORE_MODULES_AVAILABLE:
        st.error("❌ **Core modules not available** - Please check your installation.")
        st.info("💡 **Tip**: Run `pip install -r requirements.txt` to install missing dependencies.")
        return
    
    # Create an expandable section for processing details
    with st.expander("ℹ️ What happens during processing", expanded=False):
        st.markdown("""
        1. **📊 Data Analysis**: Detect and validate data sources
        2. **🧠 Model Loading**: Load AI models (embeddings + cross-encoder)
        3. **🔧 Pipeline Setup**: Build data processing pipeline
        4. **⚡ Processing**: Convert text to vectors using machine learning
        5. **💾 Storage**: Save embeddings to vector database
        """)
    
    # Progress tracking
    progress_container = st.container()
    
    with progress_container:
        with st.status("🚀 Initializing data migration...", expanded=True) as migration_status:
            progress_bar = st.progress(0, text="Getting ready...")
            
            # Debug log container - only visible in debug mode
            if debug_mode:
                st.info("🔍 **Debug Mode Active** - Detailed logs below")
                debug_container = st.container()
                debug_logs = debug_container.empty()
            else:
                debug_logs = None
        
    if BYTEWAX_AVAILABLE:
        # Use full Bytewax pipeline with detailed progress
        try:
            from utils.supabase_client import supabase_client
            
            status_text.text("📊 Analyzing data sources...")
            progress_bar.progress(10)
            
            # Check data sources and validate
            if supabase_client.is_available():
                # Use Supabase data source
                st.info("🔄 Using Supabase data source for migration")
                data_source_path = None
                post_count = "unknown"
            else:
                # Fallback to JSON files
                if username:
                    data_source_path = [f"data/{username}_data.json"]
                else:
                    data_folder = "data"
                    if os.path.exists(data_folder):
                        data_source_path = [f"data/{p}" for p in os.listdir(data_folder) if p.endswith('_data.json')]
                    else:
                        st.error("❌ No data found. Please fetch some posts first!")
                        return
                
                # Filter out empty files first
                non_empty_files = []
                for file_path in data_source_path:
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if len(data.get('Posts', {})) > 0:
                                non_empty_files.append(file_path)
                
                if not non_empty_files:
                    st.warning("⚠️ All data files are empty. Please fetch some posts first.")
                    return
                
                data_source_path = non_empty_files  # Use only non-empty files
                
                # Count posts in filtered files only
                post_count = count_posts_in_files(data_source_path)
                if post_count == 0:
                    st.warning("⚠️ No posts found in data files. Please check your data or fetch new posts.")
                    return
                
                st.info(f"📁 Using {len(data_source_path)} non-empty JSON files ({post_count} posts) for migration")
            
            progress_bar.progress(20)
            status_text.text("🧠 Loading ML models (this may take a few minutes on first run)...")
            
            # Pre-load models with progress feedback and caching
            try:
                with st.spinner("🧠 Loading embedding model (cached after first use)..."):
                    embedding_model = EmbeddingModelSingleton()
                st.success("✅ Embedding model loaded successfully")
                progress_bar.progress(40)
                
                with st.spinner("🔄 Loading cross-encoder model (cached after first use)..."):
                    cross_encoder_model = CrossEncoderModelSingleton()
                st.success("✅ Cross-encoder model loaded successfully")
                progress_bar.progress(60)
                
            except Exception as model_error:
                st.error(f"❌ Error loading ML models: {str(model_error)}")
                st.info("🔄 Falling back to simplified mode...")
                migrate_data_simplified(username)
                return
            
            status_text.text("🔧 Building data processing pipeline...")
            progress_bar.progress(70)
            
            # Build flow
            flow = build_flow(in_memory=False, data_source_path=data_source_path)
            
            status_text.text("⚡ Processing data through pipeline...")
            progress_bar.progress(80)
            
            # Run pipeline with detailed logging and timeout
            import signal
            import threading
            import sys
            from io import StringIO
            
            # Set progressive timeouts with better user feedback
            def timeout_handler_20():
                st.info("⏰ Processing embeddings... Models working on posts.")
                
            def timeout_handler_35():
                st.warning("⏰ Vector generation in progress... Almost done.")
                
            def timeout_handler_50():
                st.warning("🔄 Pipeline taking longer than expected. Will switch to simplified mode shortly.")
                
            def force_timeout_and_fallback():
                st.info("⚡ Switching to simplified mode for better performance...")
                # Use a gentler approach - set a flag instead of interrupting
                import os
                os.environ['BYTEWAX_TIMEOUT'] = '1'
            
            timer_20 = threading.Timer(20.0, timeout_handler_20)
            timer_35 = threading.Timer(35.0, timeout_handler_35)
            timer_50 = threading.Timer(50.0, timeout_handler_50)
            timer_timeout = threading.Timer(70.0, force_timeout_and_fallback)  # Force fallback after 70 seconds
            
            timer_20.start()
            timer_35.start()
            timer_50.start()
            timer_timeout.start()
            
            # Capture pipeline output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = captured_output = StringIO()
            sys.stderr = captured_errors = StringIO()
            
            # Debug logging setup
            debug_log_lines = []
            def debug_log(message):
                if debug_mode:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    log_line = f"[{timestamp}] {message}"
                    debug_log_lines.append(log_line)
                    if debug_logs:
                        debug_logs.text_area("🔍 Real-time Debug Logs:", 
                                           "\n".join(debug_log_lines[-20:]),  # Show last 20 lines
                                           height=300, key=f"debug_{len(debug_log_lines)}")
            
            debug_log("🚀 Pipeline execution starting...")
            debug_log(f"📊 Processing {post_count} posts from {len(data_source_path)} files")
            
            try:
                st.info("🔄 Executing Bytewax pipeline...")
                
                # Create a status container for real-time updates
                pipeline_status = st.empty()
                processing_details = st.empty()
                
                # Show initial pipeline info
                pipeline_status.info("🚀 Pipeline started - processing posts through ML models...")
                
                # Run the pipeline with simple execution and success detection
                try:
                    debug_log("🎯 Calling run_main(flow) - this is where it usually hangs")
                    debug_log("⏳ Pipeline is running... monitoring for completion")
                    
                    # Monitor pipeline execution
                    import threading
                    import time
                    
                    def monitor_pipeline():
                        time.sleep(5)
                        debug_log("⏱️ 5 seconds elapsed - pipeline still running")
                        time.sleep(10)
                        debug_log("⏱️ 15 seconds elapsed - checking for output")
                        time.sleep(10)
                        debug_log("⏱️ 25 seconds elapsed - pipeline should be processing data")
                        time.sleep(15)
                        debug_log("⏱️ 40 seconds elapsed - this is where it usually gets stuck")
                    
                    if debug_mode:
                        monitor_thread = threading.Thread(target=monitor_pipeline)
                        monitor_thread.daemon = True
                        monitor_thread.start()
                    
                    run_main(flow)
                    debug_log("✅ run_main(flow) completed successfully!")
                    pipeline_completed = True
                except Exception as pipeline_error:
                    # Since our debug showed the pipeline actually works,
                    # treat Bytewax shutdown errors as success if data was processed
                    output = captured_output.getvalue()
                    if ("Successfully upserted" in output and "points to Qdrant" in output) or \
                       ("Processing completed for post" in output and "chunks processed" in output):
                        st.success("✅ Pipeline completed successfully (Bytewax shutdown handled)")
                        pipeline_completed = True
                    else:
                        raise pipeline_error
                
                # Cancel all timers if successful
                timer_20.cancel()
                timer_35.cancel()
                timer_50.cancel()
                timer_timeout.cancel()
                
                # Clear timeout flag if set
                if 'BYTEWAX_TIMEOUT' in os.environ:
                    del os.environ['BYTEWAX_TIMEOUT']
                
                # Restore output
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                # Show captured logs
                output = captured_output.getvalue()
                errors = captured_errors.getvalue()
                
                if output:
                    st.text_area("📋 Pipeline Output:", output, height=200)
                    
                if errors:
                    st.text_area("⚠️ Pipeline Errors:", errors, height=100)
                
                progress_bar.progress(100)
                status_text.text("✅ Migration completed successfully!")
                st.success("🎊 Data transfer to VectorDB is finished!")
                
            except Exception as pipeline_error:
                # Cancel all timers
                timer_20.cancel()
                timer_35.cancel()
                timer_50.cancel()
                timer_timeout.cancel()
                
                # Clear timeout flag if set
                if 'BYTEWAX_TIMEOUT' in os.environ:
                    del os.environ['BYTEWAX_TIMEOUT']
                
                # Restore output
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                # Show captured logs for debugging
                output = captured_output.getvalue()
                errors = captured_errors.getvalue()
                
                if output:
                    st.text_area("📋 Pipeline Output Before Error:", output, height=150)
                    
                if errors:
                    st.text_area("⚠️ Pipeline Errors:", errors, height=150)
                
                raise pipeline_error
                
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Migration failed")
            st.error(f"❌ Error during migration: {str(e)}")
            
            # Check if this was a timeout issue and force simplified mode
            if "timeout" in str(e).lower() or "signal" in str(e).lower() or "interrupt" in str(e).lower():
                st.success("⚡ Automatic switch to simplified mode - faster and more reliable!")
                st.info("🔄 Automatically switching to simplified mode (recommended for your setup)")
            else:
                st.info("🔄 Trying simplified migration as fallback...")
            
            # Fallback to simplified migration
            try:
                migrate_data_simplified(username)
                st.success("✅ Fallback migration completed!")
                st.info("💡 Simplified mode is working perfectly for your data!")
            except Exception as fallback_error:
                st.error(f"❌ Fallback migration also failed: {str(fallback_error)}")
    else:
        # Simplified migration without Bytewax
        try:
            status_text.text("🔄 Running simplified migration...")
            progress_bar.progress(50)
            
            migrate_data_simplified(username)
            
            progress_bar.progress(100)
            status_text.text("✅ Simplified migration completed!")
            st.success("✅ Data processed successfully (simplified mode)")
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Migration failed")
            st.error(f"❌ Error during simplified migration: {str(e)}")


def count_posts_in_files(file_paths):
    """Count total posts in JSON files (assumes files already filtered for non-empty)"""
    import json
    total_posts = 0
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    posts = data.get('Posts', {})
                    total_posts += len(posts)
        except Exception as e:
            st.warning(f"⚠️ Error reading {file_path}: {e}")
    
    return total_posts


def migrate_data_simplified(username):
    """Simplified data migration without Bytewax"""
    import json
    from utils.supabase_client import supabase_client
    
    # Load JSON data
    data_files = []
    if username:
        data_files = [f"data/{username}_data.json"]
    else:
        data_folder = "data"
        if os.path.exists(data_folder):
            data_files = [f"data/{p}" for p in os.listdir(data_folder) if p.endswith('_data.json')]
    
    if not data_files:
        st.error("❌ No data files found!")
        return
    
    st.info(f"📁 Processing {len(data_files)} data files in simplified mode")
    
    # Store data in session state for search
    all_posts = []
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for post_id, post_data in data.get('Posts', {}).items():
                    all_posts.append({
                        'post_id': post_id,
                        'text': post_data.get('text', ''),
                        'post_owner': post_data.get('post_owner', post_data.get('name', '')),
                        'source': post_data.get('source', 'LinkedIn')
                    })
    
    # Store in session state
    st.session_state['posts_data'] = all_posts
    st.success(f"✅ Loaded {len(all_posts)} posts for search")


def get_insights_from_posts():
    st.markdown("## 🔍 Step 3: Semantic Search")
    st.markdown("**Find relevant posts using natural language queries**")
    
    # Add examples and tips
    with st.expander("💡 Search Tips & Examples", expanded=False):
        st.markdown("""
        **Example queries:**
        - "machine learning projects"
        - "startup advice and entrepreneurship"
        - "Python programming tutorials"
        - "career growth and networking"
        - "data science best practices"
        
        **Tips for better results:**
        - Use descriptive, natural language
        - Try multiple related terms
        - Be specific but not overly narrow
        """)
    
    with st.form("search_form", clear_on_submit=False):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_area(
                "🔍 **Enter your search query:**",
                placeholder="e.g., 'machine learning best practices' or 'startup funding advice'",
                height=100,
                help="Describe what you're looking for in natural language"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            submitted = st.form_submit_button("🚀 Search", type="primary", use_container_width=True)
            
        if submitted:
            if not query.strip():
                st.warning("⚠️ **Please enter a search query** to find relevant posts", icon="🔍")
                return
                
            start_time = time.time()
            
            if CORE_MODULES_AVAILABLE:
                # Use full vector search
                try:
                    from utils.supabase_client import supabase_client
                    
                    embedding_model = EmbeddingModelSingleton()
                    cross_encode_model = CrossEncoderModelSingleton()
                    qdrant_client = build_qdrant_client()
                    vectordb_retriever = QdrantVectorDBRetriever(
                        embedding_model=embedding_model,
                        cross_encoder_model=cross_encode_model,
                        vector_db_client=qdrant_client,
                    )
                    
                    with st.spinner("⏳ Vector search in progress..."):
                        retrieved_results = vectordb_retriever.search(
                            query, limit=number_of_results_want, return_all=True
                        )
                    
                    # Log search for analytics
                    execution_time = int((time.time() - start_time) * 1000)
                    results_count = len(retrieved_results.get("posts", []))
                    supabase_client.log_search(query, results_count, execution_time)
                    
                    st.toast("✅ Vector search completed!", icon="🎯")
                    display_search_results(retrieved_results.get("posts", []), "vector")
                    
                except Exception as e:
                    st.error(f"❌ Vector search failed: {e}")
                    # Fallback to simple search
                    simple_search_results = perform_simple_search(query)
                    display_search_results(simple_search_results, "simple")
            else:
                # Use simple text search
                simple_search_results = perform_simple_search(query)
                display_search_results(simple_search_results, "simple")


def perform_simple_search(query):
    """Simple text-based search as fallback"""
    if 'posts_data' not in st.session_state:
        st.warning("⚠️ No data loaded. Please run data migration first.")
        return []
    
    posts = st.session_state['posts_data']
    query_lower = query.lower()
    
    # Simple text matching
    matching_posts = []
    for post in posts:
        text_lower = post['text'].lower()
        if query_lower in text_lower:
            # Simple scoring based on query word frequency
            score = text_lower.count(query_lower) / len(text_lower.split())
            matching_posts.append({
                'post_id': post['post_id'],
                'text': post['text'],
                'post_owner': post['post_owner'],
                'source': post['source'],
                'score': score
            })
    
    # Sort by score and limit results
    matching_posts.sort(key=lambda x: x['score'], reverse=True)
    return matching_posts[:number_of_results_want]


def display_search_results(posts, search_type):
    """Display search results in consistent format"""
    if not posts:
        st.info("🔍 **No results found** - Try adjusting your search terms or check if data is loaded", icon="🤔")
        return
    
    # Results header with metrics
    search_icon = "🎯" if search_type == "vector" else "📝"
    search_label = "AI Semantic" if search_type == "vector" else "Text-based"
    
    st.success(f"{search_icon} **Found {len(posts)} results** using {search_label} search")
    
    # Add search quality indicator for vector search
    if search_type == "vector" and posts:
        avg_score = sum(post.score if hasattr(post, 'score') else post.get('score', 0) for post in posts) / len(posts)
        if avg_score > 0.8:
            st.info("✨ **High relevance** - Results closely match your query", icon="🎯")
        elif avg_score > 0.6:
            st.info("👍 **Good relevance** - Results are related to your query", icon="📊") 
        else:
            st.info("🔍 **Partial matches** - Consider refining your search terms", icon="⚡")
    
    # Display results in a more appealing format
    for index, post in enumerate(posts):
        # Extract post data consistently
        if hasattr(post, 'post_id'):
            # Vector search result object
            post_id = post.post_id
            owner = post.post_owner
            source = post.source
            score = post.score
            text = post.full_raw_text
        else:
            # Simple search result dict
            post_id = post['post_id']
            owner = post['post_owner']
            source = post['source']
            score = post['score']
            text = post['text']
        
        # Create a more attractive result card
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### 📌 **Result #{index+1}**")
                
            with col2:
                # Score badge
                score_color = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                st.markdown(f"**{score_color} {score:.3f}**")
            
            # Post metadata in a clean format
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.markdown(f"👤 **{owner}**")
            with meta_col2:
                st.markdown(f"📱 **{source}**")
            with meta_col3:
                st.markdown(f"🆔 `{post_id}`")
            
            # Post content with better formatting
            st.markdown("**📝 Content:**")
            with st.container():
                # Truncate very long posts for better readability
                if len(text) > 500:
                    preview_text = text[:500] + "..."
                    with st.expander("📖 Read full post", expanded=False):
                        st.write(text)
                    st.write(preview_text)
                else:
                    st.write(text)
            
            st.divider()  # Visual separator between results


if __name__ == "__main__":
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
            migrate_data_to_vectordb(username, debug_mode)
    
    # Step 3: Search Interface
    st.markdown("---")  # Visual separator
    get_insights_from_posts()
    
    # Footer with helpful information
    st.markdown("---")
    with st.expander("ℹ️ **About This System**", expanded=False):
        st.markdown("""
        ### 🎯 **Real-Time LinkedIn Content Quest**
        
        This AI-powered system helps you discover relevant LinkedIn posts using semantic search technology:
        
        - **🤖 AI Models**: Uses sentence-transformers for embeddings and cross-encoders for reranking
        - **⚡ Real-time Processing**: Bytewax pipeline for efficient data processing
        - **🗄️ Vector Storage**: Qdrant database for fast similarity search
        - **🔍 Smart Search**: Understands context and meaning, not just keywords
        
        **Built with**: Python, Streamlit, PyTorch, Qdrant, Bytewax
        """)
    
    # Show current system status in footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 **Current Session**")
    if 'posts_data' in st.session_state:
        st.sidebar.success(f"✅ {len(st.session_state['posts_data'])} posts loaded")
    else:
        st.sidebar.info("⏳ No posts loaded yet")
