import streamlit as st
import time
import os
import json

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

st.set_page_config(page_title="🎯 Real Time Linkedin Content Quest")
st.title("🎯 Real Time Linkedin Content Quest")
number_of_results_want = st.sidebar.slider("Number of results that you want to retrieve.",0,10,3)

def basic_prerequisites():
    if not SCRAPER_AVAILABLE:
        st.info("📁 LinkedIn scraper not available - you can still use existing data in the 'data' folder")
        return None
        
    linkedin_email = st.text_input("LinkedIn Email Address")
    linkedin_password = st.text_input("LinkedIn Password", type="password")
    linkedin_username_account = st.text_input(
        "Type in the username of the profile whose posts you'd like to get."
    )
    need_data = st.button("🧲 Fetch Details")
    if need_data:
        warn = st.warning(
            "Please keep in mind that this feature fetches data directly from LinkedIn. It might open LinkedIn in your web browser. Please avoid closing the browser while using this feature."
        )
        time.sleep(2)
        if not linkedin_email:
            st.warning("Please enter your linkedin email address for login!", icon="⚠")
        elif not linkedin_password:
            st.warning("Please enter your linkedin password for login!", icon="⚠")
        elif not linkedin_username_account:
            st.warning(
                "Please enter the linkedin username from which you want to fetch the posts!",
                icon="⚠",
            )
        else:
            try:
                account_posts_url = f"https://www.linkedin.com/in/{linkedin_username_account}/recent-activity/all/"
                all_posts = fetch_posts(
                    linkedin_email, linkedin_password, account_posts_url
                )
                make_post_data(all_posts, linkedin_username_account)
                warn.empty()
                warn = st.success("Success! All posts retrieved.")
                return linkedin_username_account
            except Exception as e:
                warn.empty()
                st.error(f"❌ Error fetching posts: {str(e)}")
                return None


def migrate_data_to_vectordb(username):
    if not CORE_MODULES_AVAILABLE:
        st.error("❌ Core modules not available for data migration")
        return
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.info("🚀 Starting data migration to VectorDB...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
                
                # Count posts in JSON files
                post_count = count_posts_in_files(data_source_path)
                if post_count == 0:
                    st.warning("⚠️ No posts found in data files. Please check your data or fetch new posts.")
                    return
                    
                # Filter out empty files
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
                st.info(f"📁 Using {len(data_source_path)} non-empty JSON files ({post_count} posts) for migration")
            
            progress_bar.progress(20)
            status_text.text("🧠 Loading ML models (this may take a few minutes on first run)...")
            
            # Pre-load models with progress feedback
            try:
                embedding_model = EmbeddingModelSingleton()
                st.success("✅ Embedding model loaded successfully")
                progress_bar.progress(40)
                
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
            
            # Set multiple timers for user feedback
            def timeout_handler_30():
                st.info("⏰ Pipeline is taking longer than expected. This is normal for first-time model downloads.")
                
            def timeout_handler_60():
                st.warning("⏰ Still processing... The embedding model may be downloading (can be ~500MB).")
                
            def timeout_handler_120():
                st.warning("⏰ Taking quite a while... If this persists, there might be a network issue.")
            
            timer_30 = threading.Timer(30.0, timeout_handler_30)
            timer_60 = threading.Timer(60.0, timeout_handler_60)
            timer_120 = threading.Timer(120.0, timeout_handler_120)
            
            timer_30.start()
            timer_60.start()
            timer_120.start()
            
            # Capture pipeline output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = captured_output = StringIO()
            sys.stderr = captured_errors = StringIO()
            
            try:
                st.info("🔄 Executing Bytewax pipeline...")
                
                # Create a status container for real-time updates
                pipeline_status = st.empty()
                processing_details = st.empty()
                
                # Show initial pipeline info
                pipeline_status.info("🚀 Pipeline started - processing posts through ML models...")
                
                # Run the pipeline
                run_main(flow)
                
                # Cancel all timers if successful
                timer_30.cancel()
                timer_60.cancel()
                timer_120.cancel()
                
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
                timer_30.cancel()
                timer_60.cancel()
                timer_120.cancel()
                
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
            st.info("🔄 Trying simplified migration as fallback...")
            
            # Fallback to simplified migration
            try:
                migrate_data_simplified(username)
                st.success("✅ Fallback migration completed!")
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
    """Count total posts in JSON files"""
    import json
    total_posts = 0
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    posts = data.get('Posts', {})
                    total_posts += len(posts)
                    if len(posts) == 0:
                        st.warning(f"⚠️ File {os.path.basename(file_path)} contains no posts")
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
    with st.form("my_form"):
        query = st.text_area(
            "✨ Spark a Search:",
            f"",
        )
        submitted = st.form_submit_button("Submit Query")
        if submitted:
            if not query.strip():
                st.warning("⚠️ Please enter a search query")
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
        st.info("🔍 No results found for your query")
        return
    
    st.success(f"🎯 Found {len(posts)} results using {search_type} search")
    
    for index, post in enumerate(posts):
        with st.expander(f"📌 Result-{index+1}"):
            if hasattr(post, 'post_id'):
                # Vector search result object
                st.subheader(f"PostID: {post.post_id}")
                st.markdown(f"**Post Owner**: {post.post_owner}")
                st.markdown(f"**Source**: {post.source}")
                st.markdown(f"**Similarity Score**: {post.score:.4f}")
                st.caption("Raw Text of Post:")
                st.write(post.full_raw_text)
            else:
                # Simple search result dict
                st.subheader(f"PostID: {post['post_id']}")
                st.markdown(f"**Post Owner**: {post['post_owner']}")
                st.markdown(f"**Source**: {post['source']}")
                st.markdown(f"**Match Score**: {post['score']:.4f}")
                st.caption("Raw Text of Post:")
                st.write(post['text'])


if __name__ == "__main__":
    username = None
    with st.expander("💡 Unlock LinkedIn Insights"):
        username = basic_prerequisites()
    with st.expander("🛠️ Data Ingestion to VectorDB"):
        migrate_data = st.button("🔨 Data Ingestion")
        if migrate_data:
            migrate_data_to_vectordb(username)
    get_insights_from_posts()
