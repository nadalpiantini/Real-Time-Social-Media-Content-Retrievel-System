import streamlit as st
import time
import os

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
        
    warn = st.toast(
        "Hold on tight! We're moving data to a new system (VectorDB) to improve performance. We'll be back soon. ",
        icon="🚀",
    )
    
    if BYTEWAX_AVAILABLE:
        # Use full Bytewax pipeline
        try:
            from utils.supabase_client import supabase_client
            
            # Check if we should use Supabase or JSON files
            if supabase_client.is_available():
                # Use Supabase data source
                flow = build_flow(in_memory=False, data_source_path=None)
                st.info("🔄 Using Supabase data source for migration")
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
                        warn.empty()
                        return
                
                flow = build_flow(in_memory=False, data_source_path=data_source_path)
                st.info(f"📁 Using {len(data_source_path)} JSON files for migration")
            
            run_main(flow)
            warn.empty()
            warn = st.toast("We're all set! Data transfer to VectorDB is finished.", icon="🎊")
        except Exception as e:
            warn.empty()
            st.error(f"❌ Error during migration: {str(e)}")
            print(f"Migration error: {e}")
    else:
        # Simplified migration without Bytewax
        try:
            migrate_data_simplified(username)
            warn.empty()
            warn = st.toast("✅ Data processed successfully (simplified mode)", icon="🎊")
        except Exception as e:
            warn.empty()
            st.error(f"❌ Error during simplified migration: {str(e)}")


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
