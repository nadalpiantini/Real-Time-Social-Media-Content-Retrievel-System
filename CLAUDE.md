# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Real-Time Social Media Content Retrieval System that fetches LinkedIn posts and provides semantic search capabilities using vector embeddings. The system uses Bytewax for data processing pipelines, Qdrant as the vector database, and Streamlit for the web interface.

## Architecture

### Core Components
- **Data Sources**: Dual-mode system supporting both JSON files (`data/` folder) and Supabase database
- **Processing Pipeline**: Bytewax dataflow that processes raw posts → cleaned posts → chunked posts → embedded chunks
- **Vector Storage**: Qdrant vector database for similarity search
- **Web Interface**: Streamlit app (`app.py`) for user interaction
- **LinkedIn Scraper**: Selenium-based scraper (`linkedin_posts_scrapper.py`) for real-time data fetching

### Key Architecture Patterns
- **Singleton Pattern**: Used for embedding models (`EmbeddingModelSingleton`, `CrossEncoderModelSingleton`)
- **Unified Data Source**: `UnifiedDataSource` class automatically detects and switches between Supabase and JSON file sources
- **Pipeline Processing**: Bytewax dataflow handles the complete ETL pipeline from raw data to vector embeddings

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start Qdrant vector database (required)
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Run the Streamlit app
streamlit run app.py
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t social-media-retrieval .
docker run -p 8501:8501 social-media-retrieval
```

### Data Processing Pipeline
```bash
# Run the processing pipeline standalone (for testing)
python flow.py
```

## Configuration

### Environment Variables (`.env` file)
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase anon key
- `SUPABASE_SERVICE_KEY`: Supabase service role key
- `QDRANT_URL`: Qdrant server URL (default: localhost:6333)
- `QDRANT_API_KEY`: Optional Qdrant API key
- `USE_SUPABASE`: Boolean flag to enable Supabase over JSON files

### Model Configuration
Default models in `models/settings.py`:
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Embedding Size: 384 dimensions

## Data Flow

1. **Data Ingestion**: LinkedIn posts scraped via Selenium or loaded from Supabase
2. **Processing Pipeline**: 
   - Raw posts → Cleaned posts (text preprocessing)
   - Cleaned posts → Chunked posts (text segmentation)
   - Chunked posts → Embedded posts (vector generation)
3. **Storage**: Vectors stored in Qdrant with metadata
4. **Retrieval**: Semantic search using embedding similarity + cross-encoder reranking

## Important Implementation Details

### Data Source Switching
The system automatically detects available data sources:
- If Supabase credentials are configured and available, uses Supabase
- Otherwise falls back to JSON files in `data/` folder
- JSON files must follow format: `{username}_data.json`

### Processing Pipeline
The Bytewax flow in `flow.py` is stateless and can be run in-memory or persistent mode:
- In-memory mode: Uses `:memory:` Qdrant client for testing
- Persistent mode: Connects to external Qdrant instance

### Model Management
Embedding models are managed as singletons to avoid reloading:
- Models are loaded once and reused across the application
- Default device is CPU, can be configured via `EMBEDDING_MODEL_DEVICE`

## Testing and Development

### Running Tests
The project doesn't include a specific test framework setup. When adding tests:
- Follow the existing project structure
- Test the data processing pipeline components individually
- Mock external dependencies (LinkedIn, Supabase, Qdrant) for unit tests

### Development Workflow
1. Use the Streamlit interface for interactive development
2. Test pipeline components with `ingest.py` or `flow.py`
3. Use the Jupyter notebook `retrieve.ipynb` for analysis and debugging
4. LinkedIn scraping requires valid credentials and may open browser windows

## External Dependencies

### Required Services
- **Qdrant**: Vector database (Docker container or cloud instance)
- **Supabase** (optional): PostgreSQL database with real-time subscriptions
- **Chrome/Chromium**: Required for Selenium LinkedIn scraping

### LinkedIn Scraping Considerations
- Requires valid LinkedIn credentials
- May trigger browser automation detection
- Respects LinkedIn's rate limits and terms of service
- Selenium WebDriver manages browser instances automatically

## 🔧 Debugging & Troubleshooting

### Enhanced Data Ingestion Debugging
The system includes comprehensive debugging for the "🛠️ Data Ingestion to VectorDB" process:

#### **Progress Indicators**
- Real-time progress bars for each stage (20%, 40%, 60%, 80%, 100%)
- ML model loading progress (shows ~500MB download on first run)
- Step-by-step pipeline execution tracking
- Post counting and file validation

#### **Detailed Logging**
```
📁 Using 1 non-empty JSON files (17 posts) for migration
🧠 Loading ML models...
✅ Embedding model loaded successfully
✅ Cross-encoder model loaded successfully
🔧 Building data processing pipeline...
📊 Data source paths: ['data/manthanbhikadiya_data.json']
📖 Reading file: data/manthanbhikadiya_data.json
📊 File manthanbhikadiya_data.json: 17 posts found
🔄 Input: Processing tuple
🔄 Raw Post: Processing post Post_1
🔄 Cleaned Post: Processing post Post_1
🔄 Chunked Post: Processing post Post_1
🔄 Embedded Post: Processing post Post_1
🔍 Final: Post Post_1 → 380 chars → 384 dims
💾 QdrantVectorSink: Writing batch of 18 chunks...
✅ Successfully upserted 18 points to Qdrant!
```

#### **Timeout Handling**
- **30s warning**: "Pipeline taking longer than expected"
- **60s warning**: "Model may be downloading (~500MB)"
- **120s warning**: "Possible network issue"

#### **Error Capture**
- Pipeline output and errors displayed in Streamlit UI
- Fallback mechanisms for missing dependencies
- Clear error messages with suggested fixes

### **Common Issues & Solutions**

#### **1. Data Ingestion Hanging**
If "🛠️ Data Ingestion to VectorDB" hangs, the enhanced logging shows exactly where:

**Check the logs for:**
- **Empty Files**: System automatically filters empty JSON files
- **Model Downloads**: First run downloads ~500MB embedding models
- **Missing Dependencies**: See dependency installation below
- **Qdrant Connection**: System falls back to in-memory mode automatically

#### **2. Missing Dependencies Error**
```bash
# Install core ML dependencies
pip install sentence-transformers==2.7.0 qdrant-client==1.9.0 bytewax pandas numpy

# Fix protobuf version conflicts (required for Bytewax)
pip install 'protobuf<4,>=3.12'

# Install remaining dependencies
pip install pydantic-settings supabase python-dotenv langchain langchain-text-splitters unstructured
```

#### **3. File Processing Issues**
The system now shows detailed file processing:
- **manthanbhikadiya_data.json**: 17 posts ✅
- **nadalpiantini_data.json**: 0 posts ⚠️ (automatically filtered)

#### **4. Pipeline Stage Debugging**
Each pipeline stage is now logged:
1. **Input**: Raw data loading from JSON files
2. **Raw Post**: Converting to RawPost objects  
3. **Cleaned Post**: Text cleaning and preprocessing
4. **Chunked Post**: Text segmentation for embedding
5. **Embedded Post**: Vector generation using ML models
6. **VectorDB**: Storage in Qdrant database

### **Performance Notes**
- **First Run**: Downloads embedding models (~500MB) - can take 5-10 minutes
- **Subsequent Runs**: Uses cached models - much faster
- **Processing Speed**: ~1-2 seconds per post after models are loaded
- **Memory Usage**: ~2GB RAM for ML models

### **Dependency Requirements**
```txt
# Core requirements from requirements.txt
streamlit>=1.47.0
sentence-transformers>=2.7.0
torch>=2.3.0
qdrant-client>=1.9.0
bytewax  # ⚠️ Was missing - now added
pandas>=2.2.0
numpy>=1.26.0
protobuf<4,>=3.12  # ⚠️ Version critical for Bytewax
```