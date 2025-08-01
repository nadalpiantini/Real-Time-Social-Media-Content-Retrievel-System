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