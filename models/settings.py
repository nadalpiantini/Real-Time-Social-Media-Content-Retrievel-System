from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env"))
    
    # ML Models
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH:  int = 256
    EMBEDDING_SIZE : int = 384
    EMBEDDING_MODEL_DEVICE: str ="cpu"
    
    # Vector Database (Qdrant)
    VECTOR_DB_OUTPUT_COLLECTION_NAME: str = "linkedin_posts"
    QDRANT_URL: str ="localhost:6333"
    QDRANT_API_KEY: Optional[str]= None
    
    # Supabase Database
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_SERVICE_KEY: Optional[str] = None
    DATABASE_URL: Optional[str] = None  # PostgreSQL connection string
    
    # App Config
    USE_SUPABASE: bool = False  # Flag para alternar entre JSON y Supabase
    DATA_FOLDER: str = "data"

settings = AppSettings()