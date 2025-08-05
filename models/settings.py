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

class SettingsWrapper:
    """Wrapper para compatibilidad con tests que buscan EMBEDDING_MODEL_NAME"""
    def __init__(self, settings_obj):
        self._settings = settings_obj
    
    def __getattr__(self, name):
        if name == 'EMBEDDING_MODEL_NAME':
            return self._settings.EMBEDDING_MODEL_ID
        return getattr(self._settings, name)

# Función de compatibilidad para tests
def get_settings():
    """Obtener instancia de configuración legacy"""
    return SettingsWrapper(settings)