"""
Sistema de configuración unificado con validación automática
Mantiene compatibilidad total con la configuración existente
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path

class DatabaseSettings(BaseModel):
    """Configuración de bases de datos y almacenamiento"""
    
    # Qdrant Vector Database
    qdrant_url: str = Field(
        default="localhost:6333", 
        description="URL del servidor Qdrant"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None, 
        description="API Key para Qdrant (opcional)"
    )
    collection_name: str = Field(
        default="linkedin_posts", 
        description="Nombre de la colección en Qdrant"
    )
    
    # Supabase Database
    supabase_url: Optional[str] = Field(
        default=None, 
        description="URL del proyecto Supabase"
    )
    supabase_key: Optional[str] = Field(
        default=None, 
        description="Key pública de Supabase"
    )
    supabase_service_key: Optional[str] = Field(
        default=None, 
        description="Service key de Supabase"
    )
    database_url: Optional[str] = Field(
        default=None, 
        description="URL de conexión directa a PostgreSQL"
    )
    
    # Data Storage
    use_supabase: bool = Field(
        default=False, 
        description="Usar Supabase en lugar de archivos JSON"
    )
    data_folder: str = Field(
        default="data", 
        description="Carpeta para archivos de datos JSON"
    )

class MLSettings(BaseModel):
    """Configuración de modelos de Machine Learning"""
    
    # Embedding Model
    embedding_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="ID del modelo de embeddings"
    )
    embedding_size: int = Field(
        default=384, 
        description="Dimensiones del vector embedding",
        gt=0
    )
    max_input_length: int = Field(
        default=256, 
        description="Longitud máxima de input para el modelo",
        gt=0
    )
    
    # Cross-Encoder Model
    cross_encoder_model_id: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="ID del modelo cross-encoder para reranking"
    )
    
    # Device Configuration
    device: str = Field(
        default="cpu", 
        description="Device para ejecutar modelos (cpu/cuda/mps)"
    )
    
    # Backward compatibility properties
    @property
    def embedding_model_name(self) -> str:
        """Alias for embedding_model_id for backward compatibility"""
        return self.embedding_model_id
    
    @property
    def embedding_dimension(self) -> int:
        """Alias for embedding_size for backward compatibility"""
        return self.embedding_size

class AppConfig(BaseSettings):
    """Configuración principal de la aplicación"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8", 
        case_sensitive=False,
        extra="ignore"  # Ignorar campos extra en lugar de fallar
    )
    
    # Configuraciones anidadas
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    
    # App Configuration
    app_name: str = Field(
        default="Real-Time Social Media Content Retrieval System",
        description="Nombre de la aplicación"
    )
    version: str = Field(
        default="2.0.0-architecture-improvements",
        description="Versión de la aplicación"
    )
    environment: str = Field(
        default="development", 
        description="Entorno de ejecución"
    )
    debug_mode: bool = Field(
        default=False, 
        description="Modo debug"
    )
    
    # Mapeo directo de campos del .env para compatibilidad
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Mapear valores del .env si existen
        env_mappings = {
            'QDRANT_URL': 'database.qdrant_url',
            'QDRANT_API_KEY': 'database.qdrant_api_key', 
            'VECTOR_DB_OUTPUT_COLLECTION_NAME': 'database.collection_name',
            'SUPABASE_URL': 'database.supabase_url',
            'SUPABASE_KEY': 'database.supabase_key',
            'SUPABASE_SERVICE_KEY': 'database.supabase_service_key',
            'DATABASE_URL': 'database.database_url',
            'USE_SUPABASE': 'database.use_supabase',
            'DATA_FOLDER': 'database.data_folder',
            
            'EMBEDDING_MODEL_ID': 'ml.embedding_model_id',
            'CROSS_ENCODER_MODEL_ID': 'ml.cross_encoder_model_id',
            'EMBEDDING_SIZE': 'ml.embedding_size',
            'EMBEDDING_MODEL_MAX_INPUT_LENGTH': 'ml.max_input_length',
            'EMBEDDING_MODEL_DEVICE': 'ml.device'
        }
        
        # Aplicar valores del entorno
        for env_key, nested_path in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Parsear path anidado y establecer valor
                parts = nested_path.split('.')
                if len(parts) == 2:
                    section, field = parts
                    if section == 'database' and hasattr(self.database, field):
                        # Convertir tipos apropiados
                        if field in ['use_supabase'] and isinstance(env_value, str):
                            env_value = env_value.lower() in ['true', '1', 'yes']
                        elif field in ['qdrant_api_key', 'supabase_url', 'supabase_key', 'supabase_service_key', 'database_url'] and env_value == '':
                            env_value = None
                        setattr(self.database, field, env_value)
                    elif section == 'ml' and hasattr(self.ml, field):
                        # Convertir tipos apropiados
                        if field in ['embedding_size', 'max_input_length'] and isinstance(env_value, str):
                            env_value = int(env_value)
                        setattr(self.ml, field, env_value)

# Función de compatibilidad con configuración existente
def create_legacy_settings_wrapper(config: AppConfig):
    """
    Crear wrapper que emula la interfaz de models.settings 
    para mantener compatibilidad hacia atrás
    """
    class LegacySettingsWrapper:
        def __init__(self, config: AppConfig):
            self._config = config
            
            # Mapear campos para compatibilidad
            self.EMBEDDING_MODEL_ID = config.ml.embedding_model_id
            self.CROSS_ENCODER_MODEL_ID = config.ml.cross_encoder_model_id
            self.EMBEDDING_MODEL_MAX_INPUT_LENGTH = config.ml.max_input_length
            self.EMBEDDING_SIZE = config.ml.embedding_size
            self.EMBEDDING_MODEL_DEVICE = config.ml.device
            
            self.VECTOR_DB_OUTPUT_COLLECTION_NAME = config.database.collection_name
            self.QDRANT_URL = config.database.qdrant_url
            self.QDRANT_API_KEY = config.database.qdrant_api_key
            
            self.SUPABASE_URL = config.database.supabase_url
            self.SUPABASE_KEY = config.database.supabase_key
            self.SUPABASE_SERVICE_KEY = config.database.supabase_service_key
            self.DATABASE_URL = config.database.database_url
            
            self.USE_SUPABASE = config.database.use_supabase
            self.DATA_FOLDER = config.database.data_folder
    
    return LegacySettingsWrapper(config)

# Función para obtener configuración (interface pública)
def get_app_config() -> AppConfig:
    """
    Obtener instancia de configuración de la aplicación
    
    Returns:
        AppConfig: Instancia de configuración cargada
    """
    return AppConfig()

# Instancia global de configuración (para compatibilidad)
try:
    _global_config = AppConfig()
    
    # Crear wrapper de compatibilidad
    settings = create_legacy_settings_wrapper(_global_config)
    
except Exception as e:
    # Fallback a configuración por defecto si hay problemas
    print(f"Warning: Error loading config, using defaults: {e}")
    _global_config = AppConfig()
    settings = create_legacy_settings_wrapper(_global_config)