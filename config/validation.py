"""
Sistema de validación de configuración y salud del sistema
Verifica que todos los componentes estén correctamente configurados
"""

import os
import logging
from typing import Dict, Any, List
from config.app_config import get_app_config, AppConfig

logger = logging.getLogger(__name__)

class SystemValidator:
    """
    Validador de sistema para verificar configuración y dependencias
    """
    
    def __init__(self):
        self.config = get_app_config()
        self.validation_results = {}
    
    async def validate_system(self) -> Dict[str, Any]:
        """
        Ejecutar validación completa del sistema
        
        Returns:
            Dict con resultados de validación
        """
        results = {
            "system_healthy": True,
            "validations": {},
            "warnings": [],
            "errors": []
        }
        
        # Ejecutar validaciones
        validations = [
            ("config", self._validate_config),
            ("environment", self._validate_environment),
            ("dependencies", self._validate_dependencies),
            ("directories", self._validate_directories),
        ]
        
        for validation_name, validation_func in validations:
            try:
                validation_result = await validation_func()
                results["validations"][validation_name] = validation_result
                
                if not validation_result.get("valid", False):
                    results["system_healthy"] = False
                    if "errors" in validation_result:
                        results["errors"].extend(validation_result["errors"])
                    if "warnings" in validation_result:
                        results["warnings"].extend(validation_result["warnings"])
                        
            except Exception as e:
                logger.error(f"Error en validación {validation_name}: {e}")
                results["system_healthy"] = False
                results["errors"].append(f"Validation {validation_name} failed: {str(e)}")
        
        return results
    
    async def _validate_config(self) -> Dict[str, Any]:
        """Validar configuración de la aplicación"""
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Validar configuración de ML
            if not self.config.ml.embedding_model_name:
                result["errors"].append("Embedding model name not configured")
                result["valid"] = False
            
            if self.config.ml.embedding_dimension <= 0:
                result["errors"].append("Invalid embedding dimension")
                result["valid"] = False
            
            # Validar configuración de base de datos
            if not self.config.database.collection_name:
                result["warnings"].append("No collection name configured, using default")
            
            # Validar configuración de Qdrant
            if not self.config.database.qdrant_url:
                result["warnings"].append("Qdrant URL not configured, will use default")
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Config validation error: {str(e)}")
        
        return result
    
    async def _validate_environment(self) -> Dict[str, Any]:
        """Validar variables de entorno"""
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Variables opcionales pero recomendadas
        optional_vars = [
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "QDRANT_API_KEY",
            "EMBEDDING_MODEL_DEVICE"
        ]
        
        for var in optional_vars:
            if not os.getenv(var):
                result["warnings"].append(f"Environment variable {var} not set")
        
        return result
    
    async def _validate_dependencies(self) -> Dict[str, Any]:
        """Validar dependencias del sistema"""
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validar importes críticos
        critical_imports = [
            ("sentence_transformers", "SentenceTransformer"),
            ("qdrant_client", "QdrantClient"),
            ("streamlit", None),
            ("torch", None),
            ("numpy", None),
            ("pandas", None)
        ]
        
        for module_name, class_name in critical_imports:
            try:
                module = __import__(module_name)
                if class_name:
                    getattr(module, class_name)
            except ImportError:
                result["errors"].append(f"Critical dependency {module_name} not available")
                result["valid"] = False
            except AttributeError:
                result["errors"].append(f"Class {class_name} not found in {module_name}")
                result["valid"] = False
        
        return result
    
    async def _validate_directories(self) -> Dict[str, Any]:
        """Validar directorios necesarios"""
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Directorios que deben existir
        required_dirs = [
            "data",
            "config",
            "services",
            "utils",
            "models"
        ]
        
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                result["warnings"].append(f"Directory {dir_name} does not exist")
        
        # Verificar permisos de escritura en data/
        if os.path.exists("data"):
            if not os.access("data", os.W_OK):
                result["errors"].append("Data directory is not writable")
                result["valid"] = False
        
        return result
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de salud del sistema
        
        Returns:
            Dict con resumen de salud
        """
        return {
            "config_loaded": bool(self.config),
            "ml_config": {
                "model_name": self.config.ml.embedding_model_name,
                "dimension": self.config.ml.embedding_dimension,
                "device": self.config.ml.device
            },
            "database_config": {
                "qdrant_url": self.config.database.qdrant_url,
                "collection_name": self.config.database.collection_name,
                "use_supabase": self.config.database.use_supabase
            }
        }