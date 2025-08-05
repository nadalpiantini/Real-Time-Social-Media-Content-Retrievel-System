"""
Clase base para todos los servicios del sistema
Proporciona funcionalidad común como logging, health checks e inicialización
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import asyncio
from config.app_config import get_app_config

class BaseService(ABC):
    """Clase base abstracta para todos los servicios"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.config = get_app_config()
        self.logger = logging.getLogger(f"service.{service_name}")
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """
        Inicializar el servicio de forma thread-safe
        Solo se ejecuta una vez por instancia
        """
        if self._initialized:
            return
        
        async with self._initialization_lock:
            if self._initialized:  # Double-check after acquiring lock
                return
            
            self.logger.info(f"Inicializando servicio '{self.service_name}'...")
            try:
                await self._initialize_impl()
                self._initialized = True
                self.logger.info(f"Servicio '{self.service_name}' inicializado exitosamente")
            except Exception as e:
                self.logger.error(f"Error inicializando servicio '{self.service_name}': {e}")
                raise
    
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """
        Implementación específica de inicialización del servicio
        Debe ser implementada por cada servicio concreto
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verificar estado de salud del servicio
        
        Returns:
            dict: Estado de salud con métricas relevantes
        """
        health_status = {
            "service": self.service_name,
            "initialized": self._initialized,
            "status": "healthy" if self._initialized else "not_initialized",
            "config_loaded": self.config is not None
        }
        
        if self._initialized:
            try:
                # Permitir que servicios específicos agreguen checks adicionales
                additional_checks = await self._additional_health_checks()
                health_status.update(additional_checks)
            except Exception as e:
                self.logger.warning(f"Error en health checks adicionales: {e}")
                health_status["status"] = "degraded"
                health_status["health_check_error"] = str(e)
        
        return health_status
    
    async def _additional_health_checks(self) -> Dict[str, Any]:
        """
        Health checks adicionales específicos del servicio
        Puede ser sobrescrito por servicios específicos
        
        Returns:
            dict: Métricas adicionales de salud
        """
        return {}
    
    def _ensure_initialized(self) -> None:
        """
        Verificar que el servicio está inicializado antes de usar
        Lanza excepción si no está inicializado
        """
        if not self._initialized:
            raise RuntimeError(
                f"Servicio '{self.service_name}' no está inicializado. "
                f"Llama a initialize() primero."
            )
    
    async def shutdown(self) -> None:
        """
        Cerrar el servicio limpiamente
        Puede ser sobrescrito por servicios específicos
        """
        if self._initialized:
            self.logger.info(f"Cerrando servicio '{self.service_name}'...")
            try:
                await self._shutdown_impl()
                self._initialized = False
                self.logger.info(f"Servicio '{self.service_name}' cerrado")
            except Exception as e:
                self.logger.error(f"Error cerrando servicio '{self.service_name}': {e}")
                raise
    
    async def _shutdown_impl(self) -> None:
        """
        Implementación específica de cierre del servicio
        Puede ser sobrescrita por servicios específicos
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.service_name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(service_name='{self.service_name}', initialized={self._initialized})"