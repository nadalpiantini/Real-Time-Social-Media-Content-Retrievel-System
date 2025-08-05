"""
Contenedor de servicios con dependency injection
Gestiona el ciclo de vida de todos los servicios del sistema
"""

from typing import Dict, Any, Optional, TypeVar, Type
import asyncio
import logging
from contextlib import asynccontextmanager

from services.base_service import BaseService
from services.data_processing_service import DataProcessingService
from services.resilient_data_processing_service import ResilientDataProcessingService

T = TypeVar('T', bound=BaseService)

class ServiceContainer:
    """
    Contenedor IoC para gestión de servicios
    Implementa patrón Singleton y gestión automática del ciclo de vida
    """
    
    _instance: Optional['ServiceContainer'] = None
    _initialization_lock = asyncio.Lock()
    
    def __new__(cls) -> 'ServiceContainer':
        """Implementación Singleton thread-safe"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized_container = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized_container') and self._initialized_container:
            return
        
        self.logger = logging.getLogger("service_container")
        self._services: Dict[str, BaseService] = {}
        self._service_types: Dict[str, Type[BaseService]] = {}
        self._initialized_container = False
        self._container_lock = asyncio.Lock()
        
        # Registrar servicios disponibles
        self._register_default_services()
    
    def _register_default_services(self) -> None:
        """Registrar servicios por defecto del sistema"""
        self._service_types = {
            "data_processing": DataProcessingService,
            "resilient_data_processing": ResilientDataProcessingService,
            # Aquí se pueden agregar más servicios en el futuro
            # "search": SearchService,
            # "ingestion": IngestionService,
        }
    
    async def initialize(self) -> None:
        """
        Inicializar el contenedor y todos los servicios registrados
        Thread-safe y idempotente
        """
        if self._initialized_container:
            return
        
        async with self._initialization_lock:
            if self._initialized_container:  # Double-check
                return
            
            self.logger.info("Inicializando contenedor de servicios...")
            
            # Crear instancias de todos los servicios registrados
            for service_name, service_class in self._service_types.items():
                try:
                    self.logger.info(f"Creando servicio: {service_name}")
                    service_instance = service_class()
                    self._services[service_name] = service_instance
                except Exception as e:
                    self.logger.error(f"Error creando servicio {service_name}: {e}")
                    raise
            
            # Inicializar todos los servicios en paralelo
            init_tasks = [
                service.initialize() 
                for service in self._services.values()
            ]
            
            try:
                await asyncio.gather(*init_tasks)
                self._initialized_container = True
                self.logger.info(f"Contenedor inicializado con {len(self._services)} servicios")
            except Exception as e:
                self.logger.error(f"Error inicializando servicios: {e}")
                # Limpiar servicios parcialmente inicializados
                await self._cleanup_services()
                raise
    
    def get_service(self, service_name: str) -> BaseService:
        """
        Obtener servicio por nombre con verificación de estado
        
        Args:
            service_name: Nombre del servicio a obtener
            
        Returns:
            Instancia del servicio solicitado
            
        Raises:
            RuntimeError: Si el contenedor no está inicializado
            ValueError: Si el servicio no existe
        """
        if not self._initialized_container:
            raise RuntimeError(
                "Contenedor no inicializado. Llama a initialize() primero."
            )
        
        if service_name not in self._services:
            available_services = list(self._services.keys())
            raise ValueError(
                f"Servicio '{service_name}' no encontrado. "
                f"Servicios disponibles: {available_services}"
            )
        
        return self._services[service_name]
    
    def get_service_typed(self, service_name: str, service_type: Type[T]) -> T:
        """
        Obtener servicio con tipo específico (type-safe)
        
        Args:
            service_name: Nombre del servicio
            service_type: Tipo esperado del servicio
            
        Returns:
            Instancia del servicio con el tipo correcto
        """
        service = self.get_service(service_name)
        
        if not isinstance(service, service_type):
            raise TypeError(
                f"Servicio '{service_name}' es de tipo {type(service)}, "
                f"se esperaba {service_type}"
            )
        
        return service
    
    def has_service(self, service_name: str) -> bool:
        """Verificar si un servicio está disponible"""
        return service_name in self._services
    
    def list_services(self) -> Dict[str, str]:
        """Listar todos los servicios disponibles con sus tipos"""
        return {
            name: service.__class__.__name__ 
            for name, service in self._services.items()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verificar estado de salud de todos los servicios
        
        Returns:
            dict: Estado de salud completo del contenedor
        """
        health_status = {
            "container_initialized": self._initialized_container,
            "services_count": len(self._services),
            "services": {}
        }
        
        if self._initialized_container:
            # Ejecutar health checks de todos los servicios en paralelo
            health_tasks = {
                name: service.health_check()
                for name, service in self._services.items()
            }
            
            health_results = await asyncio.gather(
                *health_tasks.values(), 
                return_exceptions=True
            )
            
            # Procesar resultados
            for (service_name, _), result in zip(health_tasks.items(), health_results):
                if isinstance(result, Exception):
                    health_status["services"][service_name] = {
                        "status": "error",
                        "error": str(result)
                    }
                else:
                    health_status["services"][service_name] = result
        
        # Determinar estado general
        if health_status["container_initialized"]:
            all_healthy = all(
                service_health.get("status") == "healthy"
                for service_health in health_status["services"].values()
                if isinstance(service_health, dict)
            )
            health_status["overall_status"] = "healthy" if all_healthy else "degraded"
        else:
            health_status["overall_status"] = "not_initialized"
        
        return health_status
    
    async def register_service(
        self, 
        service_name: str, 
        service_instance: BaseService
    ) -> None:
        """
        Registrar un servicio adicional en runtime
        
        Args:
            service_name: Nombre único del servicio
            service_instance: Instancia del servicio a registrar
        """
        async with self._container_lock:
            if service_name in self._services:
                raise ValueError(f"Servicio '{service_name}' ya está registrado")
            
            # Inicializar el servicio si el contenedor ya está inicializado
            if self._initialized_container:
                await service_instance.initialize()
            
            self._services[service_name] = service_instance
            self.logger.info(f"Servicio '{service_name}' registrado dinámicamente")
    
    async def shutdown(self) -> None:
        """
        Cerrar todos los servicios y el contenedor limpiamente
        """
        if not self._initialized_container:
            return
        
        self.logger.info("Cerrando contenedor de servicios...")
        await self._cleanup_services()
        self._initialized_container = False
        self.logger.info("Contenedor cerrado")
    
    async def _cleanup_services(self) -> None:
        """Cerrar todos los servicios de forma segura"""
        if not self._services:
            return
        
        # Cerrar servicios en paralelo
        shutdown_tasks = [
            service.shutdown() 
            for service in self._services.values()
        ]
        
        # Usar gather con return_exceptions para no fallar si un servicio falla
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Log errores de cierre
        for service_name, result in zip(self._services.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Error cerrando servicio {service_name}: {result}")
        
        self._services.clear()
    
    @asynccontextmanager
    async def service_context(self, service_name: str):
        """
        Context manager para uso temporal de un servicio
        Garantiza que el servicio esté inicializado durante el uso
        """
        service = self.get_service(service_name)
        try:
            yield service
        except Exception as e:
            self.logger.error(f"Error usando servicio {service_name}: {e}")
            raise
    
    def __del__(self):
        """Cleanup cuando el contenedor es destruido"""
        if hasattr(self, '_initialized_container') and self._initialized_container:
            # No podemos hacer async cleanup en __del__, solo log warning
            logging.getLogger("service_container").warning(
                "ServiceContainer destruido sin shutdown explícito - "
                "llama a shutdown() para cleanup apropiado"
            )

# Instancia global del contenedor (Singleton)
service_container = ServiceContainer()