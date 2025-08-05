"""
Retry Handler with Exponential Backoff
Maneja reintentos automáticos con backoff exponencial y jitter
"""

import asyncio
import random
import time
import logging
from typing import Any, Callable, Optional, Union, Type
from functools import wraps
from dataclasses import dataclass

@dataclass
class RetryResult:
    """Resultado de una operación con reintentos"""
    success: bool
    result: Any = None
    exception: Optional[Exception] = None
    attempts: int = 0
    total_time: float = 0.0

class RetryExhaustedError(Exception):
    """Excepción cuando se agotan todos los reintentos"""
    
    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts

class RetryHandler:
    """
    Manejador de reintentos con backoff exponencial y jitter
    
    Características:
    - Backoff exponencial configurable
    - Jitter para evitar thundering herd
    - Soporte para excepciones específicas
    - Logging detallado
    - Estadísticas de reintentos
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Union[Type[Exception], tuple] = Exception,
        name: str = "retry_handler"
    ):
        """
        Inicializar retry handler
        
        Args:
            max_retries: Número máximo de reintentos
            base_delay: Delay base en segundos
            max_delay: Delay máximo en segundos
            exponential_base: Base para el backoff exponencial
            jitter: Si aplicar jitter aleatorio
            exceptions: Excepciones que desencadenan retry
            name: Nombre para logging
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions
        self.name = name
        
        self._logger = logging.getLogger(f"retry.{name}")
        
        # Estadísticas
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_retries": 0,
            "max_attempts_reached": 0
        }
    
    async def execute(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Ejecutar función con reintentos
        
        Args:
            func: Función a ejecutar (sync o async)
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            RetryExhaustedError: Si se agotan todos los reintentos
        """
        self._stats["total_calls"] += 1
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self._logger.debug(
                    f"Intento {attempt + 1}/{self.max_retries + 1} para {self.name}"
                )
                
                # Ejecutar función
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Éxito - actualizar estadísticas
                self._stats["successful_calls"] += 1
                if attempt > 0:
                    self._stats["total_retries"] += attempt
                    self._logger.info(
                        f"Éxito en {self.name} después de {attempt} reintentos"
                    )
                
                return result
                
            except self.exceptions as e:
                last_exception = e
                
                # Si es el último intento, no hacer delay
                if attempt == self.max_retries:
                    break
                
                # Calcular delay y esperar
                delay = self._calculate_delay(attempt)
                self._logger.warning(
                    f"Intento {attempt + 1} falló para {self.name}: {e}. "
                    f"Reintentando en {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
            
            except Exception as e:
                # Excepciones no configuradas para retry - fallar inmediatamente
                self._logger.error(
                    f"Excepción no retryable en {self.name}: {e}"
                )
                self._stats["failed_calls"] += 1
                raise
        
        # Se agotaron todos los reintentos
        total_time = time.time() - start_time
        self._stats["failed_calls"] += 1
        self._stats["max_attempts_reached"] += 1
        self._stats["total_retries"] += self.max_retries
        
        self._logger.error(
            f"Reintentos agotados para {self.name} después de {self.max_retries + 1} intentos "
            f"en {total_time:.2f}s. Última excepción: {last_exception}"
        )
        
        raise RetryExhaustedError(
            f"Reintentos agotados para '{self.name}' después de {self.max_retries + 1} intentos",
            last_exception,
            self.max_retries + 1
        )
    
    async def execute_with_result(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> RetryResult:
        """
        Ejecutar función con reintentos y retornar resultado detallado
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            RetryResult: Resultado detallado con métricas
        """
        start_time = time.time()
        
        try:
            result = await self.execute(func, *args, **kwargs)
            return RetryResult(
                success=True,
                result=result,
                attempts=1,  # Se actualizará en execute() si hay reintentos
                total_time=time.time() - start_time
            )
        except RetryExhaustedError as e:
            return RetryResult(
                success=False,
                exception=e.last_exception,
                attempts=e.attempts,
                total_time=time.time() - start_time
            )
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calcular delay con backoff exponencial y jitter opcional
        
        Args:
            attempt: Número de intento (0-indexed)
            
        Returns:
            Delay en segundos
        """
        # Backoff exponencial
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Aplicar jitter si está habilitado
        if self.jitter:
            # Jitter entre 50% y 100% del delay calculado
            jitter_factor = 0.5 + random.random() * 0.5
            delay = delay * jitter_factor
        
        return delay
    
    def get_stats(self) -> dict:
        """Obtener estadísticas de reintentos"""
        stats = self._stats.copy()
        
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
            stats["avg_retries_per_call"] = stats["total_retries"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_retries_per_call"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Resetear estadísticas"""
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_retries": 0,
            "max_attempts_reached": 0
        }

def retry_on_failure(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Union[Type[Exception], tuple] = Exception,
    name: Optional[str] = None
):
    """
    Decorador para aplicar reintentos automáticos a funciones
    
    Args:
        max_retries: Número máximo de reintentos
        base_delay: Delay base en segundos
        max_delay: Delay máximo en segundos
        exponential_base: Base para backoff exponencial
        jitter: Si aplicar jitter aleatorio
        exceptions: Excepciones que desencadenan retry
        name: Nombre para logging (usa nombre de función si no se especifica)
    """
    def decorator(func: Callable) -> Callable:
        retry_name = name or func.__name__
        retry_handler = RetryHandler(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            exceptions=exceptions,
            name=retry_name
        )
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_handler.execute(func, *args, **kwargs)
            
            # Agregar métodos del retry handler al wrapper
            async_wrapper.retry_handler = retry_handler
            async_wrapper.get_stats = retry_handler.get_stats
            async_wrapper.reset_stats = retry_handler.reset_stats
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Para funciones sync, ejecutar en event loop
                try:
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(
                        retry_handler.execute(func, *args, **kwargs)
                    )
                except RuntimeError:
                    # Si no hay event loop, crear uno nuevo
                    return asyncio.run(
                        retry_handler.execute(func, *args, **kwargs)
                    )
            
            # Agregar métodos del retry handler al wrapper
            sync_wrapper.retry_handler = retry_handler
            sync_wrapper.get_stats = retry_handler.get_stats
            sync_wrapper.reset_stats = retry_handler.reset_stats
            return sync_wrapper
    
    return decorator

class RetryableFunction:
    """
    Wrapper para funciones que necesitan reintentos configurables
    Permite cambiar configuración de retry en runtime
    """
    
    def __init__(
        self,
        func: Callable,
        retry_handler: RetryHandler
    ):
        self.func = func
        self.retry_handler = retry_handler
        self._logger = logging.getLogger(f"retryable.{func.__name__}")
    
    async def __call__(self, *args, **kwargs) -> Any:
        """Ejecutar función con reintentos"""
        return await self.retry_handler.execute(self.func, *args, **kwargs)
    
    async def execute_with_result(self, *args, **kwargs) -> RetryResult:
        """Ejecutar con resultado detallado"""
        return await self.retry_handler.execute_with_result(self.func, *args, **kwargs)
    
    def update_config(
        self,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None
    ) -> None:
        """Actualizar configuración de reintentos"""
        if max_retries is not None:
            self.retry_handler.max_retries = max_retries
        if base_delay is not None:
            self.retry_handler.base_delay = base_delay
        if max_delay is not None:
            self.retry_handler.max_delay = max_delay
        
        self._logger.info(f"Configuración de retry actualizada para {self.func.__name__}")
    
    def get_stats(self) -> dict:
        """Obtener estadísticas"""
        return self.retry_handler.get_stats()
    
    def reset_stats(self) -> None:
        """Resetear estadísticas"""
        self.retry_handler.reset_stats()