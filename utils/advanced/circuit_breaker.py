"""
Circuit Breaker Pattern Implementation
Previene fallos en cascada y permite recuperación automática
"""

import time
import asyncio
from typing import Any, Callable, Optional, Type, Union
from enum import Enum
import logging
from functools import wraps

class CircuitState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"        # Operación normal
    OPEN = "open"           # Fallando rápido
    HALF_OPEN = "half_open"  # Probando recuperación

class CircuitBreakerError(Exception):
    """Excepción lanzada cuando el circuit está abierto"""
    pass

class CircuitBreaker:
    """
    Implementación del patrón Circuit Breaker
    
    Características:
    - Estados: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing) -> CLOSED
    - Configuración flexible de umbrales y timeouts
    - Soporte para async y sync functions
    - Logging detallado para debugging
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Union[Type[Exception], tuple] = Exception,
        name: str = "circuit_breaker",
        half_open_max_calls: int = 3
    ):
        """
        Inicializar circuit breaker
        
        Args:
            failure_threshold: Número de fallos antes de abrir el circuit
            recovery_timeout: Segundos antes de intentar recuperación
            expected_exception: Excepciones que cuenta como fallos
            name: Nombre del circuit para logging
            half_open_max_calls: Máximo calls en estado HALF_OPEN
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        self.half_open_max_calls = half_open_max_calls
        
        # Estado interno
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        # Logging
        self._logger = logging.getLogger(f"circuit_breaker.{name}")
        
        # Estadísticas
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "circuit_open_count": 0,
            "recovery_attempts": 0
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecutar función con circuit breaker protection
        
        Args:
            func: Función a ejecutar (sync o async)
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            CircuitBreakerError: Si el circuit está abierto
            Exception: Excepción original de la función
        """
        async with self._lock:
            self._stats["total_calls"] += 1
            
            # Verificar estado del circuit
            await self._update_state()
            
            if self._state == CircuitState.OPEN:
                self._logger.warning(f"Circuit {self.name} está OPEN - rechazando llamada")
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' está abierto. "
                    f"Próximo intento en {self._time_until_retry():.1f} segundos"
                )
            
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._logger.warning(f"Circuit {self.name} HALF_OPEN ha alcanzado límite de calls")
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' en estado HALF_OPEN ha alcanzado "
                        f"el límite de {self.half_open_max_calls} llamadas"
                    )
                self._half_open_calls += 1
        
        # Ejecutar función fuera del lock para evitar bloqueos largos
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Función exitosa
            await self._on_success()
            return result
            
        except self.expected_exception as e:
            await self._on_failure(e)
            raise
        except Exception as e:
            # Excepciones no esperadas no cuentan como fallos del circuit
            self._logger.debug(f"Excepción no esperada en circuit {self.name}: {e}")
            raise
    
    async def _update_state(self) -> None:
        """Actualizar estado del circuit basado en tiempo y condiciones"""
        
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._stats["recovery_attempts"] += 1
                self._logger.info(f"Circuit {self.name} pasando a HALF_OPEN para test de recuperación")
    
    def _should_attempt_reset(self) -> bool:
        """Verificar si es tiempo de intentar recuperación"""
        if not self._last_failure_time:
            return False
        
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def _time_until_retry(self) -> float:
        """Calcular tiempo restante hasta próximo intento"""
        if not self._last_failure_time:
            return 0.0
        
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.recovery_timeout - elapsed)
    
    async def _on_success(self) -> None:
        """Manejar llamada exitosa"""
        async with self._lock:
            self._stats["successful_calls"] += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self._logger.debug(
                    f"Circuit {self.name} HALF_OPEN: éxito {self._success_count}"
                )
                
                # Si tenemos suficientes éxitos, cerrar el circuit
                if self._success_count >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    self._logger.info(f"Circuit {self.name} RECUPERADO -> CLOSED")
            
            elif self._state == CircuitState.CLOSED:
                # Reset failure count en estado normal
                if self._failure_count > 0:
                    self._failure_count = 0
                    self._logger.debug(f"Circuit {self.name} failures reseteados por éxito")
    
    async def _on_failure(self, exception: Exception) -> None:
        """Manejar fallo de llamada"""
        async with self._lock:
            self._stats["failed_calls"] += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            self._logger.warning(
                f"Circuit {self.name} fallo #{self._failure_count}: {exception}"
            )
            
            if self._state == CircuitState.HALF_OPEN:
                # Un fallo en HALF_OPEN vuelve a abrir el circuit
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._half_open_calls = 0
                self._stats["circuit_open_count"] += 1
                self._logger.error(
                    f"Circuit {self.name} falló en HALF_OPEN -> OPEN de nuevo"
                )
            
            elif self._state == CircuitState.CLOSED:
                # Verificar si alcanzamos el umbral para abrir
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._stats["circuit_open_count"] += 1
                    self._logger.error(
                        f"Circuit {self.name} ABIERTO después de {self._failure_count} fallos"
                    )
    
    @property
    def state(self) -> CircuitState:
        """Estado actual del circuit"""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Verificar si el circuit está cerrado (operación normal)"""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Verificar si el circuit está abierto (fallando rápido)"""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Verificar si el circuit está en estado de prueba"""
        return self._state == CircuitState.HALF_OPEN
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del circuit breaker"""
        stats = self._stats.copy()
        stats.update({
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "half_open_calls": self._half_open_calls,
            "time_until_retry": self._time_until_retry(),
            "failure_rate": (
                self._stats["failed_calls"] / max(1, self._stats["total_calls"])
            )
        })
        return stats
    
    def reset(self) -> None:
        """
        Reset manual del circuit breaker
        Útil para testing o recuperación manual
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
        self._logger.info(f"Circuit {self.name} reseteado manualmente")

def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: Union[Type[Exception], tuple] = Exception,
    name: Optional[str] = None
):
    """
    Decorador para aplicar circuit breaker a funciones
    
    Args:
        failure_threshold: Fallos antes de abrir circuit
        recovery_timeout: Segundos antes de intentar recuperación
        expected_exception: Excepciones que cuentan como fallos
        name: Nombre del circuit (usa nombre de función si no se especifica)
    """
    def decorator(func: Callable) -> Callable:
        circuit_name = name or func.__name__
        circuit = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=circuit_name
        )
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await circuit.call(func, *args, **kwargs)
            
            # Agregar métodos del circuit al wrapper
            async_wrapper.circuit = circuit
            async_wrapper.get_stats = circuit.get_stats
            async_wrapper.reset = circuit.reset
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Para funciones sync, necesitamos ejecutar en event loop
                try:
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(circuit.call(func, *args, **kwargs))
                except RuntimeError:
                    # Si no hay event loop, crear uno nuevo
                    return asyncio.run(circuit.call(func, *args, **kwargs))
            
            # Agregar métodos del circuit al wrapper
            sync_wrapper.circuit = circuit
            sync_wrapper.get_stats = circuit.get_stats
            sync_wrapper.reset = circuit.reset
            return sync_wrapper
    
    return decorator