"""
Sistema de Error Recovery Production-Ready
Manejo avanzado de errores, fallbacks automáticos y graceful degradation
"""

import asyncio
import time
import logging
import traceback
from typing import Any, Dict, List, Optional, Callable, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import threading
from contextlib import asynccontextmanager
import inspect
from functools import wraps
import pickle
import hashlib

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Severidad de errores"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FallbackStrategy(Enum):
    """Estrategias de fallback"""
    SKIP = "skip"                    # Saltar elemento fallido
    DEFAULT_VALUE = "default_value"  # Usar valor por defecto
    CACHE_FALLBACK = "cache_fallback" # Usar valor del cache
    ALTERNATIVE_METHOD = "alternative_method"  # Método alternativo
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Degradación gradual

@dataclass
class ErrorInfo:
    """Información detallada de error"""
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    severity: ErrorSeverity
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "severity": self.severity.value,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "recovery_method": self.recovery_method,
            "retry_count": self.retry_count
        }

class ErrorPattern:
    """Patrón de error para detección automática"""
    
    def __init__(
        self,
        name: str,
        error_types: List[Type[Exception]],
        keywords: List[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        fallback_strategy: FallbackStrategy = FallbackStrategy.SKIP,
        max_retries: int = 3
    ):
        self.name = name
        self.error_types = error_types
        self.keywords = keywords or []
        self.severity = severity
        self.fallback_strategy = fallback_strategy
        self.max_retries = max_retries
    
    def matches(self, error: Exception, error_message: str) -> bool:
        """Verificar si el error coincide con este patrón"""
        # Verificar tipo de error
        if any(isinstance(error, error_type) for error_type in self.error_types):
            return True
        
        # Verificar palabras clave en mensaje
        if self.keywords:
            error_msg_lower = error_message.lower()
            return any(keyword.lower() in error_msg_lower for keyword in self.keywords)
        
        return False

class ErrorRecoveryManager:
    """Manager principal de recuperación de errores"""
    
    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        
        # Historial de errores
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_patterns: List[ErrorPattern] = []
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # Estadísticas
        self.error_stats = defaultdict(int)
        self.recovery_stats = defaultdict(int)
        self.last_error_time = defaultdict(float)
        
        # Lock para thread safety
        self.lock = threading.RLock()
        
        # Registro de patrones por defecto
        self._register_default_patterns()
    
    def _register_default_patterns(self):
        """Registrar patrones de error comunes"""
        
        # Errores de memoria
        self.register_pattern(ErrorPattern(
            name="memory_error",
            error_types=[MemoryError, OverflowError],
            keywords=["memory", "overflow", "out of memory"],
            severity=ErrorSeverity.HIGH,
            fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
            max_retries=1
        ))
        
        # Errores de red/conectividad
        self.register_pattern(ErrorPattern(
            name="network_error",
            error_types=[ConnectionError, TimeoutError],
            keywords=["connection", "timeout", "network", "unreachable"],
            severity=ErrorSeverity.MEDIUM,
            fallback_strategy=FallbackStrategy.CACHE_FALLBACK,
            max_retries=3
        ))
        
        # Errores de E/O
        self.register_pattern(ErrorPattern(
            name="io_error",
            error_types=[IOError, OSError, FileNotFoundError],
            keywords=["file not found", "permission denied", "disk full"],
            severity=ErrorSeverity.MEDIUM,
            fallback_strategy=FallbackStrategy.ALTERNATIVE_METHOD,
            max_retries=2
        ))
        
        # Errores de validación/datos
        self.register_pattern(ErrorPattern(
            name="data_error",
            error_types=[ValueError, TypeError, KeyError, AttributeError],
            keywords=["invalid", "missing", "malformed"],
            severity=ErrorSeverity.LOW,
            fallback_strategy=FallbackStrategy.DEFAULT_VALUE,
            max_retries=1
        ))
        
        # Errores críticos del sistema
        self.register_pattern(ErrorPattern(
            name="system_critical",
            error_types=[SystemError, RuntimeError],
            keywords=["critical", "fatal", "shutdown"],
            severity=ErrorSeverity.CRITICAL,
            fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
            max_retries=0
        ))
    
    def register_pattern(self, pattern: ErrorPattern):
        """Registrar patrón de error"""
        with self.lock:
            self.error_patterns.append(pattern)
            logger.info(f"✅ Error pattern registered: {pattern.name}")
    
    def register_fallback_handler(self, strategy: str, handler: Callable):
        """Registrar manejador de fallback"""
        with self.lock:
            self.fallback_handlers[strategy] = handler
            logger.info(f"✅ Fallback handler registered: {strategy}")
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        operation_name: str = "unknown"
    ) -> Tuple[bool, Any]:
        """
        Manejar error con recovery automático
        Returns: (recovery_successful, recovered_value)
        """
        error_id = self._generate_error_id(error, context)
        error_message = str(error)
        
        # Crear info de error
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=error_message,
            stack_trace=traceback.format_exc(),
            context=context or {},
            severity=ErrorSeverity.MEDIUM  # Por defecto, se actualizará
        )
        
        # Encontrar patrón coincidente
        matching_pattern = self._find_matching_pattern(error, error_message)
        if matching_pattern:
            error_info.severity = matching_pattern.severity
            logger.info(f"🎯 Error pattern matched: {matching_pattern.name}")
        
        # Registrar error
        with self.lock:
            self.error_history.append(error_info)
            self.error_stats[error_info.error_type] += 1
            self.last_error_time[operation_name] = time.time()
        
        # Intentar recovery
        recovery_successful = False
        recovered_value = None
        
        if matching_pattern:
            try:
                error_info.recovery_attempted = True
                
                # Aplicar estrategia de fallback
                recovered_value = await self._apply_fallback_strategy(
                    matching_pattern.fallback_strategy,
                    error,
                    context,
                    operation_name
                )
                
                recovery_successful = recovered_value is not None
                error_info.recovery_successful = recovery_successful
                error_info.recovery_method = matching_pattern.fallback_strategy.value
                
                if recovery_successful:
                    with self.lock:
                        self.recovery_stats[matching_pattern.name] += 1
                    logger.info(f"✅ Error recovery successful: {matching_pattern.name}")
                
            except Exception as recovery_error:
                logger.error(f"❌ Error recovery failed: {recovery_error}")
                error_info.recovery_successful = False
        
        # Log según severidad
        self._log_error(error_info, matching_pattern)
        
        return recovery_successful, recovered_value
    
    def _find_matching_pattern(self, error: Exception, error_message: str) -> Optional[ErrorPattern]:
        """Encontrar patrón de error coincidente"""
        with self.lock:
            for pattern in self.error_patterns:
                if pattern.matches(error, error_message):
                    return pattern
        return None
    
    async def _apply_fallback_strategy(
        self,
        strategy: FallbackStrategy,
        error: Exception,
        context: Dict[str, Any],
        operation_name: str
    ) -> Any:
        """Aplicar estrategia de fallback"""
        
        if strategy == FallbackStrategy.SKIP:
            return None
        
        elif strategy == FallbackStrategy.DEFAULT_VALUE:
            # Buscar valor por defecto en contexto
            return context.get("default_value")
        
        elif strategy == FallbackStrategy.CACHE_FALLBACK:
            # Intentar obtener del cache
            cache_key = context.get("cache_key")
            if cache_key and hasattr(self, 'cache'):
                return self.cache.get(cache_key)
            return None
        
        elif strategy == FallbackStrategy.ALTERNATIVE_METHOD:
            # Usar método alternativo si está disponible
            alt_method = context.get("alternative_method")
            if alt_method and callable(alt_method):
                try:
                    if asyncio.iscoroutinefunction(alt_method):
                        return await alt_method()
                    else:
                        return alt_method()
                except Exception as e:
                    logger.warning(f"Alternative method failed: {e}")
            return None
        
        elif strategy == FallbackStrategy.GRACEFUL_DEGRADATION:
            # Degradación gradual - reducir funcionalidad
            degraded_handler = self.fallback_handlers.get("graceful_degradation")
            if degraded_handler:
                return await self._call_handler(degraded_handler, error, context)
            return None
        
        # Handler personalizado
        handler = self.fallback_handlers.get(strategy.value)
        if handler:
            return await self._call_handler(handler, error, context)
        
        return None
    
    async def _call_handler(self, handler: Callable, error: Exception, context: Dict[str, Any]) -> Any:
        """Llamar handler de forma segura"""
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(error, context)
            else:
                return handler(error, context)
        except Exception as e:
            logger.error(f"Fallback handler failed: {e}")
            return None
    
    def _generate_error_id(self, error: Exception, context: Dict[str, Any]) -> str:
        """Generar ID único para error"""
        content = f"{type(error).__name__}:{str(error)}:{str(context)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _log_error(self, error_info: ErrorInfo, pattern: Optional[ErrorPattern]):
        """Log error según severidad"""
        base_msg = f"{error_info.error_type}: {error_info.error_message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"🚨 CRITICAL ERROR: {base_msg}")
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(f"❌ HIGH SEVERITY: {base_msg}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"⚠️ MEDIUM SEVERITY: {base_msg}")
        else:
            logger.info(f"ℹ️ LOW SEVERITY: {base_msg}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de errores"""
        with self.lock:
            recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Última hora
            
            return {
                "total_errors": len(self.error_history),
                "recent_errors_1h": len(recent_errors),
                "error_types": dict(self.error_stats),
                "recovery_stats": dict(self.recovery_stats),
                "recovery_rate": (
                    sum(self.recovery_stats.values()) / len(self.error_history) * 100
                    if self.error_history else 0
                ),
                "severity_distribution": self._get_severity_distribution(),
                "error_patterns_count": len(self.error_patterns),
                "last_errors": [e.to_dict() for e in list(self.error_history)[-5:]]
            }
    
    def _get_severity_distribution(self) -> Dict[str, int]:
        """Obtener distribución de severidad"""
        distribution = defaultdict(int)
        for error_info in self.error_history:
            distribution[error_info.severity.value] += 1
        return dict(distribution)
    
    def clear_error_history(self):
        """Limpiar historial de errores"""
        with self.lock:
            self.error_history.clear()
            self.error_stats.clear()
            self.recovery_stats.clear()
            self.last_error_time.clear()
        logger.info("🧹 Error history cleared")

class ResilientExecutor:
    """Ejecutor resiliente con manejo automático de errores"""
    
    def __init__(self, error_manager: ErrorRecoveryManager):
        self.error_manager = error_manager
    
    async def execute_with_recovery(
        self,
        func: Callable,
        *args,
        context: Dict[str, Any] = None,
        operation_name: str = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> Any:
        """
        Ejecutar función con recovery automático
        """
        operation_name = operation_name or func.__name__
        context = context or {}
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Ejecutar función
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"🔄 Attempt {attempt + 1}/{max_retries + 1} failed for {operation_name}: {e}")
                
                # Intentar recovery
                recovery_successful, recovered_value = await self.error_manager.handle_error(
                    e, context, operation_name
                )
                
                if recovery_successful:
                    logger.info(f"✅ Recovery successful for {operation_name}")
                    return recovered_value
                
                # Si no es el último intento, esperar antes de retry
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        # Todos los intentos fallaron
        logger.error(f"❌ All attempts failed for {operation_name}")
        raise last_error

def resilient_operation(
    operation_name: str = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_strategy: FallbackStrategy = FallbackStrategy.SKIP,
    default_value: Any = None
):
    """
    Decorador para hacer operaciones resilientes
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_manager = get_global_error_manager()
            executor = ResilientExecutor(error_manager)
            
            context = {
                "default_value": default_value,
                "fallback_strategy": fallback_strategy
            }
            
            return await executor.execute_with_recovery(
                func,
                *args,
                context=context,
                operation_name=operation_name or func.__name__,
                max_retries=max_retries,
                retry_delay=retry_delay,
                **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Para funciones síncronas, convertir a async temporalmente
            async def async_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Retornar wrapper apropiado según el tipo de función
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@asynccontextmanager
async def error_recovery_context(
    operation_name: str,
    context: Dict[str, Any] = None,
    error_manager: ErrorRecoveryManager = None
):
    """Context manager para manejo de errores"""
    if error_manager is None:
        error_manager = get_global_error_manager()
    
    try:
        yield
    except Exception as e:
        recovery_successful, recovered_value = await error_manager.handle_error(
            e, context, operation_name
        )
        
        if not recovery_successful:
            raise

# Global instance
_global_error_manager: Optional[ErrorRecoveryManager] = None

def get_global_error_manager() -> ErrorRecoveryManager:
    """Obtener instancia global del error manager"""
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = ErrorRecoveryManager()
    return _global_error_manager

def configure_global_error_handling():
    """Configurar manejo global de errores no capturados"""
    error_manager = get_global_error_manager()
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Manejar excepciones no capturadas"""
        if issubclass(exc_type, KeyboardInterrupt):
            return  # No manejar Ctrl+C
        
        logger.critical("🚨 Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Intentar recovery asíncrono
        try:
            asyncio.run(error_manager.handle_error(
                exc_value,
                {"uncaught": True},
                "uncaught_exception"
            ))
        except Exception as e:
            logger.error(f"Failed to handle uncaught exception: {e}")
    
    import sys
    sys.excepthook = handle_exception