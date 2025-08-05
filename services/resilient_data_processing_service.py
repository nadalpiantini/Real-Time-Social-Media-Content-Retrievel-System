"""
Versión resiliente del servicio de procesamiento de datos
Integra Circuit Breaker y Retry patterns para operaciones robustas
"""

from typing import List, Optional
import asyncio
import time
import logging

from services.data_processing_service import DataProcessingService
from models.post import RawPost, EmbeddedChunkedPost
from utils.advanced.circuit_breaker import CircuitBreaker, CircuitBreakerError
from utils.advanced.retry_handler import RetryHandler, RetryExhaustedError

class ResilientDataProcessingService(DataProcessingService):
    """
    Servicio de procesamiento de datos con patterns de resilencia integrados
    Extiende DataProcessingService con circuit breakers y retry logic
    """
    
    def __init__(self):
        super().__init__()
        self.service_name = "resilient_data_processing"
        
        # Circuit breakers para diferentes operaciones
        self.embedding_circuit = None
        self.processing_circuit = None
        
        # Retry handlers
        self.embedding_retry = None
        self.processing_retry = None
        
        # Estadísticas de resilencia
        self._resilience_stats = {
            "circuit_breaker_activations": 0,
            "retry_attempts": 0,
            "fallback_activations": 0,
            "degraded_mode_activations": 0
        }
    
    async def _initialize_impl(self) -> None:
        """Inicializar servicio con patterns de resilencia"""
        
        # Inicializar servicio base
        await super()._initialize_impl()
        
        # Configurar circuit breakers
        self.embedding_circuit = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            name="embedding_model",
            expected_exception=(Exception,)  # Cualquier excepción del modelo
        )
        
        self.processing_circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            name="post_processing",
            expected_exception=(Exception,)
        )
        
        # Configurar retry handlers
        self.embedding_retry = RetryHandler(
            max_retries=2,
            base_delay=0.5,
            max_delay=5.0,
            exceptions=(Exception,),
            name="embedding_processing"
        )
        
        self.processing_retry = RetryHandler(
            max_retries=1,
            base_delay=1.0,
            max_delay=10.0,
            exceptions=(Exception,),
            name="post_processing"
        )
        
        self.logger.info("Patterns de resilencia inicializados:")
        self.logger.info(f"  - Circuit breakers: embedding, processing")
        self.logger.info(f"  - Retry handlers: embedding (2 retries), processing (1 retry)")
    
    async def _additional_health_checks(self) -> dict:
        """Health checks incluyendo estado de resilencia"""
        
        # Health checks base
        base_checks = await super()._additional_health_checks()
        
        # Agregar checks de resilencia
        resilience_checks = {
            "circuit_breakers": {
                "embedding": {
                    "state": self.embedding_circuit.state.value if self.embedding_circuit else "not_initialized",
                    "stats": self.embedding_circuit.get_stats() if self.embedding_circuit else {}
                },
                "processing": {
                    "state": self.processing_circuit.state.value if self.processing_circuit else "not_initialized", 
                    "stats": self.processing_circuit.get_stats() if self.processing_circuit else {}
                }
            },
            "retry_handlers": {
                "embedding": self.embedding_retry.get_stats() if self.embedding_retry else {},
                "processing": self.processing_retry.get_stats() if self.processing_retry else {}
            },
            "resilience_stats": self._resilience_stats.copy()
        }
        
        # Combinar checks
        base_checks["resilience"] = resilience_checks
        return base_checks
    
    async def process_posts_batch_resilient(
        self,
        raw_posts: List[RawPost],
        progress_callback: Optional[callable] = None,
        enable_fallback: bool = True
    ) -> List[EmbeddedChunkedPost]:
        """
        Procesar batch con resilencia completa
        
        Args:
            raw_posts: Posts a procesar
            progress_callback: Callback de progreso
            enable_fallback: Si habilitar modo degradado en caso de fallos
            
        Returns:
            Lista de chunks procesados (puede estar parcialmente completa en modo degradado)
        """
        self._ensure_initialized()
        
        if not raw_posts:
            return []
        
        self.logger.info(f"Procesando batch resiliente de {len(raw_posts)} posts")
        
        # Intentar procesamiento normal primero
        try:
            return await self._process_with_circuit_breaker(
                raw_posts, progress_callback
            )
        except CircuitBreakerError as e:
            self._resilience_stats["circuit_breaker_activations"] += 1
            self.logger.warning(f"Circuit breaker activo: {e}")
            
            if enable_fallback:
                return await self._fallback_processing(raw_posts, progress_callback)
            else:
                raise
        except Exception as e:
            self.logger.error(f"Error en procesamiento resiliente: {e}")
            
            if enable_fallback:
                return await self._fallback_processing(raw_posts, progress_callback)
            else:
                raise
    
    async def _process_with_circuit_breaker(
        self,
        raw_posts: List[RawPost],
        progress_callback: Optional[callable] = None
    ) -> List[EmbeddedChunkedPost]:
        """Procesar posts usando circuit breaker para protección"""
        
        return await self.processing_circuit.call(
            self._process_with_retry,
            raw_posts,
            progress_callback
        )
    
    async def _process_with_retry(
        self,
        raw_posts: List[RawPost],
        progress_callback: Optional[callable] = None
    ) -> List[EmbeddedChunkedPost]:
        """Procesar posts con retry logic"""
        
        return await self.processing_retry.execute(
            self._process_batch_core,
            raw_posts,
            progress_callback
        )
    
    async def _process_batch_core(
        self,
        raw_posts: List[RawPost],
        progress_callback: Optional[callable] = None
    ) -> List[EmbeddedChunkedPost]:
        """Procesamiento core con resilencia en embeddings"""
        
        start_time = time.time()
        successful_chunks = []
        failed_posts = []
        
        # Procesar posts individualmente con resilencia
        for i, raw_post in enumerate(raw_posts):
            try:
                chunks = await self._process_single_post_resilient(raw_post)
                successful_chunks.extend(chunks)
                
                # Callback de progreso
                if progress_callback:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, progress_callback, i + 1, len(raw_posts)
                        )
                    except Exception:
                        pass  # No fallar por errores en callback
                        
            except Exception as e:
                failed_posts.append((raw_post.post_id, str(e)))
                self.logger.warning(f"Fallo procesando post {raw_post.post_id}: {e}")
        
        # Log resultados
        processing_time = time.time() - start_time
        self.logger.info(
            f"Batch resiliente completado: {len(successful_chunks)} chunks, "
            f"{len(failed_posts)} fallos, {processing_time:.2f}s"
        )
        
        # Si hay demasiados fallos, considerar como fallo del batch
        failure_rate = len(failed_posts) / len(raw_posts)
        if failure_rate > 0.5:  # >50% de fallos
            raise Exception(
                f"Tasa de fallos muy alta: {failure_rate:.1%} "
                f"({len(failed_posts)}/{len(raw_posts)} posts)"
            )
        
        return successful_chunks
    
    async def _process_single_post_resilient(self, raw_post: RawPost) -> List[EmbeddedChunkedPost]:
        """Procesar post individual con resilencia en embeddings"""
        
        # Usar circuit breaker y retry para operaciones de embedding
        return await self.embedding_circuit.call(
            self._process_single_post_with_embedding_retry,
            raw_post
        )
    
    async def _process_single_post_with_embedding_retry(self, raw_post: RawPost) -> List[EmbeddedChunkedPost]:
        """Procesar post con retry en embeddings"""
        
        return await self.embedding_retry.execute(
            self._process_single_post,  # Método del servicio base
            raw_post
        )
    
    async def _fallback_processing(
        self,
        raw_posts: List[RawPost],
        progress_callback: Optional[callable] = None
    ) -> List[EmbeddedChunkedPost]:
        """
        Modo de procesamiento degradado como fallback
        Procesa posts con configuración más conservadora
        """
        self._resilience_stats["fallback_activations"] += 1
        self.logger.warning("Activando modo de procesamiento degradado")
        
        # Procesar en batches más pequeños
        batch_size = min(5, len(raw_posts))
        successful_chunks = []
        
        for i in range(0, len(raw_posts), batch_size):
            batch = raw_posts[i:i + batch_size]
            
            try:
                # Usar procesamiento base sin circuit breakers
                chunks = await super().process_posts_batch(batch)
                successful_chunks.extend(chunks)
                
                # Delay entre batches para reducir carga
                if i + batch_size < len(raw_posts):
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                self.logger.error(f"Fallo en batch degradado {i}-{i+batch_size}: {e}")
                # Continuar con siguiente batch
                continue
        
        self.logger.info(
            f"Procesamiento degradado completado: {len(successful_chunks)} chunks "
            f"de {len(raw_posts)} posts originales"
        )
        
        return successful_chunks
    
    async def get_resilience_status(self) -> dict:
        """Obtener estado completo de resilencia"""
        
        status = {
            "service_name": self.service_name,
            "initialized": self._initialized,
            "resilience_stats": self._resilience_stats.copy(),
            "circuit_breakers": {},
            "retry_handlers": {}
        }
        
        # Estado de circuit breakers
        if self.embedding_circuit:
            status["circuit_breakers"]["embedding"] = {
                "state": self.embedding_circuit.state.value,
                "stats": self.embedding_circuit.get_stats()
            }
        
        if self.processing_circuit:
            status["circuit_breakers"]["processing"] = {
                "state": self.processing_circuit.state.value,
                "stats": self.processing_circuit.get_stats()
            }
        
        # Estado de retry handlers
        if self.embedding_retry:
            status["retry_handlers"]["embedding"] = self.embedding_retry.get_stats()
        
        if self.processing_retry:
            status["retry_handlers"]["processing"] = self.processing_retry.get_stats()
        
        return status
    
    def reset_resilience_stats(self) -> None:
        """Resetear estadísticas de resilencia"""
        
        self._resilience_stats = {
            "circuit_breaker_activations": 0,
            "retry_attempts": 0,
            "fallback_activations": 0,
            "degraded_mode_activations": 0
        }
        
        # Resetear stats de circuit breakers
        if self.embedding_circuit:
            self.embedding_circuit.reset()
        if self.processing_circuit:
            self.processing_circuit.reset()
        
        # Resetear stats de retry handlers
        if self.embedding_retry:
            self.embedding_retry.reset_stats()
        if self.processing_retry:
            self.processing_retry.reset_stats()
        
        self.logger.info("Estadísticas de resilencia reseteadas")