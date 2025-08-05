"""
Servicio de procesamiento de datos con soporte asíncrono
Maneja la pipeline completa: Raw -> Cleaned -> Chunked -> Embedded
"""

from typing import List, Optional, AsyncGenerator
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from services.base_service import BaseService
from models.post import RawPost, CleanedPost, ChunkedPost, EmbeddedChunkedPost
from utils.embedding import EmbeddingModelSingleton


class DataProcessingService(BaseService):
    """Servicio para procesamiento asíncrono de datos de posts"""
    
    def __init__(self):
        super().__init__("data_processing")
        self.embedding_model = None
        self.semaphore = None
        self.executor = None
        self._processing_stats = {
            "posts_processed": 0,
            "chunks_generated": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
    
    async def _initialize_impl(self) -> None:
        """Inicializar modelos ML y configurar concurrencia"""
        
        # Configurar semáforo para control de concurrencia
        max_concurrent = getattr(self.config, 'max_concurrent_tasks', 10)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Configurar thread pool para operaciones síncronas (ML models)
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{self.service_name}_")
        
        # Inicializar modelo de embeddings en thread pool
        self.logger.info("Cargando modelo de embeddings...")
        loop = asyncio.get_event_loop()
        self.embedding_model = await loop.run_in_executor(
            self.executor, 
            EmbeddingModelSingleton
        )
        
        self.logger.info(f"Modelo cargado: {self.embedding_model.model_id}")
        self.logger.info(f"Concurrencia configurada: {max_concurrent} tareas simultáneas")
    
    async def _additional_health_checks(self) -> dict:
        """Health checks específicos del servicio de procesamiento"""
        
        checks = {
            "embedding_model_loaded": self.embedding_model is not None,
            "concurrency_limit": self.semaphore._value if self.semaphore else 0,
            "thread_pool_active": self.executor is not None and not self.executor._shutdown,
            "processing_stats": self._processing_stats.copy()
        }
        
        # Test básico del modelo si está cargado
        if self.embedding_model:
            try:
                # Test rápido con texto pequeño
                loop = asyncio.get_event_loop()
                test_result = await loop.run_in_executor(
                    self.executor,
                    lambda: self.embedding_model("test", to_list=True)
                )
                checks["model_test"] = "pass" if len(test_result) > 0 else "fail"
            except Exception as e:
                checks["model_test"] = f"error: {e}"
        
        return checks
    
    async def _shutdown_impl(self) -> None:
        """Cerrar thread pool y limpiar recursos"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
    
    async def process_posts_batch(
        self, 
        raw_posts: List[RawPost],
        progress_callback: Optional[callable] = None
    ) -> List[EmbeddedChunkedPost]:
        """
        Procesar un batch de posts en paralelo
        
        Args:
            raw_posts: Lista de posts a procesar
            progress_callback: Función callback para progreso (opcional)
            
        Returns:
            Lista de chunks embedidos
        """
        self._ensure_initialized()
        
        if not raw_posts:
            self.logger.info("Batch vacío recibido")
            return []
        
        start_time = time.time()
        self.logger.info(f"Procesando batch de {len(raw_posts)} posts")
        
        # Procesar posts en paralelo con control de concurrencia
        tasks = [
            self._process_single_post_safe(post, i, len(raw_posts), progress_callback)
            for i, post in enumerate(raw_posts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados y separar éxitos de errores
        successful_chunks = []
        error_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_count += 1
                self.logger.error(f"Error procesando post {i}: {result}")
                self._processing_stats["errors"] += 1
            elif result:  # result es una lista de chunks
                successful_chunks.extend(result)
                self._processing_stats["posts_processed"] += 1
                self._processing_stats["chunks_generated"] += len(result)
        
        # Actualizar estadísticas
        processing_time = time.time() - start_time
        self._processing_stats["total_processing_time"] += processing_time
        
        self.logger.info(
            f"Batch completado: {len(successful_chunks)} chunks generados, "
            f"{error_count} errores, {processing_time:.2f}s"
        )
        
        return successful_chunks
    
    async def _process_single_post_safe(
        self, 
        raw_post: RawPost, 
        index: int, 
        total: int,
        progress_callback: Optional[callable] = None
    ) -> List[EmbeddedChunkedPost]:
        """
        Procesar un post individual con manejo seguro de errores
        """
        async with self.semaphore:  # Control de concurrencia
            try:
                result = await self._process_single_post(raw_post)
                
                # Callback de progreso si se proporciona
                if progress_callback:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, progress_callback, index + 1, total
                        )
                    except Exception as e:
                        self.logger.warning(f"Error en progress callback: {e}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error procesando post {raw_post.post_id}: {e}")
                raise  # Re-raise para que gather() lo capture
    
    async def _process_single_post(self, raw_post: RawPost) -> List[EmbeddedChunkedPost]:
        """
        Procesamiento core de un post individual
        Raw -> Cleaned -> Chunked -> Embedded
        """
        
        loop = asyncio.get_event_loop()
        
        # Paso 1: Raw -> Cleaned (rápido, CPU-bound pero ligero)
        cleaned_post = await loop.run_in_executor(
            None, CleanedPost.from_raw_post, raw_post
        )
        
        # Paso 2: Cleaned -> Chunked (puede ser intensivo en texto largo)
        chunked_posts = await loop.run_in_executor(
            self.executor,
            lambda: ChunkedPost.from_cleaned_post(cleaned_post, self.embedding_model)
        )
        
        # Paso 3: Chunked -> Embedded (intensivo en ML)
        embedded_posts = []
        for chunked_post in chunked_posts:
            embedded_post = await loop.run_in_executor(
                self.executor,
                lambda cp=chunked_post: EmbeddedChunkedPost.from_chunked_post(cp, self.embedding_model)
            )
            embedded_posts.append(embedded_post)
        
        return embedded_posts
    
    async def process_posts_stream(
        self, 
        raw_posts: AsyncGenerator[RawPost, None],
        batch_size: int = 10
    ) -> AsyncGenerator[List[EmbeddedChunkedPost], None]:
        """
        Procesar posts en streaming con batching automático
        
        Args:
            raw_posts: Stream asíncrono de posts
            batch_size: Tamaño de batch para procesamiento
            
        Yields:
            Batches de chunks embedidos
        """
        self._ensure_initialized()
        
        batch = []
        async for raw_post in raw_posts:
            batch.append(raw_post)
            
            if len(batch) >= batch_size:
                # Procesar batch y yield resultado
                result = await self.process_posts_batch(batch)
                if result:
                    yield result
                batch = []
        
        # Procesar último batch si tiene elementos
        if batch:
            result = await self.process_posts_batch(batch)
            if result:
                yield result
    
    def get_processing_stats(self) -> dict:
        """Obtener estadísticas de procesamiento"""
        stats = self._processing_stats.copy()
        
        if stats["posts_processed"] > 0:
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["posts_processed"]
            stats["avg_chunks_per_post"] = stats["chunks_generated"] / stats["posts_processed"]
            stats["error_rate"] = stats["errors"] / (stats["posts_processed"] + stats["errors"])
        else:
            stats["avg_processing_time"] = 0.0
            stats["avg_chunks_per_post"] = 0.0
            stats["error_rate"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Resetear estadísticas de procesamiento"""
        self._processing_stats = {
            "posts_processed": 0,
            "chunks_generated": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }