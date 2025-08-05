"""
Enhanced Data Processing Pipeline - Phase 2 Architecture Improvements
Pipeline optimizado con servicios resilientes, procesamiento por lotes y métricas
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from bytewax import operators as op
from bytewax.dataflow import Dataflow
from qdrant_client import QdrantClient

from services.service_container import service_container
from services.resilient_data_processing_service import ResilientDataProcessingService
from models.data_source import UnifiedDataSource
from models.post import ChunkedPost, CleanedPost, EmbeddedChunkedPost, RawPost
from utils.qdrant import QdrantVectorOutput
from config.app_config import get_app_config

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetrics:
    """Métricas de rendimiento del pipeline"""
    total_posts: int = 0
    processed_posts: int = 0
    failed_posts: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    throughput_posts_per_second: float = 0.0
    throughput_chunks_per_second: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def start_processing(self):
        """Iniciar métricas de procesamiento"""
        self.start_time = time.time()
        logger.info("📊 Pipeline metrics started")
    
    def complete_processing(self):
        """Completar métricas de procesamiento"""
        self.end_time = time.time()
        if self.start_time:
            self.processing_time = self.end_time - self.start_time
            if self.processing_time > 0:
                self.throughput_posts_per_second = self.processed_posts / self.processing_time
                self.throughput_chunks_per_second = self.total_chunks / self.processing_time
        
        logger.info(f"📈 Pipeline completed - {self.processed_posts}/{self.total_posts} posts in {self.processing_time:.2f}s")
        logger.info(f"⚡ Throughput: {self.throughput_posts_per_second:.2f} posts/s, {self.throughput_chunks_per_second:.2f} chunks/s")
    
    def record_post_processed(self, chunks_count: int = 1):
        """Registrar post procesado"""
        self.processed_posts += 1
        self.total_chunks += chunks_count
    
    def record_post_failed(self):
        """Registrar post fallido"""
        self.failed_posts += 1

class OptimizedPipelineManager:
    """Manager del pipeline optimizado con servicios resilientes"""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        self.config = get_app_config()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.metrics = PipelineMetrics()
        self.service_container = None
        self.resilient_service = None
        
        # Progress callback para seguimiento
        self.progress_callback: Optional[Callable[[int, int], None]] = None
    
    async def initialize(self):
        """Inicializar servicios del pipeline"""
        logger.info("🔧 Initializing optimized pipeline...")
        
        # Inicializar contenedor de servicios
        self.service_container = service_container
        await self.service_container.initialize()
        
        # Obtener servicio resiliente
        self.resilient_service = self.service_container.get_service_typed(
            "resilient_data_processing", 
            ResilientDataProcessingService
        )
        
        logger.info("✅ Optimized pipeline initialized")
    
    async def process_batch_resilient(
        self, 
        raw_posts: List[RawPost],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[EmbeddedChunkedPost]:
        """
        Procesar lote de posts con resilencia y optimización
        """
        if not self.resilient_service:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        self.progress_callback = progress_callback
        self.metrics.total_posts = len(raw_posts)
        self.metrics.start_processing()
        
        logger.info(f"🚀 Processing batch of {len(raw_posts)} posts with resilience")
        
        try:
            # Usar servicio resiliente para procesamiento por lotes
            results = await self.resilient_service.process_posts_batch_resilient(
                raw_posts,
                progress_callback=self._internal_progress_callback
            )
            
            self.metrics.total_chunks = len(results)
            self.metrics.processed_posts = len(set(chunk.post_id for chunk in results))
            self.metrics.complete_processing()
            
            logger.info(f"✅ Batch processing completed: {len(results)} chunks from {self.metrics.processed_posts} posts")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            self.metrics.failed_posts = self.metrics.total_posts
            self.metrics.complete_processing()
            raise
    
    def _internal_progress_callback(self, current: int, total: int):
        """Callback interno de progreso"""
        if self.progress_callback:
            self.progress_callback(current, total)
        
        # Actualizar métricas en tiempo real
        progress_pct = (current / total) * 100 if total > 0 else 0
        logger.info(f"📊 Processing progress: {current}/{total} ({progress_pct:.1f}%)")
    
    async def process_parallel_batches(
        self,
        raw_posts: List[RawPost],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[EmbeddedChunkedPost]:
        """
        Procesamiento paralelo de múltiples lotes
        """
        if len(raw_posts) <= self.batch_size:
            return await self.process_batch_resilient(raw_posts, progress_callback)
        
        # Dividir en lotes
        batches = [
            raw_posts[i:i + self.batch_size] 
            for i in range(0, len(raw_posts), self.batch_size)
        ]
        
        logger.info(f"🔄 Processing {len(raw_posts)} posts in {len(batches)} parallel batches")
        
        all_results = []
        completed_batches = 0
        
        # Procesar lotes con concurrencia limitada
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_batch(batch: List[RawPost], batch_idx: int):
            async with semaphore:
                try:
                    logger.info(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} posts)")
                    results = await self.process_batch_resilient(batch)
                    
                    nonlocal completed_batches
                    completed_batches += 1
                    
                    if progress_callback:
                        total_progress = (completed_batches / len(batches)) * len(raw_posts)
                        progress_callback(int(total_progress), len(raw_posts))
                    
                    return results
                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx + 1} failed: {e}")
                    return []
        
        # Ejecutar todas las tareas de lotes
        tasks = [
            process_single_batch(batch, idx) 
            for idx, batch in enumerate(batches)
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Consolidar resultados
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"❌ Batch processing exception: {result}")
            elif isinstance(result, list):
                all_results.extend(result)
        
        logger.info(f"✅ Parallel batch processing completed: {len(all_results)} total chunks")
        return all_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento"""
        return {
            "total_posts": self.metrics.total_posts,
            "processed_posts": self.metrics.processed_posts,
            "failed_posts": self.metrics.failed_posts,
            "total_chunks": self.metrics.total_chunks,
            "processing_time": self.metrics.processing_time,
            "throughput_posts_per_second": self.metrics.throughput_posts_per_second,
            "throughput_chunks_per_second": self.metrics.throughput_chunks_per_second,
            "success_rate": (self.metrics.processed_posts / self.metrics.total_posts * 100) if self.metrics.total_posts > 0 else 0
        }
    
    async def shutdown(self):
        """Limpiar recursos del pipeline"""
        if self.service_container:
            await self.service_container.shutdown()
        logger.info("🧹 Pipeline shutdown completed")

# Funciones de compatibilidad con Bytewax existente
def build_optimized(in_memory: bool = False, data_source_path: List[str] = None, batch_size: int = 10):
    """
    Construir pipeline optimizado con arquitectura mejorada
    """
    logger.info("🚀 Building optimized Bytewax flow...")
    
    config = get_app_config()
    flow = Dataflow("optimized_flow")
    
    logger.info(f"📊 Data source paths: {data_source_path}")
    logger.info(f"⚙️ Batch size: {batch_size}")
    
    # Usar UnifiedDataSource
    stream = op.input(
        "input", flow, UnifiedDataSource(json_files=data_source_path)
    )
    
    # Pipeline optimizado con logging mejorado
    def enhanced_log_step(step_name: str):
        def log_func(data):
            if hasattr(data, 'post_id'):
                logger.info(f"🔄 {step_name}: Processing post {data.post_id}")
            else:
                logger.info(f"🔄 {step_name}: Processing {type(data).__name__}")
            return data
        return log_func
    
    # Etapas del pipeline con logging mejorado
    stream = op.map("log_input", stream, enhanced_log_step("Input"))
    stream = op.map("raw_post", stream, RawPost.from_source)
    stream = op.map("log_raw", stream, enhanced_log_step("Raw Post"))
    
    # Aquí se podría integrar el procesamiento por lotes optimizado
    # Por ahora mantenemos compatibilidad con el pipeline existente
    stream = op.map("cleaned_post", stream, CleanedPost.from_raw_post)
    stream = op.map("log_cleaned", stream, enhanced_log_step("Cleaned Post"))
    
    # Chunking optimizado
    stream = op.flat_map(
        "chunked_post",
        stream,
        lambda cleaned_post: ChunkedPost.from_cleaned_post(cleaned_post)
    )
    stream = op.map("log_chunked", stream, enhanced_log_step("Chunked Post"))
    
    # Embedding con métricas
    def embedding_with_metrics(chunked_post):
        start_time = time.time()
        result = EmbeddedChunkedPost.from_chunked_post(chunked_post)
        processing_time = time.time() - start_time
        logger.info(f"⚡ Embedding took {processing_time:.3f}s for post {result.post_id}")
        return result
    
    stream = op.map("embedded_chunked_post", stream, embedding_with_metrics)
    stream = op.map("log_embedded", stream, enhanced_log_step("Embedded Post"))
    
    # Inspect mejorado con métricas detalladas
    def detailed_inspect_with_metrics(data):
        timestamp = time.strftime("%H:%M:%S")
        message = f"🔍 [{timestamp}] Final: Post {data.post_id} → {len(data.text)} chars → {len(data.text_embedding)} dims"
        logger.info(message)
        return data
    
    op.inspect("inspect", stream, detailed_inspect_with_metrics)
    
    # Output con configuración mejorada
    def build_optimized_output():
        config = get_app_config()
        if in_memory:
            return QdrantVectorOutput(
                vector_size=config.ml.embedding_size,
                client=QdrantClient(":memory:"),
            )
        else:
            return QdrantVectorOutput(
                vector_size=config.ml.embedding_size,
            )
    
    op.output("output", stream, build_optimized_output())
    
    logger.info("✅ Optimized Bytewax flow built successfully")
    return flow

# Función de compatibilidad hacia atrás
def build(in_memory: bool = False, data_source_path: List[str] = None):
    """Función de compatibilidad con el pipeline original"""
    return build_optimized(in_memory=in_memory, data_source_path=data_source_path)