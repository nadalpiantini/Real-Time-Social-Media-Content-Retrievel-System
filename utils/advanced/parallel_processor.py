"""
Arquitectura de Procesamiento Paralelo - Worker Pools y Load Balancing
Sistema avanzado de procesamiento concurrente con distribución inteligente de carga
"""

import asyncio
import time
import logging
from typing import Any, List, Dict, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue, Empty
from threading import Lock, Event, Semaphore
from enum import Enum
import multiprocessing as mp
import psutil
import uuid
from contextlib import asynccontextmanager

from models.post import RawPost, EmbeddedChunkedPost
from utils.advanced.smart_cache import get_memory_manager

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Prioridades de tareas"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class ProcessingTask:
    """Tarea de procesamiento con metadatos"""
    id: str
    data: Any
    priority: TaskPriority = TaskPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    assigned_worker: Optional[str] = None
    processing_started_at: Optional[float] = None
    processing_completed_at: Optional[float] = None
    
    def __lt__(self, other):
        """Comparación para priority queue"""
        return self.priority.value < other.priority.value
    
    @property
    def processing_time(self) -> Optional[float]:
        """Tiempo de procesamiento en segundos"""
        if self.processing_started_at and self.processing_completed_at:
            return self.processing_completed_at - self.processing_started_at
        return None
    
    @property
    def age(self) -> float:
        """Edad de la tarea en segundos"""
        return time.time() - self.created_at

@dataclass 
class WorkerStats:
    """Estadísticas de un worker"""
    worker_id: str
    tasks_processed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    last_activity: float = field(default_factory=time.time)
    current_task: Optional[str] = None
    is_busy: bool = False
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito del worker"""
        total = self.tasks_processed + self.tasks_failed
        return (self.tasks_processed / total * 100) if total > 0 else 0.0
    
    @property
    def average_processing_time(self) -> float:
        """Tiempo promedio de procesamiento"""
        return (self.total_processing_time / self.tasks_processed) if self.tasks_processed > 0 else 0.0

class LoadBalancer:
    """Load balancer inteligente para distribución de tareas"""
    
    def __init__(self):
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.lock = Lock()
    
    def register_worker(self, worker_id: str):
        """Registrar un nuevo worker"""
        with self.lock:
            self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
            logger.info(f"✅ Worker {worker_id} registered")
    
    def unregister_worker(self, worker_id: str):
        """Desregistrar worker"""
        with self.lock:
            if worker_id in self.worker_stats:
                del self.worker_stats[worker_id]
                logger.info(f"❌ Worker {worker_id} unregistered")
    
    def select_best_worker(self, exclude_busy: bool = True) -> Optional[str]:
        """Seleccionar el mejor worker disponible"""
        with self.lock:
            available_workers = []
            
            for worker_id, stats in self.worker_stats.items():
                if exclude_busy and stats.is_busy:
                    continue
                
                # Score basado en múltiples factores
                score = self._calculate_worker_score(stats)
                available_workers.append((worker_id, score))
            
            if not available_workers:
                return None
            
            # Seleccionar worker con mejor score
            available_workers.sort(key=lambda x: x[1], reverse=True)
            best_worker = available_workers[0][0]
            
            logger.debug(f"Selected worker {best_worker} from {len(available_workers)} available")
            return best_worker
    
    def _calculate_worker_score(self, stats: WorkerStats) -> float:
        """Calcular score de un worker para load balancing"""
        # Factores del score (mayor es mejor)
        success_rate_factor = stats.success_rate / 100.0  # 0-1
        
        # Penalizar workers lentos
        avg_time = stats.average_processing_time
        speed_factor = 1.0 / (1.0 + avg_time) if avg_time > 0 else 1.0
        
        # Preferir workers menos cargados
        load_factor = 1.0 / (1.0 + stats.tasks_processed / 100.0)
        
        # Activity factor (preferir workers activos recientemente)
        time_since_activity = time.time() - stats.last_activity
        activity_factor = 1.0 / (1.0 + time_since_activity / 3600.0)  # Penalizar inactividad > 1h
        
        # Score final (0-4)
        score = success_rate_factor + speed_factor + load_factor + activity_factor
        return score
    
    def update_worker_stats(self, worker_id: str, task: ProcessingTask, success: bool):
        """Actualizar estadísticas de worker"""
        with self.lock:
            if worker_id not in self.worker_stats:
                return
            
            stats = self.worker_stats[worker_id]
            stats.last_activity = time.time()
            stats.current_task = None
            stats.is_busy = False
            
            if success:
                stats.tasks_processed += 1
                if task.processing_time:
                    stats.total_processing_time += task.processing_time
            else:
                stats.tasks_failed += 1
    
    def mark_worker_busy(self, worker_id: str, task_id: str):
        """Marcar worker como ocupado"""
        with self.lock:
            if worker_id in self.worker_stats:
                stats = self.worker_stats[worker_id]
                stats.is_busy = True
                stats.current_task = task_id
                stats.last_activity = time.time()
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de load balancing"""
        with self.lock:
            total_workers = len(self.worker_stats)
            busy_workers = sum(1 for stats in self.worker_stats.values() if stats.is_busy)
            
            avg_success_rate = sum(stats.success_rate for stats in self.worker_stats.values()) / total_workers if total_workers > 0 else 0
            
            return {
                "total_workers": total_workers,
                "busy_workers": busy_workers,
                "available_workers": total_workers - busy_workers,
                "average_success_rate": avg_success_rate,
                "worker_details": {
                    worker_id: {
                        "tasks_processed": stats.tasks_processed,
                        "tasks_failed": stats.tasks_failed,
                        "success_rate": stats.success_rate,
                        "avg_processing_time": stats.average_processing_time,
                        "is_busy": stats.is_busy
                    }
                    for worker_id, stats in self.worker_stats.items()
                }
            }

class ParallelProcessor:
    """
    Procesador paralelo con arquitectura avanzada de workers
    """
    
    def __init__(
        self,
        max_workers: int = None,
        use_process_pool: bool = False,
        queue_size: int = 1000,
        worker_timeout: float = 300.0  # 5 minutos
    ):
        # Configuración automática de workers basada en sistema
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 4) * 2)
        self.use_process_pool = use_process_pool
        self.queue_size = queue_size
        self.worker_timeout = worker_timeout
        
        # Estructuras de datos
        self.task_queue: PriorityQueue = PriorityQueue(maxsize=queue_size)
        self.result_futures: Dict[str, asyncio.Future] = {}
        self.active_tasks: Dict[str, ProcessingTask] = {}
        
        # Workers y load balancer
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self.load_balancer = LoadBalancer()
        self.worker_semaphore = Semaphore(self.max_workers)
        
        # Control y estado
        self.is_running = False
        self.shutdown_event = Event()
        self._processor_task: Optional[asyncio.Task] = None
        
        # Métricas
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.start_time = time.time()
        
        logger.info(f"🚀 ParallelProcessor initialized: {self.max_workers} workers, process_pool={use_process_pool}")
    
    async def start(self):
        """Iniciar el procesador paralelo"""
        if self.is_running:
            logger.warning("ParallelProcessor already running")
            return
        
        logger.info("🚀 Starting ParallelProcessor...")
        
        # Inicializar executor
        if self.use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="parallel-worker")
        
        # Registrar workers en load balancer
        for i in range(self.max_workers):
            worker_id = f"worker-{i:03d}"
            self.load_balancer.register_worker(worker_id)
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Iniciar tarea de procesamiento
        self._processor_task = asyncio.create_task(self._process_tasks())
        
        logger.info(f"✅ ParallelProcessor started with {self.max_workers} workers")
    
    async def shutdown(self, timeout: float = 30.0):
        """Detener el procesador paralelo"""
        if not self.is_running:
            return
        
        logger.info("🛑 Shutting down ParallelProcessor...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Esperar a que termine el procesamiento
        if self._processor_task:
            try:
                await asyncio.wait_for(self._processor_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Processor task shutdown timeout")
                self._processor_task.cancel()
        
        # Cerrar executor
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        
        logger.info("✅ ParallelProcessor shutdown completed")
    
    async def submit_task(
        self,
        task_func: Callable,
        data: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Enviar tarea para procesamiento paralelo
        """
        if not self.is_running:
            raise RuntimeError("ParallelProcessor not started")
        
        # Crear tarea
        task = ProcessingTask(
            id=str(uuid.uuid4()),
            data=data,
            priority=priority
        )
        
        # Crear future para resultado
        result_future = asyncio.Future()
        self.result_futures[task.id] = result_future
        
        # Añadir a cola con prioridad
        try:
            self.task_queue.put_nowait((task.priority.value, task, task_func))
            self.total_tasks_submitted += 1
            logger.debug(f"📤 Task {task.id} submitted with priority {priority.name}")
        except:
            # Cola llena
            del self.result_futures[task.id]
            raise RuntimeError("Task queue is full")
        
        # Esperar resultado con timeout
        try:
            if timeout:
                result = await asyncio.wait_for(result_future, timeout=timeout)
            else:
                result = await result_future
            return result
        except asyncio.TimeoutError:
            # Cleanup en caso de timeout
            if task.id in self.result_futures:
                del self.result_futures[task.id]
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            raise
    
    async def _process_tasks(self):
        """Loop principal de procesamiento de tareas"""
        logger.info("🔄 Task processing loop started")
        
        while self.is_running or not self.task_queue.empty():
            try:
                # Obtener tarea de la cola (con timeout para permitir shutdown)
                try:
                    priority, task, task_func = self.task_queue.get(timeout=1.0)
                except Empty:
                    if self.shutdown_event.is_set():
                        break
                    continue
                
                # Verificar memoria antes de procesar
                memory_manager = get_memory_manager()
                if memory_manager.should_trigger_gc():
                    memory_manager.optimize_memory()
                
                # Procesar tarea
                await self._execute_task(task, task_func)
                
            except Exception as e:
                logger.error(f"❌ Error in task processing loop: {e}")
                await asyncio.sleep(0.1)  # Breve pausa para evitar busy loop
        
        logger.info("🏁 Task processing loop ended")
    
    async def _execute_task(self, task: ProcessingTask, task_func: Callable):
        """Ejecutar una tarea específica"""
        # Seleccionar worker
        worker_id = self.load_balancer.select_best_worker()
        if not worker_id:
            # No hay workers disponibles, reencolar
            try:
                self.task_queue.put_nowait((task.priority.value, task, task_func))
                await asyncio.sleep(0.1)
                return
            except:
                # Cola llena, marcar como fallida
                await self._handle_task_result(task, None, Exception("No workers available"), worker_id)
                return
        
        # Marcar worker como ocupado
        self.load_balancer.mark_worker_busy(worker_id, task.id)
        self.active_tasks[task.id] = task
        task.assigned_worker = worker_id
        task.processing_started_at = time.time()
        
        # Ejecutar tarea en executor
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, task_func, task.data)
            task.processing_completed_at = time.time()
            await self._handle_task_result(task, result, None, worker_id)
            
        except Exception as e:
            task.processing_completed_at = time.time()
            await self._handle_task_result(task, None, e, worker_id)
    
    async def _handle_task_result(self, task: ProcessingTask, result: Any, error: Optional[Exception], worker_id: Optional[str]):
        """Manejar resultado de tarea"""
        # Actualizar estadísticas de worker
        if worker_id:
            self.load_balancer.update_worker_stats(worker_id, task, error is None)
        
        # Limpiar tarea activa
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        # Resolver future
        if task.id in self.result_futures:
            future = self.result_futures[task.id]
            del self.result_futures[task.id]
            
            if error:
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    logger.warning(f"🔄 Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
                    try:
                        # Reencolar para retry
                        self.task_queue.put_nowait((task.priority.value, task, None))  # task_func ya no está disponible aquí
                        return
                    except:
                        pass  # Cola llena, fallar
                
                # Fallar definitivamente
                self.total_tasks_failed += 1
                future.set_exception(error)
                logger.error(f"❌ Task {task.id} failed after {task.retry_count} retries: {error}")
            else:
                # Éxito
                self.total_tasks_completed += 1
                future.set_result(result)
                logger.debug(f"✅ Task {task.id} completed in {task.processing_time:.3f}s")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento"""
        runtime = time.time() - self.start_time
        
        return {
            "total_tasks_submitted": self.total_tasks_submitted,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "runtime_seconds": runtime,
            "tasks_per_second": self.total_tasks_completed / runtime if runtime > 0 else 0,
            "success_rate": (self.total_tasks_completed / self.total_tasks_submitted * 100) if self.total_tasks_submitted > 0 else 0,
            "load_balancer": self.load_balancer.get_load_balancing_stats()
        }

# Context manager para uso fácil
@asynccontextmanager
async def parallel_processor(
    max_workers: int = None,
    use_process_pool: bool = False
):
    """Context manager para ParallelProcessor"""
    processor = ParallelProcessor(
        max_workers=max_workers,
        use_process_pool=use_process_pool
    )
    
    try:
        await processor.start()
        yield processor
    finally:
        await processor.shutdown()

# Funciones de utilidad para casos comunes
async def process_posts_parallel(
    posts: List[RawPost],
    processing_func: Callable,
    max_workers: int = None,
    batch_size: int = 10
) -> List[Any]:
    """
    Procesar posts en paralelo con función específica
    """
    async with parallel_processor(max_workers=max_workers) as processor:
        # Dividir en lotes si es necesario
        if len(posts) > batch_size:
            batches = [posts[i:i + batch_size] for i in range(0, len(posts), batch_size)]
            tasks = []
            
            for batch in batches:
                task = processor.submit_task(processing_func, batch, TaskPriority.NORMAL)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Consolidar resultados
            results = []
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error(f"❌ Batch processing failed: {batch_result}")
                elif isinstance(batch_result, list):
                    results.extend(batch_result)
            
            return results
        else:
            # Procesar directamente
            return await processor.submit_task(processing_func, posts, TaskPriority.NORMAL)

# Global instance para reutilización
_global_processor: Optional[ParallelProcessor] = None

async def get_global_processor() -> ParallelProcessor:
    """Obtener instancia global del procesador paralelo"""
    global _global_processor
    
    if _global_processor is None or not _global_processor.is_running:
        _global_processor = ParallelProcessor()
        await _global_processor.start()
    
    return _global_processor