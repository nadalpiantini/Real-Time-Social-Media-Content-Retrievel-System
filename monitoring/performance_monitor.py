"""
Sistema Completo de Monitoreo y Observabilidad
Métricas en tiempo real, health checks y alertas automáticas
"""

import asyncio
import time
import logging
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MetricDataPoint:
    """Punto de dato de métrica"""
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "labels": self.labels
        }

@dataclass
class HealthCheckResult:
    """Resultado de health check"""
    service: str
    status: str  # "healthy", "warning", "critical"
    message: str
    response_time_ms: float
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"

class MetricsCollector:
    """Colector de métricas con ventanas deslizantes"""
    
    def __init__(self, max_points: int = 1000, retention_hours: int = 24):
        self.max_points = max_points
        self.retention_seconds = retention_hours * 3600
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.RLock()
        
        # Estadísticas agregadas
        self.aggregated_stats: Dict[str, Dict[str, float]] = {}
        
        # Auto-cleanup
        self._start_cleanup_task()
    
    def record_metric(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Registrar punto de métrica"""
        with self.lock:
            point = MetricDataPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            
            self.metrics[name].append(point)
            self._update_aggregated_stats(name)
    
    def _update_aggregated_stats(self, metric_name: str):
        """Actualizar estadísticas agregadas"""
        points = self.metrics[metric_name]
        if not points:
            return
        
        values = [p.value for p in points]
        
        self.aggregated_stats[metric_name] = {
            "current": values[-1],
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values),
            "rate_per_minute": self._calculate_rate(points, 60),
            "rate_per_hour": self._calculate_rate(points, 3600)
        }
    
    def _calculate_rate(self, points: deque, window_seconds: int) -> float:
        """Calcular tasa en ventana de tiempo"""
        if len(points) < 2:
            return 0.0
        
        cutoff_time = time.time() - window_seconds
        recent_points = [p for p in points if p.timestamp >= cutoff_time]
        
        if len(recent_points) < 2:
            return 0.0
        
        time_span = recent_points[-1].timestamp - recent_points[0].timestamp
        if time_span <= 0:
            return 0.0
        
        value_change = recent_points[-1].value - recent_points[0].value
        return value_change / time_span
    
    def get_metric_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtener estadísticas de métrica"""
        with self.lock:
            return self.aggregated_stats.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todas las métricas"""
        with self.lock:
            return dict(self.aggregated_stats)
    
    def get_recent_points(self, name: str, minutes: int = 5) -> List[MetricDataPoint]:
        """Obtener puntos recientes de métrica"""
        cutoff_time = time.time() - (minutes * 60)
        with self.lock:
            if name not in self.metrics:
                return []
            
            return [p for p in self.metrics[name] if p.timestamp >= cutoff_time]
    
    def _start_cleanup_task(self):
        """Iniciar tarea de limpieza automática"""
        def cleanup_old_metrics():
            while True:
                try:
                    time.sleep(3600)  # Cleanup cada hora
                    self._cleanup_old_points()
                except Exception as e:
                    logger.error(f"Metrics cleanup error: {e}")
        
        thread = threading.Thread(target=cleanup_old_metrics, daemon=True)
        thread.start()
    
    def _cleanup_old_points(self):
        """Limpiar puntos antiguos"""
        cutoff_time = time.time() - self.retention_seconds
        cleaned_count = 0
        
        with self.lock:
            for metric_name, points in self.metrics.items():
                original_len = len(points)
                
                # Filtrar puntos antiguos
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                
                cleaned = original_len - len(points)
                cleaned_count += cleaned
                
                if cleaned > 0:
                    self._update_aggregated_stats(metric_name)
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned {cleaned_count} old metric points")

class HealthChecker:
    """Sistema de health checks con alertas"""
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.alert_callback = alert_callback
        self.check_interval = 30  # 30 segundos por defecto
        self.is_running = False
        self._check_task: Optional[asyncio.Task] = None
        
        # Thresholds
        self.warning_threshold_ms = 1000  # 1 segundo
        self.critical_threshold_ms = 5000  # 5 segundos
    
    def register_check(self, name: str, check_func: Callable):
        """Registrar health check"""
        self.checks[name] = check_func
        logger.info(f"✅ Health check registered: {name}")
    
    async def start_monitoring(self):
        """Iniciar monitoreo de health checks"""
        if self.is_running:
            return
        
        self.is_running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("🏥 Health check monitoring started")
    
    async def stop_monitoring(self):
        """Detener monitoreo"""
        self.is_running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Health check monitoring stopped")
    
    async def _health_check_loop(self):
        """Loop principal de health checks"""
        while self.is_running:
            try:
                await self._run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Health check loop error: {e}")
                await asyncio.sleep(5)  # Pausa corta en caso de error
    
    async def _run_all_checks(self):
        """Ejecutar todos los health checks"""
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                
                # Ejecutar check (puede ser sync o async)
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                response_time = (time.time() - start_time) * 1000  # ms
                
                # Procesar resultado
                if isinstance(result, HealthCheckResult):
                    health_result = result
                    health_result.response_time_ms = response_time
                elif isinstance(result, dict):
                    health_result = HealthCheckResult(
                        service=name,
                        status=result.get("status", "healthy"),
                        message=result.get("message", "OK"),
                        response_time_ms=response_time,
                        details=result.get("details", {})
                    )
                elif isinstance(result, bool):
                    health_result = HealthCheckResult(
                        service=name,
                        status="healthy" if result else "critical",
                        message="OK" if result else "Check failed",
                        response_time_ms=response_time
                    )
                else:
                    health_result = HealthCheckResult(
                        service=name,
                        status="healthy",
                        message=str(result),
                        response_time_ms=response_time
                    )
                
                # Ajustar status basado en tiempo de respuesta
                if health_result.response_time_ms > self.critical_threshold_ms:
                    health_result.status = "critical"
                    health_result.message += f" (slow: {health_result.response_time_ms:.1f}ms)"
                elif health_result.response_time_ms > self.warning_threshold_ms:
                    if health_result.status == "healthy":
                        health_result.status = "warning"
                        health_result.message += f" (slow: {health_result.response_time_ms:.1f}ms)"
                
                self.results[name] = health_result
                
                # Alerta si hay problemas
                if not health_result.is_healthy and self.alert_callback:
                    try:
                        await self.alert_callback(health_result)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
                
            except Exception as e:
                # Health check falló
                error_result = HealthCheckResult(
                    service=name,
                    status="critical",
                    message=f"Check failed: {str(e)}",
                    response_time_ms=(time.time() - start_time) * 1000 if 'start_time' in locals() else 0
                )
                
                self.results[name] = error_result
                logger.error(f"❌ Health check {name} failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado general de salud"""
        if not self.results:
            return {"overall_status": "unknown", "services": {}}
        
        healthy_count = sum(1 for r in self.results.values() if r.status == "healthy")
        warning_count = sum(1 for r in self.results.values() if r.status == "warning")
        critical_count = sum(1 for r in self.results.values() if r.status == "critical")
        
        # Estado general
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "healthy_services": healthy_count,
            "warning_services": warning_count,
            "critical_services": critical_count,
            "total_services": len(self.results),
            "services": {name: asdict(result) for name, result in self.results.items()}
        }

class SystemMetricsCollector:
    """Colector de métricas del sistema"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.is_running = False
        self._collection_task: Optional[asyncio.Task] = None
        self.collection_interval = 10  # 10 segundos
    
    async def start_collection(self):
        """Iniciar recolección de métricas del sistema"""
        if self.is_running:
            return
        
        self.is_running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("📊 System metrics collection started")
    
    async def stop_collection(self):
        """Detener recolección"""
        self.is_running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 System metrics collection stopped")
    
    async def _collection_loop(self):
        """Loop de recolección de métricas"""
        while self.is_running:
            try:
                self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ System metrics collection error: {e}")
                await asyncio.sleep(5)
    
    def _collect_system_metrics(self):
        """Recopilar métricas del sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.record_metric("system.cpu.percent", cpu_percent)
            
            # Memoria
            memory = psutil.virtual_memory()
            self.metrics.record_metric("system.memory.percent", memory.percent)
            self.metrics.record_metric("system.memory.used_gb", memory.used / (1024**3))
            self.metrics.record_metric("system.memory.available_gb", memory.available / (1024**3))
            
            # Disco
            disk = psutil.disk_usage('/')
            self.metrics.record_metric("system.disk.percent", (disk.used / disk.total) * 100)
            self.metrics.record_metric("system.disk.used_gb", disk.used / (1024**3))
            self.metrics.record_metric("system.disk.free_gb", disk.free / (1024**3))
            
            # Red (si está disponible)
            try:
                net_io = psutil.net_io_counters()
                self.metrics.record_metric("system.network.bytes_sent", net_io.bytes_sent)
                self.metrics.record_metric("system.network.bytes_recv", net_io.bytes_recv)
            except:
                pass  # Red no disponible en algunos sistemas
            
            # Número de procesos
            self.metrics.record_metric("system.processes.count", len(psutil.pids()))
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")

class PerformanceMonitor:
    """Monitor principal de rendimiento y observabilidad"""
    
    def __init__(self, data_dir: str = "monitoring"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Componentes
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker(alert_callback=self._handle_alert)
        self.system_metrics = SystemMetricsCollector(self.metrics)
        
        # Estado
        self.start_time = time.time()
        self.is_running = False
        
        # Alertas
        self.alert_handlers: List[Callable] = []
        
        # Registrar health checks básicos
        self._register_basic_health_checks()
    
    def _register_basic_health_checks(self):
        """Registrar health checks básicos del sistema"""
        
        def memory_check():
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return HealthCheckResult(
                    service="memory",
                    status="critical",
                    message=f"High memory usage: {memory.percent:.1f}%",
                    response_time_ms=0
                )
            elif memory.percent > 80:
                return HealthCheckResult(
                    service="memory",
                    status="warning", 
                    message=f"Memory usage: {memory.percent:.1f}%",
                    response_time_ms=0
                )
            else:
                return HealthCheckResult(
                    service="memory",
                    status="healthy",
                    message=f"Memory usage: {memory.percent:.1f}%",
                    response_time_ms=0
                )
        
        def disk_check():
            disk = psutil.disk_usage('/')
            percent = (disk.used / disk.total) * 100
            
            if percent > 90:
                return {"status": "critical", "message": f"High disk usage: {percent:.1f}%"}
            elif percent > 80:
                return {"status": "warning", "message": f"Disk usage: {percent:.1f}%"}
            else:
                return {"status": "healthy", "message": f"Disk usage: {percent:.1f}%"}
        
        self.health_checker.register_check("memory", memory_check)
        self.health_checker.register_check("disk", disk_check)
    
    async def start(self):
        """Iniciar monitoreo completo"""
        if self.is_running:
            return
        
        logger.info("🚀 Starting Performance Monitor...")
        
        self.is_running = True
        
        # Iniciar componentes
        await self.health_checker.start_monitoring()
        await self.system_metrics.start_collection()
        
        logger.info("✅ Performance Monitor started")
    
    async def stop(self):
        """Detener monitoreo"""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping Performance Monitor...")
        
        self.is_running = False
        
        # Detener componentes
        await self.health_checker.stop_monitoring()
        await self.system_metrics.stop_collection()
        
        logger.info("✅ Performance Monitor stopped")
    
    def record_custom_metric(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Registrar métrica personalizada"""
        self.metrics.record_metric(name, value, labels)
    
    def register_health_check(self, name: str, check_func: Callable):
        """Registrar health check personalizado"""
        self.health_checker.register_check(name, check_func)
    
    def add_alert_handler(self, handler: Callable):
        """Agregar manejador de alertas"""
        self.alert_handlers.append(handler)
    
    async def _handle_alert(self, health_result: HealthCheckResult):
        """Manejar alerta de health check"""
        logger.warning(f"🚨 Health alert: {health_result.service} - {health_result.message}")
        
        # Llamar handlers personalizados
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(health_result)
                else:
                    handler(health_result)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtener datos para dashboard"""
        runtime = time.time() - self.start_time
        
        return {
            "status": "running" if self.is_running else "stopped",
            "runtime_seconds": runtime,
            "runtime_formatted": str(timedelta(seconds=int(runtime))),
            "health": self.health_checker.get_health_status(),
            "metrics": self.metrics.get_all_metrics(),
            "system": {
                "timestamp": time.time(),
                "uptime": runtime
            }
        }
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Exportar métricas a archivo JSON"""
        if not filename:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.data_dir / filename
        
        data = {
            "export_timestamp": time.time(),
            "export_date": datetime.now().isoformat(),
            "metrics": self.metrics.get_all_metrics(),
            "health_status": self.health_checker.get_health_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📄 Metrics exported to {filepath}")
        return str(filepath)

# Global instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_global_monitor() -> PerformanceMonitor:
    """Obtener instancia global del monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor