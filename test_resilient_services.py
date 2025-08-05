#!/usr/bin/env python3
"""
Test script para validar la integración de servicios resilientes
Prueba circuit breaker, retry patterns y fallback mechanisms
"""

import sys
import os
import asyncio
import logging
import time
from typing import List

# Agregar el directorio actual al path para imports
sys.path.append('.')

from services.service_container import service_container
from services.resilient_data_processing_service import ResilientDataProcessingService
from models.post import RawPost
from utils.advanced.circuit_breaker import CircuitBreakerError
from utils.advanced.retry_handler import RetryExhaustedError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resilience_test")

class ResilientServiceTester:
    """Tester para validar servicios resilientes"""
    
    def __init__(self):
        self.container = None
        self.resilient_service = None
        
    async def setup(self):
        """Configurar ambiente de pruebas"""
        logger.info("🔧 Configurando ambiente de pruebas...")
        
        # Inicializar contenedor de servicios
        self.container = service_container
        await self.container.initialize()
        
        # Obtener servicio resiliente
        self.resilient_service = self.container.get_service_typed(
            "resilient_data_processing", 
            ResilientDataProcessingService
        )
        
        logger.info("✅ Ambiente configurado correctamente")
    
    async def test_service_health(self):
        """Test 1: Verificar health checks del servicio"""
        logger.info("\n📊 Test 1: Health checks del servicio resiliente")
        
        try:
            # Health check del contenedor
            container_health = await self.container.health_check()
            logger.info(f"Estado del contenedor: {container_health['overall_status']}")
            
            # Health check específico del servicio resiliente
            resilience_status = await self.resilient_service.get_resilience_status()
            logger.info(f"Servicio inicializado: {resilience_status['initialized']}")
            logger.info(f"Circuit breakers: {list(resilience_status['circuit_breakers'].keys())}")
            logger.info(f"Retry handlers: {list(resilience_status['retry_handlers'].keys())}")
            
            logger.info("✅ Test 1 PASADO: Health checks funcionando")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test 1 FALLIDO: {e}")
            return False
    
    async def test_normal_processing(self):
        """Test 2: Procesamiento normal sin fallos"""
        logger.info("\n🔄 Test 2: Procesamiento normal")
        
        try:
            # Crear posts de prueba
            test_posts = [
                RawPost(
                    post_id=f"test_post_{i}",
                    text=f"Contenido de prueba {i} para testing de servicios resilientes. "
                         f"Este post tiene suficiente texto para generar embeddings correctamente.",
                    post_owner=f"test_author_{i}",
                    source="test",
                    image=None
                )
                for i in range(3)
            ]
            
            # Procesar con servicio resiliente
            start_time = time.time()
            results = await self.resilient_service.process_posts_batch_resilient(
                test_posts,
                progress_callback=self._progress_callback
            )
            processing_time = time.time() - start_time
            
            logger.info(f"Procesados {len(results)} chunks en {processing_time:.2f}s")
            logger.info(f"Posts procesados: {len(test_posts)}")
            
            # Verificar resultados
            if len(results) > 0:
                logger.info("✅ Test 2 PASADO: Procesamiento normal exitoso")
                return True
            else:
                logger.error("❌ Test 2 FALLIDO: No se generaron resultados")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test 2 FALLIDO: {e}")
            return False
    
    async def test_circuit_breaker_stats(self):
        """Test 3: Verificar estadísticas de circuit breaker"""
        logger.info("\n📈 Test 3: Estadísticas de resilencia")
        
        try:
            # Obtener estadísticas después del procesamiento
            resilience_status = await self.resilient_service.get_resilience_status()
            
            logger.info("Estadísticas de resilencia:")
            logger.info(f"  - Activaciones de circuit breaker: {resilience_status['resilience_stats']['circuit_breaker_activations']}")
            logger.info(f"  - Intentos de retry: {resilience_status['resilience_stats']['retry_attempts']}")
            logger.info(f"  - Activaciones de fallback: {resilience_status['resilience_stats']['fallback_activations']}")
            
            # Estadísticas de circuit breakers
            for cb_name, cb_stats in resilience_status['circuit_breakers'].items():
                logger.info(f"Circuit Breaker '{cb_name}':")
                logger.info(f"  - Estado: {cb_stats['state']}")
                logger.info(f"  - Total calls: {cb_stats['stats'].get('total_calls', 0)}")
                logger.info(f"  - Successful calls: {cb_stats['stats'].get('successful_calls', 0)}")
            
            # Estadísticas de retry handlers
            for rh_name, rh_stats in resilience_status['retry_handlers'].items():
                logger.info(f"Retry Handler '{rh_name}':")
                logger.info(f"  - Total calls: {rh_stats.get('total_calls', 0)}")
                logger.info(f"  - Success rate: {rh_stats.get('success_rate', 0):.2%}")
            
            logger.info("✅ Test 3 PASADO: Estadísticas funcionando")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test 3 FALLIDO: {e}")
            return False
    
    async def test_empty_input_handling(self):
        """Test 4: Manejo de entrada vacía"""
        logger.info("\n🕳️ Test 4: Manejo de entrada vacía")
        
        try:
            # Procesar lista vacía
            results = await self.resilient_service.process_posts_batch_resilient([])
            
            if len(results) == 0:
                logger.info("✅ Test 4 PASADO: Manejo correcto de entrada vacía")
                return True
            else:
                logger.error(f"❌ Test 4 FALLIDO: Se esperaba lista vacía, se obtuvo {len(results)} elementos")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test 4 FALLIDO: {e}")
            return False
    
    async def test_service_integration(self):
        """Test 5: Integración con contenedor de servicios"""
        logger.info("\n🔗 Test 5: Integración del contenedor")
        
        try:
            # Verificar que el servicio está disponible
            available_services = self.container.list_services()
            logger.info(f"Servicios disponibles: {list(available_services.keys())}")
            
            # Verificar que podemos obtener el servicio
            service_check = self.container.has_service("resilient_data_processing")
            
            if service_check:
                logger.info("✅ Test 5 PASADO: Integración del contenedor exitosa")
                return True
            else:
                logger.error("❌ Test 5 FALLIDO: Servicio no encontrado en contenedor")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test 5 FALLIDO: {e}")
            return False
    
    def _progress_callback(self, current: int, total: int):
        """Callback de progreso para testing"""
        progress = (current / total) * 100
        logger.info(f"Progreso: {current}/{total} ({progress:.1f}%)")
    
    async def run_all_tests(self):
        """Ejecutar todos los tests de resilencia"""
        logger.info("🚀 Iniciando tests de servicios resilientes")
        
        tests = [
            ("Health Checks", self.test_service_health),
            ("Procesamiento Normal", self.test_normal_processing),
            ("Estadísticas", self.test_circuit_breaker_stats),
            ("Entrada Vacía", self.test_empty_input_handling),
            ("Integración", self.test_service_integration),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                if result:
                    passed += 1
            except Exception as e:
                logger.error(f"❌ Error ejecutando {test_name}: {e}")
        
        # Reporte final
        logger.info(f"\n📋 REPORTE FINAL:")
        logger.info(f"Tests pasados: {passed}/{total}")
        logger.info(f"Tasa de éxito: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 TODOS LOS TESTS PASARON - Servicios resilientes funcionando correctamente")
        else:
            logger.warning(f"⚠️ {total-passed} tests fallaron - Revisar implementación")
        
        return passed == total
    
    async def cleanup(self):
        """Limpiar recursos"""
        if self.container:
            await self.container.shutdown()
        logger.info("🧹 Cleanup completado")

async def main():
    """Función principal de testing"""
    tester = ResilientServiceTester()
    
    try:
        await tester.setup()
        success = await tester.run_all_tests()
        
        if success:
            logger.info("\n🎯 RESULTADO: Servicios resilientes validados exitosamente")
        else:
            logger.error("\n💥 RESULTADO: Fallos detectados en servicios resilientes")
            
        return success
        
    except Exception as e:
        logger.error(f"💥 Error crítico en testing: {e}")
        return False
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    # Ejecutar tests
    try:
        success = asyncio.run(main())
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Testing interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Error fatal: {e}")
        sys.exit(1)