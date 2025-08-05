#!/usr/bin/env python3
"""
Script de validación completa para Fase 1: Architecture Base Refactoring
Valida todos los componentes implementados en la fase inicial
"""

import sys
import os
import asyncio
import logging
import time
from typing import List, Dict, Any

# Agregar el directorio actual al path para imports
sys.path.append('.')

# Imports para testing
from config.app_config import get_app_config, AppConfig
from config.validation import SystemValidator
from services.service_container import service_container
from services.data_processing_service import DataProcessingService
from services.resilient_data_processing_service import ResilientDataProcessingService
from models.post import RawPost
from models.settings import get_settings  # Legacy compatibility test
from utils.advanced.circuit_breaker import CircuitBreaker
from utils.advanced.retry_handler import RetryHandler

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phase1_validation")

class Phase1Validator:
    """Validador completo para Fase 1 de refactoring arquitectural"""
    
    def __init__(self):
        self.container = None
        self.results = {
            "config_system": {"passed": 0, "total": 0, "details": []},
            "services_layer": {"passed": 0, "total": 0, "details": []},
            "resilience_patterns": {"passed": 0, "total": 0, "details": []},
            "backward_compatibility": {"passed": 0, "total": 0, "details": []},
            "integration": {"passed": 0, "total": 0, "details": []},
        }
    
    async def run_complete_validation(self):
        """Ejecutar validación completa de Fase 1"""
        logger.info("🚀 INICIANDO VALIDACIÓN COMPLETA - FASE 1: Architecture Base Refactoring")
        logger.info("=" * 80)
        
        # Ejecutar todos los grupos de tests
        test_groups = [
            ("Sistema de Configuración", self.validate_config_system),
            ("Capa de Servicios", self.validate_services_layer),
            ("Patterns de Resilencia", self.validate_resilience_patterns),
            ("Compatibilidad hacia Atrás", self.validate_backward_compatibility),
            ("Integración Completa", self.validate_integration),
        ]
        
        overall_success = True
        
        for group_name, test_func in test_groups:
            logger.info(f"\n📋 GRUPO: {group_name}")
            logger.info("-" * 50)
            
            try:
                group_success = await test_func()
                if not group_success:
                    overall_success = False
            except Exception as e:
                logger.error(f"💥 Error crítico en {group_name}: {e}")
                overall_success = False
        
        # Generar reporte final
        await self.generate_final_report(overall_success)
        return overall_success
    
    async def validate_config_system(self) -> bool:
        """Validar sistema de configuración unificado"""
        group_key = "config_system"
        
        # Test 1: Cargar configuración nueva
        try:
            config = get_app_config()
            assert isinstance(config, AppConfig), "Configuración no es instancia de AppConfig"
            self._record_success(group_key, "Cargar configuración nueva")
        except Exception as e:
            self._record_failure(group_key, "Cargar configuración nueva", str(e))
        
        # Test 2: Acceso a configuración de base de datos
        try:
            db_config = config.database
            # Verificar que la configuración existe y tiene campos válidos
            assert hasattr(db_config, 'qdrant_url'), "Configuración de Qdrant no disponible"
            assert db_config.qdrant_url is not None, "URL de Qdrant no definida"
            assert hasattr(db_config, 'collection_name'), "Nombre de colección no disponible"
            # Supabase is optional - only check if use_supabase is True
            if db_config.use_supabase:
                assert db_config.supabase_url is not None, "URL de Supabase requerida cuando use_supabase=True"
            self._record_success(group_key, "Configuración de base de datos")
        except Exception as e:
            self._record_failure(group_key, "Configuración de base de datos", str(e))
        
        # Test 3: Configuración de ML
        try:
            ml_config = config.ml
            assert ml_config.embedding_model_name is not None, "Modelo de embedding no definido"
            assert ml_config.embedding_dimension > 0, "Dimensión de embedding inválida"
            self._record_success(group_key, "Configuración de ML")
        except Exception as e:
            self._record_failure(group_key, "Configuración de ML", str(e))
        
        # Test 4: Validación del sistema
        try:
            validator = SystemValidator()
            validation_result = await validator.validate_system()
            assert validation_result["system_healthy"], "Sistema no está saludable"
            self._record_success(group_key, "Validación del sistema")
        except Exception as e:
            self._record_failure(group_key, "Validación del sistema", str(e))
        
        return self._get_group_success(group_key)
    
    async def validate_services_layer(self) -> bool:
        """Validar capa de servicios con dependency injection"""
        group_key = "services_layer"
        
        # Test 1: Inicializar contenedor
        try:
            self.container = service_container
            await self.container.initialize()
            self._record_success(group_key, "Inicialización del contenedor")
        except Exception as e:
            self._record_failure(group_key, "Inicialización del contenedor", str(e))
            return False
        
        # Test 2: Verificar servicios disponibles
        try:
            services = self.container.list_services()
            expected_services = ["data_processing", "resilient_data_processing"]
            for service_name in expected_services:
                assert service_name in services, f"Servicio {service_name} no encontrado"
            self._record_success(group_key, "Servicios disponibles")
        except Exception as e:
            self._record_failure(group_key, "Servicios disponibles", str(e))
        
        # Test 3: Health check del contenedor
        try:
            health = await self.container.health_check()
            assert health["overall_status"] == "healthy", "Contenedor no está saludable"
            self._record_success(group_key, "Health check del contenedor")
        except Exception as e:
            self._record_failure(group_key, "Health check del contenedor", str(e))
        
        # Test 4: Obtener servicios tipados
        try:
            data_service = self.container.get_service_typed("data_processing", DataProcessingService)
            resilient_service = self.container.get_service_typed("resilient_data_processing", ResilientDataProcessingService)
            assert data_service is not None, "Servicio de datos es None"
            assert resilient_service is not None, "Servicio resiliente es None"
            self._record_success(group_key, "Servicios tipados")
        except Exception as e:
            self._record_failure(group_key, "Servicios tipados", str(e))
        
        return self._get_group_success(group_key)
    
    async def validate_resilience_patterns(self) -> bool:
        """Validar patterns de resilencia (circuit breaker, retry)"""
        group_key = "resilience_patterns"
        
        # Test 1: Circuit breaker standalone
        try:
            cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5, name="test_cb")
            
            # Test función exitosa
            result = await cb.call(lambda: "success")
            assert result == "success", "Circuit breaker no ejecutó función exitosa"
            assert cb.is_closed, "Circuit breaker debería estar cerrado"
            self._record_success(group_key, "Circuit breaker standalone")
        except Exception as e:
            self._record_failure(group_key, "Circuit breaker standalone", str(e))
        
        # Test 2: Retry handler standalone
        try:
            retry_handler = RetryHandler(max_retries=2, base_delay=0.1, name="test_retry")
            
            # Test función exitosa
            result = await retry_handler.execute(lambda: "retry_success")
            assert result == "retry_success", "Retry handler no ejecutó función exitosa"
            self._record_success(group_key, "Retry handler standalone")
        except Exception as e:
            self._record_failure(group_key, "Retry handler standalone", str(e))
        
        # Test 3: Servicio resiliente integrado
        try:
            if self.container:
                resilient_service = self.container.get_service("resilient_data_processing")
                status = await resilient_service.get_resilience_status()
                
                assert status["initialized"], "Servicio resiliente no inicializado"
                assert "embedding" in status["circuit_breakers"], "Circuit breaker de embedding no encontrado"
                assert "processing" in status["circuit_breakers"], "Circuit breaker de processing no encontrado"
                self._record_success(group_key, "Servicio resiliente integrado")
        except Exception as e:
            self._record_failure(group_key, "Servicio resiliente integrado", str(e))
        
        # Test 4: Procesamiento con resilencia
        try:
            if self.container:
                resilient_service = self.container.get_service("resilient_data_processing")
                test_posts = [
                    RawPost(
                        post_id="resilience_test",
                        text="Post de prueba para validar resilencia del sistema completo.",
                        post_owner="test_user",
                        source="test",
                        image=None
                    )
                ]
                
                results = await resilient_service.process_posts_batch_resilient(test_posts)
                assert len(results) > 0, "No se generaron resultados con resilencia"
                self._record_success(group_key, "Procesamiento con resilencia")
        except Exception as e:
            self._record_failure(group_key, "Procesamiento con resilencia", str(e))
        
        return self._get_group_success(group_key)
    
    async def validate_backward_compatibility(self) -> bool:
        """Validar compatibilidad hacia atrás con sistema existente"""
        group_key = "backward_compatibility"
        
        # Test 1: Configuración legacy
        try:
            legacy_settings = get_settings()  # Función legacy
            assert legacy_settings is not None, "Configuración legacy no disponible"
            assert hasattr(legacy_settings, 'EMBEDDING_MODEL_NAME'), "Configuración legacy incompleta"
            self._record_success(group_key, "Configuración legacy")
        except Exception as e:
            self._record_failure(group_key, "Configuración legacy", str(e))
        
        # Test 2: Modelo de datos existente
        try:
            # Test que el modelo RawPost sigue funcionando
            post = RawPost(
                post_id="backward_test",
                text="Test de compatibilidad hacia atrás",
                post_owner="test_user",
                source="test",
                image=None
            )
            assert post.post_id == "backward_test", "Modelo RawPost no funciona"
            self._record_success(group_key, "Modelo de datos existente")
        except Exception as e:
            self._record_failure(group_key, "Modelo de datos existente", str(e))
        
        # Test 3: Servicio básico sin resilencia
        try:
            if self.container:
                basic_service = self.container.get_service("data_processing")
                assert basic_service is not None, "Servicio básico no disponible"
                
                # Verificar que es la clase correcta
                assert isinstance(basic_service, DataProcessingService), "Servicio básico tipo incorrecto"
                self._record_success(group_key, "Servicio básico sin resilencia")
        except Exception as e:
            self._record_failure(group_key, "Servicio básico sin resilencia", str(e))
        
        # Test 4: Importes existentes
        try:
            # Verificar que los importes existentes siguen funcionando
            from models.settings import get_settings as legacy_get_settings
            from utils.qdrant import QdrantClient
            from models.post import RawPost as LegacyRawPost
            
            assert legacy_get_settings is not None, "Import legacy get_settings falla"
            assert QdrantClient is not None, "Import QdrantClient falla"
            assert LegacyRawPost is not None, "Import legacy RawPost falla"
            self._record_success(group_key, "Importes existentes")
        except Exception as e:
            self._record_failure(group_key, "Importes existentes", str(e))
        
        return self._get_group_success(group_key)
    
    async def validate_integration(self) -> bool:
        """Validar integración completa del sistema"""
        group_key = "integration"
        
        # Test 1: Pipeline completo con configuración nueva
        try:
            config = get_app_config()
            if self.container:
                service = self.container.get_service("data_processing")
                
                # Verificar que el servicio usa la configuración nueva
                assert service.embedding_model is not None, "Modelo de embeddings no cargado"
                self._record_success(group_key, "Pipeline con configuración nueva")
        except Exception as e:
            self._record_failure(group_key, "Pipeline con configuración nueva", str(e))
        
        # Test 2: Gestión de recursos
        try:
            if self.container:
                # Health check completo
                health = await self.container.health_check()
                
                # Verificar todos los servicios están saludables
                for service_name, service_health in health["services"].items():
                    if isinstance(service_health, dict):
                        assert service_health.get("status") == "healthy", f"Servicio {service_name} no saludable"
                
                self._record_success(group_key, "Gestión de recursos")
        except Exception as e:
            self._record_failure(group_key, "Gestión de recursos", str(e))
        
        # Test 3: Cleanup y shutdown
        try:
            if self.container:
                await self.container.shutdown()
                # Reinicializar para otros tests
                await self.container.initialize()
                self._record_success(group_key, "Cleanup y shutdown")
        except Exception as e:
            self._record_failure(group_key, "Cleanup y shutdown", str(e))
        
        # Test 4: Estadísticas y monitoreo
        try:
            if self.container:
                resilient_service = self.container.get_service("resilient_data_processing")
                stats = await resilient_service.get_resilience_status()
                
                assert "resilience_stats" in stats, "Estadísticas de resilencia no disponibles"
                assert "circuit_breakers" in stats, "Estado de circuit breakers no disponible"
                self._record_success(group_key, "Estadísticas y monitoreo")
        except Exception as e:
            self._record_failure(group_key, "Estadísticas y monitoreo", str(e))
        
        return self._get_group_success(group_key)
    
    def _record_success(self, group_key: str, test_name: str):
        """Registrar test exitoso"""
        self.results[group_key]["passed"] += 1
        self.results[group_key]["total"] += 1
        self.results[group_key]["details"].append(f"✅ {test_name}")
        logger.info(f"✅ {test_name}")
    
    def _record_failure(self, group_key: str, test_name: str, error: str):
        """Registrar test fallido"""
        self.results[group_key]["total"] += 1
        self.results[group_key]["details"].append(f"❌ {test_name}: {error}")
        logger.error(f"❌ {test_name}: {error}")
    
    def _get_group_success(self, group_key: str) -> bool:
        """Verificar si un grupo pasó todos los tests"""
        group_data = self.results[group_key]
        return group_data["passed"] == group_data["total"]
    
    async def generate_final_report(self, overall_success: bool):
        """Generar reporte final de validación"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 REPORTE FINAL - VALIDACIÓN FASE 1")
        logger.info("=" * 80)
        
        total_passed = 0
        total_tests = 0
        
        for group_name, group_data in self.results.items():
            group_display = group_name.replace("_", " ").title()
            passed = group_data["passed"]
            total = group_data["total"]
            success_rate = (passed / total * 100) if total > 0 else 0
            
            status_emoji = "✅" if passed == total else "❌"
            logger.info(f"{status_emoji} {group_display}: {passed}/{total} ({success_rate:.1f}%)")
            
            # Mostrar detalles de fallos
            for detail in group_data["details"]:
                if detail.startswith("❌"):
                    logger.info(f"    {detail}")
            
            total_passed += passed
            total_tests += total
        
        logger.info("-" * 80)
        overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"📈 RESULTADO GENERAL: {total_passed}/{total_tests} ({overall_rate:.1f}%)")
        
        if overall_success:
            logger.info("🎉 ¡FASE 1 COMPLETADA EXITOSAMENTE!")
            logger.info("✨ Todos los componentes de Architecture Base Refactoring están funcionando")
            logger.info("🚀 Sistema listo para Fase 2: Pipeline Optimization")
        else:
            logger.warning("⚠️ FASE 1 TIENE PROBLEMAS")
            logger.warning("🔧 Revisar y corregir fallos antes de continuar")
        
        logger.info("=" * 80)
    
    async def cleanup(self):
        """Limpiar recursos"""
        if self.container:
            await self.container.shutdown()

async def main():
    """Función principal de validación"""
    validator = Phase1Validator()
    
    try:
        success = await validator.run_complete_validation()
        return success
    except Exception as e:
        logger.error(f"💥 Error crítico en validación: {e}")
        return False
    finally:
        await validator.cleanup()

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit_code = 0 if success else 1
        logger.info(f"🏁 Validación terminada con código: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Validación interrumpida por usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Error fatal: {e}")
        sys.exit(1)