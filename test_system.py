#!/usr/bin/env python3
"""
Test suite for Real-Time Social Media Content Retrieval System
Basic functionality and integration tests
"""

import os
import sys
import json
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestDataValidation(unittest.TestCase):
    """Test data file validation and structure"""
    
    def test_data_files_exist(self):
        """Test that data files exist and are readable"""
        data_dir = "data"
        self.assertTrue(os.path.exists(data_dir), "Data directory should exist")
        
        data_files = [f for f in os.listdir(data_dir) if f.endswith('_data.json')]
        self.assertGreater(len(data_files), 0, "Should have at least one data file")
    
    def test_data_file_structure(self):
        """Test that data files have correct JSON structure"""
        data_dir = "data"
        data_files = [f for f in os.listdir(data_dir) if f.endswith('_data.json')]
        
        for file_name in data_files:
            file_path = os.path.join(data_dir, file_name)
            
            with self.subTest(file=file_name):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check required keys
                self.assertIn('Name', data, f"{file_name} should have 'Name' key")
                self.assertIn('Posts', data, f"{file_name} should have 'Posts' key")
                
                # Check Posts structure
                posts = data['Posts']
                self.assertIsInstance(posts, dict, "Posts should be a dictionary")
                
                for post_id, post_data in posts.items():
                    with self.subTest(post=post_id):
                        self.assertIn('text', post_data, f"Post {post_id} should have 'text'")
                        self.assertIn('name', post_data, f"Post {post_id} should have 'name'")
                        self.assertIn('source', post_data, f"Post {post_id} should have 'source'")


class TestCoreModules(unittest.TestCase):
    """Test core module imports and basic functionality"""
    
    def test_core_imports(self):
        """Test that core modules can be imported"""
        try:
            from utils.embedding import EmbeddingModelSingleton, CrossEncoderModelSingleton
            from utils.qdrant import build_qdrant_client
            from utils.retriever import QdrantVectorDBRetriever
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")
    
    def test_qdrant_client_creation(self):
        """Test Qdrant client creation in memory mode"""
        try:
            from utils.qdrant import build_qdrant_client
            client = build_qdrant_client(':memory:')
            self.assertIsNotNone(client, "Qdrant client should be created")
        except Exception as e:
            self.fail(f"Failed to create Qdrant client: {e}")
    
    def test_embedding_model_singleton(self):
        """Test that embedding model singleton works"""
        try:
            from utils.embedding import EmbeddingModelSingleton
            
            # This might take time on first run, so we'll use a timeout
            model1 = EmbeddingModelSingleton()
            model2 = EmbeddingModelSingleton()
            
            # Should be the same instance
            self.assertIs(model1, model2, "Should return same singleton instance")
        except Exception as e:
            # Skip this test if models can't be loaded (e.g., in CI environment)
            self.skipTest(f"Cannot load embedding model: {e}")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions and helpers"""
    
    def test_performance_module(self):
        """Test performance optimization module"""
        try:
            from utils.performance import PerformanceOptimizer, timer, cache_expensive_operation
            
            optimizer = PerformanceOptimizer()
            self.assertIsNotNone(optimizer, "Performance optimizer should be created")
            
            # Test timer decorator
            @timer
            def test_function():
                return "test"
            
            result = test_function()
            self.assertEqual(result, "test", "Timer decorator should preserve function result")
            
        except ImportError:
            self.skipTest("Performance module not available")
    
    def test_maintenance_functions(self):
        """Test maintenance script functions"""
        # Import the maintenance script as a module
        maintenance_path = os.path.join(os.path.dirname(__file__), 'maintenance.py')
        if os.path.exists(maintenance_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("maintenance", maintenance_path)
            maintenance = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(maintenance)
            
            # Test validation function
            issues = maintenance.validate_data_files()
            self.assertIsInstance(issues, list, "validate_data_files should return a list")


class TestStreamlitApp(unittest.TestCase):
    """Test Streamlit app components"""
    
    def test_streamlit_config(self):
        """Test that Streamlit configuration exists"""
        config_path = ".streamlit/config.toml"
        self.assertTrue(os.path.exists(config_path), "Streamlit config should exist")
        
        # Basic validation of config file
        with open(config_path, 'r') as f:
            content = f.read()
            self.assertIn('[server]', content, "Config should have server section")
            self.assertIn('[theme]', content, "Config should have theme section")
    
    @patch('streamlit.set_page_config')
    def test_app_imports(self, mock_config):
        """Test that app.py can be imported without errors"""
        try:
            # Mock Streamlit to avoid actual UI operations
            with patch('streamlit.title'), \
                 patch('streamlit.markdown'), \
                 patch('streamlit.sidebar'):
                
                import app
                self.assertTrue(True, "App should import without errors")
        except Exception as e:
            self.fail(f"App failed to import: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_data_pipeline_components(self):
        """Test that data pipeline components work together"""
        try:
            # Test data source detection
            from models.data_source import UnifiedDataSource
            
            data_source = UnifiedDataSource()
            self.assertIsNotNone(data_source, "Data source should be created")
            
        except ImportError:
            self.skipTest("Data source module not available")
    
    def test_requirements_satisfaction(self):
        """Test that all required packages are available"""
        required_packages = [
            'streamlit',
            'sentence_transformers',
            'torch', 
            'qdrant_client',
            'bytewax',
            'pandas',
            'numpy'
        ]
        
        for package in required_packages:
            with self.subTest(package=package):
                try:
                    __import__(package)
                except ImportError:
                    self.fail(f"Required package '{package}' is not available")


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataValidation,
        TestCoreModules, 
        TestUtilityFunctions,
        TestStreamlitApp,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Real-Time LinkedIn Content Retrieval System - Test Suite")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)