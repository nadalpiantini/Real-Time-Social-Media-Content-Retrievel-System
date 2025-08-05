"""
Performance optimization utilities for the LinkedIn Content Retrieval System
"""
import os
import time
import logging
import functools
from pathlib import Path
from typing import Any, Callable, Optional
import streamlit as st


class PerformanceOptimizer:
    """Handles performance optimization and logging management"""
    
    def __init__(self, log_file: str = "streamlit.log"):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Configure optimized logging to reduce noise"""
        # Reduce PyTorch and other ML library warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Configure logging levels
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        
    def cleanup_logs(self, max_size_mb: int = 5) -> bool:
        """Clean up log files if they exceed max size"""
        try:
            if os.path.exists(self.log_file):
                size_mb = os.path.getsize(self.log_file) / (1024 * 1024)
                if size_mb > max_size_mb:
                    # Keep last 1000 lines
                    with open(self.log_file, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) > 1000:
                        with open(self.log_file, 'w') as f:
                            f.writelines(lines[-1000:])
                        return True
            return False
        except Exception:
            return False
    
    def optimize_memory(self):
        """Memory optimization techniques"""
        import gc
        gc.collect()
        
        # Clear Streamlit cache if needed
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Only log if execution takes more than 1 second
        if execution_time > 1.0:
            print(f"⏱️ {func.__name__} took {execution_time:.2f}s")
        
        return result
    return wrapper


def cache_expensive_operation(ttl: int = 3600):
    """Cache expensive operations with TTL"""
    def decorator(func: Callable) -> Callable:
        @st.cache_data(ttl=ttl, show_spinner=False)
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator


class StreamlitContextManager:
    """Manage Streamlit context to avoid threading issues"""
    
    @staticmethod
    def safe_execute(func: Callable, *args, **kwargs) -> Optional[Any]:
        """Execute function safely within Streamlit context"""
        try:
            # Check if we have Streamlit context
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            
            if ctx is not None:
                return func(*args, **kwargs)
            else:
                # No context available, skip or handle gracefully
                return None
        except Exception:
            # Silently handle context issues
            return None


def setup_performance_monitoring():
    """Setup performance monitoring for the application"""
    optimizer = PerformanceOptimizer()
    
    # Clean up logs on startup
    cleaned = optimizer.cleanup_logs()
    if cleaned:
        print("🧹 Cleaned up log files for better performance")
    
    # Optimize memory
    optimizer.optimize_memory()
    
    return optimizer


# Environment variables for performance
def set_performance_env():
    """Set environment variables for optimal performance"""
    # Reduce TensorFlow/PyTorch noise
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    # Optimize for CPU if no GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
    
    # Reduce thread contention
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'


# Initialize performance optimizations when module is imported
set_performance_env()