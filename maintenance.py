#!/usr/bin/env python3
"""
Maintenance script for Real-Time Social Media Content Retrieval System
Performs cleanup and optimization tasks for better performance
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse


def cleanup_logs(max_size_mb=5, keep_lines=1000):
    """Clean up log files that exceed maximum size"""
    log_files = ["streamlit.log", "app.log", "error.log"]
    cleaned_files = []
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size_mb = os.path.getsize(log_file) / (1024 * 1024)
            
            if size_mb > max_size_mb:
                print(f"🧹 Cleaning {log_file} ({size_mb:.2f}MB)")
                
                # Keep only the last N lines
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                if len(lines) > keep_lines:
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.writelines(lines[-keep_lines:])
                    
                    cleaned_files.append(log_file)
                    new_size_mb = os.path.getsize(log_file) / (1024 * 1024)
                    print(f"✅ {log_file} reduced to {new_size_mb:.2f}MB")
    
    return cleaned_files


def cleanup_cache():
    """Clean up cache directories and temporary files"""
    cache_dirs = [
        ".streamlit/cache",
        "__pycache__",
        ".cache",
        "temp",
        "tmp"
    ]
    
    cleaned_dirs = []
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                cleaned_dirs.append(cache_dir)
                print(f"🗑️ Removed cache directory: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Could not remove {cache_dir}: {e}")
    
    # Clean Python bytecode files
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                try:
                    os.remove(os.path.join(root, file))
                except Exception:
                    pass
    
    return cleaned_dirs


def validate_data_files():
    """Validate and fix data files"""
    data_dir = "data"
    issues_found = []
    
    if not os.path.exists(data_dir):
        print(f"⚠️ Data directory '{data_dir}' not found")
        return ["missing_data_dir"]
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('_data.json')]
    
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check structure
            if 'Name' not in data or 'Posts' not in data:
                print(f"⚠️ Invalid structure in {file_name}")
                issues_found.append(f"invalid_structure_{file_name}")
            
            # Check if empty
            posts_count = len(data.get('Posts', {}))
            if posts_count == 0:
                print(f"⚠️ Empty posts in {file_name}")
                issues_found.append(f"empty_posts_{file_name}")
            else:
                print(f"✅ {file_name}: {posts_count} posts")
                
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {file_name}")
            issues_found.append(f"invalid_json_{file_name}")
        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")
            issues_found.append(f"read_error_{file_name}")
    
    return issues_found


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit',
        'sentence-transformers', 
        'torch',
        'qdrant-client',
        'bytewax',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages


def optimize_environment():
    """Set optimal environment variables"""
    optimizations = {
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_VERBOSITY': 'error',
        'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4'
    }
    
    for var, value in optimizations.items():
        os.environ[var] = value
        print(f"🔧 Set {var}={value}")
    
    return optimizations


def generate_report():
    """Generate maintenance report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd()
        },
        'maintenance_actions': []
    }
    
    return report


def main():
    """Main maintenance function"""
    parser = argparse.ArgumentParser(description='Maintenance script for LinkedIn Content Retrieval System')
    parser.add_argument('--logs-only', action='store_true', help='Only clean up log files')
    parser.add_argument('--cache-only', action='store_true', help='Only clean up cache files')
    parser.add_argument('--check-only', action='store_true', help='Only check system status')
    parser.add_argument('--full', action='store_true', help='Full maintenance (default)')
    
    args = parser.parse_args()
    
    print("🔧 LinkedIn Content Retrieval System - Maintenance Script")
    print("=" * 60)
    
    report = generate_report()
    
    if args.logs_only:
        cleaned_logs = cleanup_logs()
        report['maintenance_actions'].append({'action': 'log_cleanup', 'files': cleaned_logs})
    
    elif args.cache_only:
        cleaned_cache = cleanup_cache()
        report['maintenance_actions'].append({'action': 'cache_cleanup', 'dirs': cleaned_cache})
    
    elif args.check_only:
        print("\n📊 System Status Check:")
        data_issues = validate_data_files()
        missing_deps = check_dependencies()
        
        report['maintenance_actions'].extend([
            {'action': 'data_validation', 'issues': data_issues},
            {'action': 'dependency_check', 'missing': missing_deps}
        ])
        
    else:
        # Full maintenance (default)
        print("\n🧹 Cleaning up log files...")
        cleaned_logs = cleanup_logs()
        
        print("\n🗑️ Cleaning up cache files...")
        cleaned_cache = cleanup_cache()
        
        print("\n📊 Validating data files...")
        data_issues = validate_data_files()
        
        print("\n📦 Checking dependencies...")
        missing_deps = check_dependencies()
        
        print("\n🔧 Optimizing environment...")
        optimized_env = optimize_environment()
        
        report['maintenance_actions'].extend([
            {'action': 'log_cleanup', 'files': cleaned_logs},
            {'action': 'cache_cleanup', 'dirs': cleaned_cache},
            {'action': 'data_validation', 'issues': data_issues},
            {'action': 'dependency_check', 'missing': missing_deps},
            {'action': 'environment_optimization', 'variables': list(optimized_env.keys())}
        ])
    
    # Save report
    with open('maintenance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Maintenance completed - Report saved to maintenance_report.json")
    
    # Summary
    total_actions = len(report['maintenance_actions'])
    print(f"📋 Summary: {total_actions} maintenance actions performed")


if __name__ == "__main__":
    main()