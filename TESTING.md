# Testing Guide

## Overview

This document describes the testing strategy and available tests for the Real-Time LinkedIn Content Retrieval System.

## Test Suite

### Running Tests

```bash
# Run all tests
python test_system.py

# Run maintenance checks
python maintenance.py --check-only

# Run performance optimization
python maintenance.py --full
```

### Test Categories

#### 1. Data Validation Tests (`TestDataValidation`)
- **test_data_files_exist**: Verifies data directory and files exist
- **test_data_file_structure**: Validates JSON structure of data files

#### 2. Core Module Tests (`TestCoreModules`)
- **test_core_imports**: Ensures all core modules can be imported
- **test_qdrant_client_creation**: Tests vector database client creation
- **test_embedding_model_singleton**: Validates AI model singleton pattern

#### 3. Utility Function Tests (`TestUtilityFunctions`)
- **test_performance_module**: Tests performance optimization utilities
- **test_maintenance_functions**: Validates maintenance script functions

#### 4. Streamlit App Tests (`TestStreamlitApp`)
- **test_streamlit_config**: Validates Streamlit configuration
- **test_app_imports**: Tests app module imports (with mocking)

#### 5. Integration Tests (`TestIntegration`)
- **test_data_pipeline_components**: Tests component integration
- **test_requirements_satisfaction**: Validates all dependencies

## Test Results Interpretation

### ✅ Success Indicators
- All data files have valid JSON structure
- Core modules import successfully
- Qdrant in-memory client works
- Performance optimizations are functional
- All required dependencies are available

### ⚠️ Known Limitations
- Embedding model tests may be skipped in CI environments
- App import tests require careful Streamlit mocking
- Vector database persistence tests require Docker

## Manual Testing

### 1. System Functionality
```bash
# 1. Start the application
streamlit run app.py

# 2. Test data processing
# - Navigate to "Step 2: Process Data for AI Search"
# - Click "Start Processing"
# - Verify successful completion

# 3. Test search functionality
# - Enter a query like "machine learning"
# - Verify results are returned
# - Check result relevance scores
```

### 2. Performance Testing
```bash
# Check system resources before/after
python maintenance.py --check-only

# Monitor logs during operation
tail -f streamlit.log

# Test with different query complexities
# - Simple queries: "python"
# - Complex queries: "machine learning best practices for startups"
```

### 3. Data Quality Testing
```bash
# Validate data file integrity
python -c "
import json
with open('data/manthanbhikadiya_data.json') as f:
    data = json.load(f)
    print(f'Posts: {len(data[\"Posts\"])}')
    print(f'Structure: {list(data.keys())}')
"
```

## Continuous Integration

### Prerequisites
- Python 3.11+
- All dependencies from requirements.txt
- Data files present in `data/` directory

### CI Pipeline
```yaml
# Example GitHub Actions workflow
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python test_system.py
      - run: python maintenance.py --check-only
```

## Troubleshooting Tests

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Loading Failures**
   - First run downloads ~500MB of AI models
   - Ensure stable internet connection
   - Models are cached after first download

3. **Qdrant Connection Issues**
   - Tests use in-memory mode by default
   - Docker not required for basic testing
   - Full persistence requires Docker setup

4. **Data File Issues**
   ```bash
   python maintenance.py --check-only
   ```

### Test Environment Setup

```bash
# Development environment
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Verify setup
python -c "import streamlit, torch, qdrant_client; print('✅ Core deps OK')"
```

## Performance Benchmarks

### Expected Performance
- **Data Processing**: ~1-2 seconds per post
- **Model Loading**: ~30-60 seconds (first run only)
- **Search Query**: <1 second response time
- **Memory Usage**: ~2GB RAM for AI models

### Optimization Targets
- Log file size: <5MB
- Cache cleanup: Weekly recommended
- Model caching: Persistent across sessions
- Vector search: Sub-second response

## Test Coverage

Current test coverage includes:
- ✅ Data validation and structure
- ✅ Core module functionality  
- ✅ Performance optimizations
- ✅ Configuration validation
- ✅ Dependency satisfaction
- ⚠️ UI testing (limited due to Streamlit)
- ⚠️ End-to-end workflows (manual testing)

## Adding New Tests

### Test Structure
```python
class TestNewFeature(unittest.TestCase):
    def test_specific_functionality(self):
        # Arrange
        setup_test_data()
        
        # Act
        result = call_function_under_test()
        
        # Assert
        self.assertEqual(result, expected_value)
```

### Best Practices
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies (Streamlit, network calls)
- Add performance assertions where relevant
- Document test purpose and expected behavior