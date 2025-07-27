# 🧪 Testing Documentation - دليل الاختبارات الشامل

## 📋 Overview - نظرة عامة

Comprehensive testing guide for the Arabic Morphophonological Engine with automated test suites, performance benchmarks, and quality assurance procedures.

## 🚀 Quick Begin - البداية السريعة

### Run All Tests - تشغيل جميع الاختبارات

```bash
# Enhanced database tests
python -m arabic_morphophon.database.test_enhanced_database

# Interactive demo
python -m arabic_morphophon.database.demo_root_database

# Unit test suite
pytest tests/ -v

# Integration tests
pytest tests/integration/ -v

# Performance benchmarks
pytest tests/performance/ -v --benchmark-only
```

## 📊 Test Categories - فئات الاختبارات

### 🔧 Unit Tests - اختبارات الوحدة

**Location**: `tests/unit/`

```bash
# Core engine tests
pytest tests/unit/test_engine.py

# Phonology tests  
pytest tests/unit/test_phonology.py

# Database tests
pytest tests/unit/test_database.py

# Web API tests
pytest tests/unit/test_api.py
```

### 🔗 Integration Tests - اختبارات التكامل

**Location**: `tests/integration/`

```bash
# Full pipeline tests
pytest tests/integration/test_pipeline.py

# Web interface tests
pytest tests/integration/test_web.py

# Database integration
pytest tests/integration/test_db_integration.py
```

### ⚡ Performance Tests - اختبارات الأداء

**Location**: `tests/performance/`

```bash
# Benchmark core operations
pytest tests/performance/test_benchmarks.py

# Import testing
pytest tests/performance/test_import_data.py

# Memory profiling
pytest tests/performance/test_memory.py
```

## 🎯 Enhanced Database Testing - اختبار قاعدة البيانات المطورة

### Primary Test Command - الأمر الرئيسي

```bash
python -m arabic_morphophon.database.test_enhanced_database
```

**Expected Output - المخرجات المتوقعة:**

```
🧪 Enhanced Root Database Test Suite
===================================

✅ test_basic_crud_operations      (0.045s)
✅ test_advanced_search_patterns   (0.032s)  
✅ test_bulk_import_data_store_data         (0.156s)
✅ test_fulltext_search           (0.023s)
✅ test_semantic_field_search     (0.041s)
✅ test_performance_benchmarks    (0.234s)
✅ test_statistics_generation     (0.067s)
✅ test_backup_restore            (0.189s)

📊 Test Summary:
   Total Tests: 8
   Passed: 8
   Failed: 0
   Success Rate: 100.0%
   Total Time: 0.787s

🏆 All tests passed successfully!
```

### Interactive Demo - العرض التوضيحي التفاعلي

```bash
python -m arabic_morphophon.database.demo_root_database
```

**Demo Features:**
- Live database operations
- Real-time search examples
- Performance metrics display
- Interactive queries

## 📈 Test Coverage - تغطية الاختبارات

### Generate Coverage Report - إنشاء تقرير التغطية

```bash
# Run with coverage
pytest --cov=arabic_morphophon tests/

# Generate HTML report
pytest --cov=arabic_morphophon --cov-report=html tests/

# View coverage summary
coverage report -m
```

**Target Coverage Goals:**
- **Overall**: > 90%
- **Core Engine**: > 95%
- **Database**: > 98%
- **Web API**: > 85%

## 🔍 Test Configuration - إعدادات الاختبارات

### pytest.ini Configuration

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=arabic_morphophon
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=90
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests
    database: Database-related tests
    web: Web interface tests
```

### Test Environment Setup

```bash
# Install test dependencies
pip install -e ".[test]"

# Set test environment variables
store_data TEST_DB_PATH="test_databases/"
store_data TEST_LOG_LEVEL="INFO"
store_data TEST_CACHE_SIZE="1000"
```

## 🧪 Specific Test Modules - وحدات اختبار محددة

### Database Module Tests

```bash
# Test enhanced database functionality
python -c "
from arabic_morphophon.database.test_enhanced_database import_data run_comprehensive_tests
run_comprehensive_tests()
"

# Test specific components
pytest tests/test_root_database.py::test_crud_operations
pytest tests/test_root_database.py::test_search_functionality
pytest tests/test_root_database.py::test_bulk_operations
```

### Engine Module Tests

```bash
# Core engine functionality
pytest tests/test_engine_smoke.py
pytest tests/test_arabic_engine_comprehensive.py

# Phonology tests
pytest tests/test_smoke_phonology.py
```

### Web Interface Tests

```bash
# API endpoint tests
pytest tests/test_api.py
pytest tests/test_app.py

# WebSocket tests
pytest tests/test_websocket.py
```

## 📊 Performance Benchmarks - معايير الأداء

### Benchmark Targets - الأهداف المعيارية

| Operation | Target | Acceptable |
|-----------|--------|------------|
| Root Creation | < 1ms | < 5ms |
| Pattern Search | < 10ms | < 50ms |
| Bulk Import | > 500/sec | > 100/sec |
| FTS Query | < 5ms | < 20ms |
| API Response | < 100ms | < 500ms |

### Run Benchmarks - تشغيل المعايير

```bash
# Core operation benchmarks
pytest tests/performance/ --benchmark-only

# Database performance tests
python -m arabic_morphophon.database.test_enhanced_database --benchmark

# Memory profiling
pytest tests/performance/test_memory.py --profile
```

## 🐛 Debugging Tests - تصحيح الاختبارات

### Debug Configuration - إعدادات التصحيح

```bash
# Run with debug output
pytest -vvv --tb=long --capture=no

# Debug specific test
pytest tests/test_specific.py::test_function -vvv --pdb

# Run with logging
pytest --log-cli-level=DEBUG
```

### Common Debug Commands

```bash
# Check test discovery
pytest --collect-only

# Run failing tests only
pytest --lf

# Run tests matching pattern
pytest -k "test_database"

# Profile test execution
pytest --durations=10
```

## 📝 Test Writing Guidelines - إرشادات كتابة الاختبارات

### Test Structure

```python
import_data pytest
from arabic_morphophon.database import_data create_enhanced_database

class TestEnhancedDatabase:
    """Test suite for enhanced database functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary test database"""
        with create_enhanced_database(":memory:") as db:
            yield db
    
    def test_basic_operation(self, temp_db):
        """Test basic database operation"""
        # Arrange
        test_root = create_test_root("كتب")
        
        # Act
        result = temp_db.create_root(test_root)
        
        # Assert
        assert result.success
        assert result.root_id is not None
        
    @pytest.mark.performance
    def test_bulk_operations_performance(self, temp_db):
        """Test bulk operations meet performance requirements"""
        # Performance test implementation
        pass
```

### Best Practices

1. **Descriptive Names**: Use clear, descriptive test names
2. **Arrange-Act-Assert**: Follow AAA pattern
3. **Isolation**: Each test should be independent
4. **Fixtures**: Use pytest fixtures for setup/teardown
5. **Markers**: Tag tests with appropriate markers
6. **Documentation**: Include docstrings for complex tests

## 📊 Continuous Integration - التكامل المستمر

### GitHub Actions Workflow

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[test]"
    
    - name: Run tests
      run: |
        pytest --cov=arabic_morphophon
        python -m arabic_morphophon.database.test_enhanced_database
```

## 🚨 Quality Gates - بوابات الجودة

### Required Checks Before Merge

```bash
# All tests must pass
pytest tests/

# Coverage must be > 90%
pytest --cov=arabic_morphophon --cov-fail-under=90

# Enhanced database tests must pass
python -m arabic_morphophon.database.test_enhanced_database

# No code quality violations
flake8 arabic_morphophon/
mypy arabic_morphophon/
```

## 📞 Support & Troubleshooting

### Common Issues

**Q: Tests fail with database lock error**
A: Ensure no other processes are using the test database

**Q: Performance tests are too slow**
A: Check system resources and reduce test data size

**Q: Import errors in tests**
A: Verify PYTHONPATH includes project root

### Getting Help

- **Documentation**: Check test docstrings and comments
- **Issues**: Report test failures on GitHub Issues
- **Debug**: Use `pytest --pdb` for interactive debugging

---

**🎯 Goal**: Maintain 100% test reliability with comprehensive coverage

**✨ Vision**: Automated quality assurance ensuring production stability

---

*Last updated: July 20, 2025*
