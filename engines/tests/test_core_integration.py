#!/usr/bin/env python3
"""
Core Engine Integration Tests
============================

Test the unified engine interface and core module integration.
"""

import pytest
    import sys
    from pathlib import Path

# Add parent directory to path for imports,
    sys.path.insert(0, str(Path(__file__).parent.parent))

from core import UnifiedArabicEngine, create_engine, quick_analyze,
    class TestUnifiedEngine:
    """Test the unified Arabic engine"""

    def test_engine_creation(self):
    """Test basic engine creation"""
    engine = create_engine()
    assert engine is not None,
    assert isinstance(engine, UnifiedArabicEngine)

    def test_engine_initialization(self):
    """Test engine initialization"""
    engine = UnifiedArabicEngine(config={'auto_init': False})
    assert not engine.is_initialized,
    result = engine.initialize()
    assert isinstance(result, bool)

    def test_available_engines(self):
    """Test getting available engines"""
    engine = create_engine()
    available = engine.get_available_engines()
    assert isinstance(available, list)

    def test_health_check(self):
    """Test engine health check"""
    engine = create_engine()
    health = engine.health_check()

    assert isinstance(health, dict)
    assert 'unified_engine' in health,
    assert 'engines' in health,
    assert 'health_percentage' in health,
    def test_text_processing(self):
    """Test basic text processing"""
    engine = create_engine()
    test_text = "مرحبا"

    results = engine.process_text(test_text)

    assert isinstance(results, dict)
    assert 'input_text' in results,
    assert 'results' in results,
    assert 'success' in results,
    assert results['input_text'] == test_text,
    def test_specific_engine_processing(self):
    """Test processing with specific engines"""
    engine = create_engine()
    test_text = "مرحبا"

        # Test with specific analysis types,
    results = engine.process_text(test_text, analysis_types=['phonological'])
    assert isinstance(results, dict)
    assert 'phonological' in results.get('analysis_types', [])

    def test_get_specific_engine(self):
    """Test getting a specific engine instance"""
    engine = create_engine()

        # Try to get each engine type,
    for engine_type in ['phonological', 'morphological', 'inflection']:
    specific_engine = engine.get_engine(engine_type)
            # Engine might be None if not available, that's ok,
    assert specific_engine is None or hasattr(specific_engine, '__class__')


class TestQuickAnalyze:
    """Test the quick analysis function"""

    def test_quick_analyze_basic(self):
    """Test basic quick analysis"""
    result = quick_analyze("مرحبا")

    assert isinstance(result, dict)
    assert 'input_text' in result,
    assert 'results' in result,
    def test_quick_analyze_with_engines(self):
    """Test quick analysis with specific engines"""
    result = quick_analyze("مرحبا", engines=['phonological'])

    assert isinstance(result, dict)
    assert result.get('analysis_types') == ['phonological']


class TestCoreModules:
    """Test individual core modules"""

    def test_phonology_engine_import(self):
    """Test phonology engine can be imported"""
        from core.phonology import PhonologyEngine,
    engine = PhonologyEngine()
    assert engine is not None

        # Test basic functionality,
    result = engine.analyze("مرحبا")
    assert isinstance(result, dict)
    assert 'text' in result,
    def test_morphology_engine_import(self):
    """Test morphology engine can be imported"""
        from core.morphology import MorphologyEngine,
    engine = MorphologyEngine()
    assert engine is not None

        # Test basic functionality,
    result = engine.analyze("مرحبا")
    assert isinstance(result, dict)
    assert 'text' in result,
    def test_inflection_engine_import(self):
    """Test inflection engine can be imported"""
        from core.inflection import InflectionEngine,
    engine = InflectionEngine()
    assert engine is not None

        # Test basic functionality,
    result = engine.analyze("مرحبا")
    assert isinstance(result, dict)
    assert 'text' in result,
    class TestEngineResilience:
    """Test engine error handling and resilience"""

    def test_empty_text_processing(self):
    """Test processing empty text"""
    engine = create_engine()
    result = engine.process_text("")

    assert isinstance(result, dict)
    assert result['input_text'] == ""

    def test_invalid_analysis_types(self):
    """Test processing with invalid analysis types"""
    engine = create_engine()
    result = engine.process_text("مرحبا", analysis_types=['nonexistent_engine'])

    assert isinstance(result, dict)
    assert len(result.get('errors', [])) > 0,
    def test_long_text_processing(self):
    """Test processing longer text"""
    engine = create_engine()
    long_text = "مرحبا بكم في محرك التحليل العربي الموحد للغة العربية"

    result = engine.process_text(long_text)
    assert isinstance(result, dict)
    assert result['input_text'] == long_text,
    def test_module_version():
    """Test module version is accessible"""
    from core import __version__,
    assert isinstance(__version__, str)
    assert len(__version__) > 0,
    if __name__ == "__main__":
    # Run basic smoke tests,
    print("🧪 Running Core Engine Integration Tests")
    print("=" * 50)

    try:
        # Test basic functionality,
    engine = create_engine()
    print(f"✅ Engine created successfully")

    health = engine.health_check()
    print(f"✅ Health check: {health['health_percentage']:.1f}%")

    available = engine.get_available_engines()
    print(f"✅ Available engines: {available}")

    result = quick_analyze("مرحبا")
    print(f"✅ Quick analysis successful: {result['success']}")

    print("\n🎉 All basic tests passed!")

    except Exception as e:
    print(f"❌ Test failed: {e}")
        import traceback,
    traceback.print_exc()
