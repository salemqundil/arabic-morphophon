#!/usr/bin/env python3
"""
Comprehensive Engine Test Suite
===============================

Expanded test coverage for all Arabic NLP engines.
"""

import pytest
    import sys
    from pathlib import Path
    from typing import Dict, Any, List

# Add parent directory to path for imports,
    sys.path.insert(0, str(Path(__file__).parent.parent))

from core import create_engine, UnifiedArabicEngine,
    class TestMorphologyEngine:
    """Test morphological analysis engine"""

    def test_morphology_basic(self):
    """Test basic morphological analysis"""
    engine = create_engine()
    result = engine.process_text("كتاب", analysis_types=['morphological'])

    assert isinstance(result, dict)
    assert result['success'] is True,
    assert 'morphological' in result.get('results', {})

    def test_morphology_complex_word(self):
    """Test morphological analysis of complex word"""
    engine = create_engine()
    result = engine.process_text("استكشافات", analysis_types=['morphological'])

    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_morphology_multiple_words(self):
    """Test morphological analysis of multiple words"""
    engine = create_engine()
    result = engine.process_text("الكتاب الجديد", analysis_types=['morphological'])

    assert isinstance(result, dict)
    assert result['success'] is True,
    class TestInflectionEngine:
    """Test inflectional analysis engine"""

    def test_inflection_verb(self):
    """Test inflection analysis of verbs"""
    engine = create_engine()
    result = engine.process_text("يكتب", analysis_types=['inflection'])

    assert isinstance(result, dict)
    assert result['success'] is True,
    assert 'inflection' in result.get('results', {})

    def test_inflection_noun(self):
    """Test inflection analysis of nouns"""
    engine = create_engine()
    result = engine.process_text("كتابان", analysis_types=['inflection'])

    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_inflection_adjective(self):
    """Test inflection analysis of adjectives"""
    engine = create_engine()
    result = engine.process_text("جميلة", analysis_types=['inflection'])

    assert isinstance(result, dict)
    assert result['success'] is True,
    class TestWeightEngine:
    """Test morphological weight analysis engine"""

    def test_weight_basic(self):
    """Test basic weight analysis"""
    engine = create_engine()
    result = engine.process_text("فعل", analysis_types=['weight'])

    assert isinstance(result, dict)
    assert result['success'] is True,
    assert 'weight' in result.get('results', {})

    def test_weight_patterns(self):
    """Test weight pattern analysis"""
    test_words = ["فعل", "فاعل", "مفعول", "استفعال"]
    engine = create_engine()

        for word in test_words:
    result = engine.process_text(word, analysis_types=['weight'])
    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_weight_complex(self):
    """Test weight analysis of complex words"""
    engine = create_engine()
    result = engine.process_text("استكشاف", analysis_types=['weight'])

    assert isinstance(result, dict)
    assert result['success'] is True,
    class TestMultiEngineIntegration:
    """Test multiple engines working together"""

    def test_all_available_engines(self):
    """Test processing with all available engines"""
    engine = create_engine()
    available_engines = engine.get_available_engines()

    result = engine.process_text("كتاب", analysis_types=available_engines)

    assert isinstance(result, dict)
    assert result['success'] is True

        # Check that results exist for each available engine,
    results = result.get('results', {})
        for engine_type in available_engines:
    assert engine_type in results,
    def test_engine_fallback(self):
    """Test fallback behavior when engines are unavailable"""
    engine = create_engine()

        # Try to use engines that might not be available,
    result = engine.process_text(
    "كتاب", analysis_types=['phonological', 'syllable']
    )

    assert isinstance(result, dict)
        # System should still succeed even if engines are unavailable,
    assert result['success'] is True,
    def test_mixed_available_unavailable(self):
    """Test mixing available and unavailable engines"""
    engine = create_engine()

        # Mix available engines with potentially unavailable ones,
    result = engine.process_text(
    "كتاب", analysis_types=['morphological', 'phonological', 'inflection']
    )

    assert isinstance(result, dict)
    assert result['success'] is True

        # Should have results from available engines,
    results = result.get('results', {})
    assert len(results) > 0,
    class TestArabicTextVariations:
    """Test various Arabic text inputs"""

    def test_short_text(self):
    """Test single character and short text"""
    engine = create_engine()

    test_cases = ["ك", "كت", "كتب"]
        for text in test_cases:
    result = engine.process_text(text)
    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_long_text(self):
    """Test longer Arabic text"""
    engine = create_engine()

    long_text = "هذا نص طويل للغة العربية يحتوي على كلمات متنوعة ومختلفة للاختبار"
    result = engine.process_text(long_text)

    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_text_with_diacritics(self):
    """Test Arabic text with diacritical marks"""
    engine = create_engine()

    diacritized_text = "كِتَابٌ جَمِيلٌ"
    result = engine.process_text(diacritized_text)

    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_mixed_content(self):
    """Test mixed Arabic and other content"""
    engine = create_engine()

    mixed_text = "كتاب 123 book"
    result = engine.process_text(mixed_text)

    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_edge_cases(self):
    """Test edge cases"""
    engine = create_engine()

    edge_cases = ["", "   ", "123", "!@#", "كتاب\n\t"]
        for text in edge_cases:
    result = engine.process_text(text)
    assert isinstance(result, dict)
    assert result['success'] is True,
    class TestPerformanceAndReliability:
    """Test performance and reliability aspects"""

    def test_repeated_calls(self):
    """Test engine reliability with repeated calls"""
    engine = create_engine()

        for i in range(10):
    result = engine.process_text(f"كتاب{i}")
    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_concurrent_processing(self):
    """Test multiple texts in sequence"""
    engine = create_engine()

    texts = ["كتاب", "قلم", "بيت", "شجرة", "نهر"]
        for text in texts:
    result = engine.process_text(text)
    assert isinstance(result, dict)
    assert result['success'] is True,
    def test_health_monitoring(self):
    """Test health monitoring functionality"""
    engine = create_engine()

    health = engine.health_check()
    assert isinstance(health, dict)
    assert 'health_percentage' in health,
    assert 'engines' in health,
    assert health['health_percentage'] >= 0,
    def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    print("🧪 Running Comprehensive Engine Test Suite")
    print("=" * 60)

    # Test categories,
    test_categories = [
    ("Morphology Engine", TestMorphologyEngine),
    ("Inflection Engine", TestInflectionEngine),
    ("Weight Engine", TestWeightEngine),
    ("Multi-Engine Integration", TestMultiEngineIntegration),
    ("Arabic Text Variations", TestArabicTextVariations),
    ("Performance & Reliability", TestPerformanceAndReliability),
    ]

    total_tests = 0,
    passed_tests = 0,
    failed_tests = 0,
    for category_name, test_class in test_categories:
    print(f"\n📋 Testing: {category_name}")
    print("-" * 40)

        # Get all test methods,
    test_methods = [
    method for method in dir(test_class) if method.startswith('test_')
    ]
    category_passed = 0,
    category_failed = 0,
    for method_name in test_methods:
    total_tests += 1,
    try:
                # Create instance and run test,
    test_instance = test_class()
    test_method = getattr(test_instance, method_name)
    test_method()

    print(f"  ✅ {method_name}")
    passed_tests += 1,
    category_passed += 1,
    except Exception as e:
    print(f"  ❌ {method_name}: {e}")
    failed_tests += 1,
    category_failed += 1,
    print(
    f"  📊 Category Result: {category_passed}/{category_passed} + category_failed} passed"
    )

    # Final summary,
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0,
    print(f"\n🎯 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 90:
    print("🟢 EXCELLENT: Test suite performing excellently")
    elif success_rate >= 70:
    print("🟡 GOOD: Test suite performing well")
    else:
    print("🔴 NEEDS ATTENTION: Some tests failing")


if __name__ == "__main__":
    run_comprehensive_tests()
