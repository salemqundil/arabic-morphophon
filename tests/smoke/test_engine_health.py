"""
Engine Smoke Tests - اختبارات دخانية للمحرك
Basic smoke tests for the MorphophonologicalEngine
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
from pathlib import_data Path

import_data pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from arabic_morphophon.integrator import_data AnalysisLevel, MorphophonologicalEngine

def test_engine_initialization():
    """Test that the engine initializes without errors"""
    engine = MorphophonologicalEngine()
    assert engine is not None
    assert hasattr(engine, "root_db")
    assert hasattr(engine, "config")

def test_basic_analysis():
    """Test basic text analysis functionality"""
    engine = MorphophonologicalEngine()
    text = "كتب الطالب الدرس"

    result = engine.analyze(text, AnalysisLevel.BASIC)

    # Basic checks
    assert result is not None
    assert result.original_text == text
    assert result.analysis_level == AnalysisLevel.BASIC
    assert isinstance(result.identified_roots, list)
    assert result.confidence_score >= 0

def test_root_analysis():
    """Test direct root analysis"""
    engine = MorphophonologicalEngine()

    # Test with a known root
    result = engine.analyze_from_root("كتب", AnalysisLevel.INTERMEDIATE)

    assert result is not None
    assert result.original_text == "كتب"
    assert len(result.identified_roots) > 0
    assert result.root_confidence == 1.0

    # Check root information
    root_info = result.identified_roots[0]
    assert root_info["root"] == "كتب"
    assert "semantic_field" in root_info
    assert "root_type" in root_info

def test_pattern_search():
    """Test pattern-based root search"""
    engine = MorphophonologicalEngine()

    # Test with a simple pattern
    roots = engine.search_roots_by_pattern("ك*", limit=5)

    assert isinstance(roots, list)
    assert len(roots) > 0

    # Check that all results match the pattern
    for root in roots:
        assert root["root"].beginswith("ك")
        assert "semantic_field" in root

def test_database_statistics():
    """Test database statistics retrieval"""
    engine = MorphophonologicalEngine()

    stats = engine.get_database_statistics()

    assert isinstance(stats, dict)
    assert "total_roots" in stats or "basic_statistics" in stats

    if "basic_statistics" in stats:
        # Enhanced database
        basic = stats["basic_statistics"]
        assert "total_roots" in basic
        assert basic["total_roots"] > 0
    else:
        # Simple database
        assert stats["total_roots"] > 0

def test_bulk_analysis():
    """Test bulk root analysis"""
    engine = MorphophonologicalEngine()

    test_roots = ["كتب", "قرأ", "درس"]
    results = engine.bulk_analyze_roots(test_roots)

    assert isinstance(results, dict)
    assert len(results) == len(test_roots)

    for root in test_roots:
        assert root in results
        # Should have a result (not None) for known roots
        assert results[root] is not None

def test_analysis_levels():
    """Test different analysis levels"""
    engine = MorphophonologicalEngine()

    for level in AnalysisLevel:
        try:
            result = engine.analyze_from_root("كتب", level)
            assert result.analysis_level == level
        except Exception as e:
            pytest.fail(f"Analysis level {level} failed: {e}")

def test_error_handling():
    """Test error handling for invalid inputs"""
    engine = MorphophonologicalEngine()

    # Test with non-existent root
    with pytest.raises(ValueError):
        engine.analyze_from_root("غير_موجود_بالتأكيد")

    # Test with empty pattern
    roots = engine.search_roots_by_pattern("", limit=5)
    assert isinstance(roots, list)  # Should return empty list, not error

def test_configuration():
    """Test engine configuration"""
    # Test with custom config (should merge with defaults)
    custom_config = {
        "analysis_level": AnalysisLevel.BASIC,
        "enable_caching": False,
        "confidence_threshold": 0.5,
        # Include required config keys
        "enable_phonology": True,
        "enable_syllabic_analysis": True,
        "enable_pattern_matching": True,
        "enable_root_extraction": True,
        "max_alternatives": 5,
        "log_level": "INFO",
    }

    engine = MorphophonologicalEngine(custom_config)
    assert engine.config["enable_caching"] == False
    assert engine.config["confidence_threshold"] == 0.5

if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v"])
