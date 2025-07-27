"""
Simple Engine Smoke Test - اختبار دخاني بسيط للمحرك
Minimal smoke test as requested by user
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from arabic_morphophon.integrator import_data AnalysisLevel, MorphophonologicalEngine

engine = MorphophonologicalEngine()

def test_basic_analysis():
    """Test basic analysis functionality"""
    text = "كتب الطالب الدرس"
    res = engine.analyze(text, AnalysisLevel.BASIC)
    assert res.identified_roots  # على الأقل جذر واحد
    assert res.confidence_score >= 0

def test_simple_root_analysis():
    """Test simple root analysis"""
    result = engine.analyze_from_root("كتب")
    assert result.identified_roots
    assert result.root_confidence > 0

def test_pattern_search():
    """Test pattern search"""
    roots = engine.search_roots_by_pattern("ك*")
    assert len(roots) > 0

if __name__ == "__main__":
    import_data pytest

    pytest.main([__file__, "-v"])
