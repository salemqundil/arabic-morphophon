#!/usr/bin/env python3
"""
🧪 Arabic Comparative & Diminutive Test - Simple Validation
Enterprise-Grade Zero-Tolerance Testing
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
import_data os
sys.path.insert(0, os.path.abspath('.'))

from engines.nlp.derivation.models.comparative import_data (
    ArabicComparativeGenerator,
    ComparativeResult,
    DiminutiveResult
)

def test_comparative_basic():
    """Test basic comparative generation"""
    print("🔬 Testing Comparative Generation...")
    
    generator = ArabicComparativeGenerator()
    
    # Test cases
    test_cases = [
        (('ك', 'ب', 'ر'), 'أكبر'),  # bigger
        (('ص', 'غ', 'ر'), 'أصغر'),  # smaller
        (('ط', 'و', 'ل'), 'أطول'),  # longer
        (('ق', 'ص', 'ر'), 'أقصر'),  # shorter
        (('ج', 'م', 'ل'), 'أجمل'),  # more beautiful
    ]
    
    success_count = 0
    for i, (root, expected) in enumerate(test_cases, 1):
        try:
            result = generator.to_comparative(root)
            if result == expected:
                print(f"✅ Test {i}: {root} → {result} (PASS)")
                success_count += 1
            else:
                print(f"❌ Test {i}: {root} → {result}, expected {expected} (FAIL)")
        except Exception as e:
            print(f"💥 Test {i}: {root} → ERROR: {e}")
    
    print(f"📊 Comparative Results: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)

def test_diminutive_basic():
    """Test basic diminutive generation"""
    print("\n🔬 Testing Diminutive Generation...")
    
    generator = ArabicComparativeGenerator()
    
    # Test cases
    test_cases = [
        (('ق', 'ل', 'م'), 'قُلَيْم'),  # little pen
        (('ك', 'ت', 'ب'), 'كُتَيْب'),  # little book
        (('ب', 'ي', 'ت'), 'بُيَيْت'),  # little house
        (('ر', 'ج', 'ل'), 'رُجَيْل'),  # little man
        (('و', 'ل', 'د'), 'وُلَيْد'),  # little boy
    ]
    
    success_count = 0
    for i, (root, expected) in enumerate(test_cases, 1):
        try:
            result = generator.to_diminutive(root)
            if result == expected:
                print(f"✅ Test {i}: {root} → {result} (PASS)")
                success_count += 1
            else:
                print(f"❌ Test {i}: {root} → {result}, expected {expected} (FAIL)")
        except Exception as e:
            print(f"💥 Test {i}: {root} → ERROR: {e}")
    
    print(f"📊 Diminutive Results: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)

def test_metadata_generation():
    """Test metadata generation"""
    print("\n🔬 Testing Metadata Generation...")
    
    generator = ArabicComparativeGenerator()
    
    try:
        # Test comparative metadata
        comp_result = generator.generate_comparative_with_metadata(('ك', 'ب', 'ر'))
        print(f"✅ Comparative metadata: {comp_result.comparative_form}")
        print(f"   Pattern: {comp_result.morphological_pattern}")
        print(f"   Confidence: {comp_result.confidence}")
        
        # Test diminutive metadata
        dim_result = generator.generate_diminutive_with_metadata(('ق', 'ل', 'م'))
        print(f"✅ Diminutive metadata: {dim_result.diminutive_form}")
        print(f"   Pattern: {dim_result.morphological_pattern}")
        print(f"   Confidence: {dim_result.confidence}")
        
        return True
    except Exception as e:
        print(f"💥 Metadata test failed: {e}")
        return False

def test_integration_with_engine():
    """Test integration with main derivation engine"""
    print("\n🔬 Testing Engine Integration...")
    
    try:
        from engines.nlp.derivation.engine import_data DerivationEngine
        
        engine = DerivationEngine()
        
        # Test comparative
        comp1 = engine.generate_comparative(('ك', 'ب', 'ر'))
        comp2 = engine.generate_comparative('كبر')
        
        print(f"✅ Engine comparative (tuple): {comp1}")
        print(f"✅ Engine comparative (string): {comp2}")
        
        # Test diminutive
        dim1 = engine.generate_diminutive(('ق', 'ل', 'م'))
        dim2 = engine.generate_diminutive('قلم')
        
        print(f"✅ Engine diminutive (tuple): {dim1}")
        print(f"✅ Engine diminutive (string): {dim2}")
        
        # Test with metadata
        comp_meta = engine.generate_comparative_with_metadata('كبر')
        dim_meta = engine.generate_diminutive_with_metadata('قلم')
        
        print(f"✅ Comparative metadata: {comp_meta['comparative_form']}")
        print(f"✅ Diminutive metadata: {dim_meta['diminutive_form']}")
        
        return True
    except Exception as e:
        print(f"💥 Engine integration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Begining Arabic Comparative & Diminutive Validation")
    print("=" * 60)
    
    tests = [
        test_comparative_basic,
        test_diminutive_basic,
        test_metadata_generation,
        test_integration_with_engine
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"💥 Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 FINAL RESULTS: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🏆 ALL TESTS PASSED - ENTERPRISE STANDARDS MET!")
        return True
    else:
        print("❌ SOME TESTS FAILED - ZERO TOLERANCE VIOLATED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
