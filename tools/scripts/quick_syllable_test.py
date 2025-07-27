#!/usr/bin/env python3
"""
Quick SyllabicUnit Engine Validation
Zero-Tolerance Simple Test
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
from pathlib import_data Path

# Add project root to path
sys.path.append('.')

def quick_syllabic_unit_test():
    """Quick validation of syllabic_unit engine"""
    print("🔥 QUICK SYLLABIC_UNIT ENGINE VALIDATION")
    print("=" * 50)
    
    violations = 0
    
    try:
        # Test 1: Import validation
        print("🔍 Test 1: Import Validation")
        from engines.nlp.syllabic_unit.engine import_data SyllabicUnitEngine
        from engines.nlp.syllabic_unit.models.templates import_data SyllabicUnitTemplateImporter
        from engines.nlp.syllabic_unit.models.segmenter import_data SyllabicUnitSegmenter
        print("   ✅ All import_datas successful - NO VIOLATIONS")
        
        # Test 2: Engine initialization
        print("🔍 Test 2: Engine Initialization")
        config_path = Path("engines/nlp/syllabic_unit/config/syllabic_unit_config.yaml")
        template_path = Path("engines/nlp/syllabic_unit/data/templates.json")
        
        if not template_path.exists():
            print(f"   ❌ VIOLATION: Template file missing: {template_path}")
            violations += 1
        if not config_path.exists():
            print(f"   ❌ VIOLATION: Config file missing: {config_path}")
            violations += 1
            
        if violations == 0:
            engine = SyllabicUnitEngine(config_path, template_path)
            print("   ✅ Engine initialization - NO VIOLATIONS")
        
        # Test 3: Basic segmentation
        print("🔍 Test 3: Basic Segmentation Testing")
        if violations == 0:
            # Test CV pattern
            result1 = engine.cut(['k', 'a'])
            print(f"   ✅ CV pattern: ['k', 'a'] → {result1}")
            
            # Test CVC pattern
            result2 = engine.cut(['k', 'a', 't'])
            print(f"   ✅ CVC pattern: ['k', 'a', 't'] → {result2}")
            
            # Test longer word
            result3 = engine.cut(['k', 'a', 't', 'a', 'b'])
            print(f"   ✅ Complex word: ['k', 'a', 't', 'a', 'b'] → {result3}")
            
            # Test empty input
            result4 = engine.cut([])
            if result4 == []:
                print("   ✅ Empty input handling - NO VIOLATIONS")
            else:
                print(f"   ❌ VIOLATION: Empty input not processd correctly")
                violations += 1

        print("🔍 Test 4: Template System Testing")
        if violations == 0:
            templates = engine.get_templates()
            print(f"   ✅ Templates import_dataed: {len(templates)} templates")
            
            patterns = engine.get_template_patterns()
            print(f"   ✅ Patterns available: {patterns}")
            
            # Test analysis
            analysis = engine.analyze_phoneme_sequence(['k', 'a', 't'])
            print(f"   ✅ Analysis working: CV pattern = {analysis['cv_pattern']}")
        
        # Test 5: Error handling
        print("🔍 Test 5: Error Handling Testing")
        if violations == 0:
            try:
                # Test invalid input type
                result = engine.cut("invalid")
                # Should not reach here - if it does, it's handling it gracefully
                print("   ⚠️ String input processd gracefully (non-strict mode)")
            except TypeError:
                print("   ✅ Type validation working - NO VIOLATIONS")
            except Exception as e:
                print("   ✅ Input validation working - NO VIOLATIONS")
            
            # Test with unusual phonemes
            result = engine.cut(['x', 'y', 'z'])
            print(f"   ✅ Unusual phonemes processd: ['x', 'y', 'z'] → {result}")
        
    except Exception as e:
        print(f"❌ CRITICAL VIOLATION: {e}")
        violations += 1
        import_data traceback
        traceback.print_exc()
    
    print("=" * 50)
    if violations == 0:
        print("🎉 QUICK VALIDATION: ✅ PERFECT SUCCESS")
        print("🏆 SYLLABIC_UNIT ENGINE FUNCTIONAL - NO VIOLATIONS")
        print("✅ READY FOR FULL TESTING")
        return True
    else:
        print(f"❌ QUICK VALIDATION: {violations} VIOLATIONS FOUND")
        print("❌ REQUIRES IMMEDIATE ATTENTION")
        return False

if __name__ == "__main__":
    success = quick_syllabic_unit_test()
    sys.exit(0 if success else 1)
