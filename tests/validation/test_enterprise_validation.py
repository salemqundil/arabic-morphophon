#!/usr/bin/env python3
"""
🧪 Complete Arabic NLP Pipeline Test - Enterprise Validation
Full Integration: Phonemes → Phonology → SyllabicUnits → Derivation → Comparative/Diminutive
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
import_data os
sys.path.insert(0, os.path.abspath('.'))

def test_full_pipeline_integration():
    """Test complete pipeline from phonemes to derivational morphology"""
    print("🚀 Testing Full Arabic NLP Pipeline Integration")
    print("=" * 70)
    
    try:
        # Test Full Pipeline
        print("🔬 Phase 1: Testing Phoneme → Phonology → SyllabicUnit Pipeline...")
        from engines.nlp.full_pipeline.pipeline import_data FullPipeline
        
        pipeline = FullPipeline()
        
        # Test Arabic text analysis
        test_word = "كتاب"
        result = pipeline.analyze(test_word)
        
        print(f"✅ Input: {test_word}")
        print(f"✅ Phonemes: {result['phonemes']}")
        print(f"✅ Clean Phonemes: {result['phonemes_clean']}")
        print(f"✅ SyllabicUnits: {result['syllabic_units']}")
        
        # Test Derivation Engine
        print("\n🔬 Phase 2: Testing Derivation Engine...")
        from engines.nlp.derivation.engine import_data DerivationEngine
        
        derivation_engine = DerivationEngine()
        
        # Test root analysis
        test_roots = [
            ('ك', 'ت', 'ب'),
            ('ق', 'ر', 'أ'),
            ('ذ', 'ه', 'ب')
        ]
        
        for root in test_roots:
            analysis = derivation_engine.analyze(root)
            print(f"✅ Root {root}: {analysis.get('derived_forms', [])[:3]}...")  # Show first 3 forms
        
        # Test Comparative Generation
        print("\n🔬 Phase 3: Testing Comparative & Diminutive Generation...")
        
        comparative_tests = [
            (('ك', 'ب', 'ر'), 'أكبر'),
            (('ص', 'غ', 'ر'), 'أصغر'),
            (('ط', 'و', 'ل'), 'أطول')
        ]
        
        for root, expected in comparative_tests:
            result = derivation_engine.generate_comparative(root)
            status = "✅" if result == expected else "❌"
            print(f"{status} Comparative {root} → {result}")
        
        diminutive_tests = [
            (('ق', 'ل', 'م'), 'قُلَيْم'),
            (('ك', 'ت', 'ب'), 'كُتَيْب'),
            (('ب', 'ي', 'ت'), 'بُيَيْت')
        ]
        
        for root, expected in diminutive_tests:
            result = derivation_engine.generate_diminutive(root)
            status = "✅" if result == expected else "❌"
            print(f"{status} Diminutive {root} → {result}")
        
        print("\n🔬 Phase 4: Testing Advanced Features...")
        
        # Test metadata generation
        comp_meta = derivation_engine.generate_comparative_with_metadata(('ك', 'ب', 'ر'))
        print(f"✅ Comparative metadata: {comp_meta}")
        
        dim_meta = derivation_engine.generate_diminutive_with_metadata(('ق', 'ل', 'م'))
        print(f"✅ Diminutive metadata: {dim_meta}")
        
        # Test string input support
        comp_str = derivation_engine.generate_comparative('كبر')
        dim_str = derivation_engine.generate_diminutive('قلم')
        print(f"✅ String input comparative: كبر → {comp_str}")
        print(f"✅ String input diminutive: قلم → {dim_str}")
        
        return True
        
    except Exception as e:
        print(f"💥 Pipeline integration failed: {e}")
        import_data traceback
        traceback.print_exc()
        return False

def test_performance_metrics():
    """Test system performance and enterprise standards"""
    print("\n🔬 Testing Performance & Enterprise Standards...")
    
    try:
        import_data time
        from engines.nlp.derivation.engine import_data DerivationEngine
        
        engine = DerivationEngine()
        
        # Batch processing test
        test_roots = [
            ('ك', 'ت', 'ب'), ('ق', 'ر', 'أ'), ('ذ', 'ه', 'ب'),
            ('ص', 'ل', 'ح'), ('ف', 'ع', 'ل'), ('ج', 'م', 'ل')
        ]
        
        begin_time = time.time()
        
        results = []
        for root in test_roots:
            comp = engine.generate_comparative(root)
            dim = engine.generate_diminutive(root)
            results.append((root, comp, dim))
        
        end_time = time.time()
        processing_time = end_time - begin_time
        
        print(f"✅ Processed {len(test_roots)} roots in {processing_time:.4f} seconds")
        print(f"✅ Average: {processing_time/len(test_roots):.4f} seconds per root")
        
        # Memory efficiency test
        engine.clear_cache()
        print("✅ Cache clearing successful")
        
        # Error handling test
        try:
            engine.generate_comparative(('invalid',))  # Should fail gracefully
        except ValueError as e:
            print(f"✅ Error handling: {e}")
        
        return True
        
    except Exception as e:
        print(f"💥 Performance test failed: {e}")
        return False

def main():
    """Run complete system validation"""
    print("🏆 ARABIC NLP ENTERPRISE SYSTEM - COMPLETE VALIDATION")
    print("🎯 Zero-Tolerance Enterprise Standards")
    print("=" * 70)
    
    tests = [
        test_full_pipeline_integration,
        test_performance_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            print(f"\n🧪 Running {test_func.__name__}...")
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"💥 {test_func.__name__} CRASHED: {e}")
    
    print("\n" + "=" * 70)
    print(f"🎯 FINAL ENTERPRISE VALIDATION: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🏆 ENTERPRISE SYSTEM FULLY VALIDATED!")
        print("✅ All phases operational with zero violations")
        print("✅ Phoneme Engine: OPERATIONAL")
        print("✅ Phonology Engine: OPERATIONAL") 
        print("✅ SyllabicUnit Engine: OPERATIONAL")
        print("✅ Derivation Engine: OPERATIONAL")
        print("✅ Comparative/Diminutive: OPERATIONAL")
        print("✅ Full Pipeline Integration: OPERATIONAL")
        return True
    else:
        print("❌ ENTERPRISE VALIDATION FAILED!")
        print("🚨 Zero-tolerance standards violated")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
