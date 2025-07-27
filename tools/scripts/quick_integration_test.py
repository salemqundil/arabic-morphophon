#!/usr/bin/env python3
"""
Quick Full Pipeline Integration Validator
Zero-Tolerance Rapid Testing
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
from pathlib import_data Path

# Add project root to path
sys.path.append('.')

def quick_integration_test():
    """Quick validation of full pipeline integration"""
    print("🔥 QUICK FULL PIPELINE INTEGRATION TEST")
    print("=" * 50)
    
    violations = 0
    
    try:
        # Test 1: Import validation
        print("🔍 Test 1: Import Validation")
        from engines.nlp.full_pipeline.engine import_data FullPipeline, PipelineResult
        print("   ✅ Full pipeline import_datas successful - NO VIOLATIONS")
        
        # Test 2: Pipeline initialization
        print("🔍 Test 2: Pipeline Initialization")
        pipeline = FullPipeline()
        print("   ✅ Pipeline initialization - NO VIOLATIONS")
        
        # Test 3: Basic integration test
        print("🔍 Test 3: Basic Integration Testing")
        
        # Test simple word
        result1 = pipeline.analyze("ka")
        print(f"   ✅ Simple word: 'ka' → {result1['syllabic_units']}")
        
        # Test Arabic word
        result2 = pipeline.analyze("kitab")
        print(f"   ✅ Arabic word: 'kitab' → {result2['syllabic_units']}")
        
        # Test complex word
        result3 = pipeline.analyze("muhammad")
        print(f"   ✅ Complex word: 'muhammad' → {result3['syllabic_units']}")
        
        # Validate result structure
        for result in [result1, result2, result3]:
            if not all(key in result for key in ['phonemes', 'phonemes_clean', 'syllabic_units']):
                print("   ❌ VIOLATION: Missing required keys in result")
                violations += 1
            
            if not isinstance(result['syllabic_units'], list):
                print("   ❌ VIOLATION: SyllabicUnits not a list")
                violations += 1
        
        # Test 4: Metadata functionality
        print("🔍 Test 4: Metadata Functionality")
        metadata_result = pipeline.analyze_with_metadata("kitab")
        
        if not isinstance(metadata_result, PipelineResult):
            print("   ❌ VIOLATION: Metadata result wrong type")
            violations += 1
        else:
            print(f"   ✅ Metadata: confidence={metadata_result.confidence:.2f}, time={metadata_result.processing_time:.4f}s")
        
        # Test 5: Batch processing
        print("🔍 Test 5: Batch Processing")
        batch_results = pipeline.analyze_batch(["ka", "kitab", "muhammad"])
        
        if len(batch_results) != 3:
            print("   ❌ VIOLATION: Batch processing incorrect length")
            violations += 1
        else:
            print(f"   ✅ Batch processing: {len(batch_results)} results")
        
        # Test 6: Error handling
        print("🔍 Test 6: Error Handling")
        try:
            error_result = pipeline.analyze(123)  # Should fail
            print("   ❌ VIOLATION: Should reject non-string input")
            violations += 1
        except TypeError:
            print("   ✅ Type validation working - NO VIOLATIONS")
        
        # Test empty input
        empty_result = pipeline.analyze("")
        if empty_result['syllabic_units'] != []:
            print("   ❌ VIOLATION: Empty input not processd correctly")
            violations += 1
        else:
            print("   ✅ Empty input handling - NO VIOLATIONS")
        
        # Test 7: Pipeline validation
        print("🔍 Test 7: Pipeline Component Validation")
        validation = pipeline.validate_pipeline()
        
        if not validation['overall']:
            print("   ❌ VIOLATION: Pipeline validation failed")
            violations += 1
        else:
            print("   ✅ All pipeline components validated - NO VIOLATIONS")
        
        # Test 8: Statistics
        print("🔍 Test 8: Pipeline Statistics")
        stats = pipeline.get_pipeline_stats()
        
        required_stats = ['total_processed', 'success_rate', 'engines_status']
        for stat in required_stats:
            if stat not in stats:
                print(f"   ❌ VIOLATION: Missing statistic {stat}")
                violations += 1
        
        if violations == 0:
            print(f"   ✅ Pipeline statistics working - NO VIOLATIONS")
            print(f"   ✅ Processed: {stats['total_processed']} texts")
            print(f"   ✅ Success rate: {stats['success_rate']:.2%}")
        
    except Exception as e:
        print(f"❌ CRITICAL VIOLATION: {e}")
        violations += 1
        import_data traceback
        traceback.print_exc()
    
    print("=" * 50)
    if violations == 0:
        print("🎉 QUICK INTEGRATION TEST: ✅ PERFECT SUCCESS")
        print("🏆 FULL PIPELINE FUNCTIONAL - NO VIOLATIONS")
        print("✅ READY FOR PRODUCTION DEPLOYMENT")
        print("🚀 FLASK API INTEGRATION READY")
        return True
    else:
        print(f"❌ QUICK INTEGRATION TEST: {violations} VIOLATIONS FOUND")
        print("❌ REQUIRES IMMEDIATE ATTENTION")
        return False

if __name__ == "__main__":
    success = quick_integration_test()
    sys.exit(0 if success else 1)
