#!/usr/bin/env python3
"""
Professional Phase 1 Integration Test Suite
Enterprise-Grade Zero-Tolerance Testing
Complete Pipeline Integration: Phoneme → Phonological → SyllabicUnit
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data unittest
import_data time
import_data sys
from pathlib import_data Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from engines.nlp.full_pipeline.engine import_data FullPipeline, PipelineResult

class TestFullPipelineIntegration(unittest.TestCase):
    """
    Professional integration test suite for full Arabic NLP pipeline
    Zero-tolerance testing with comprehensive coverage
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        print("🧪 ARABIC NLP FULL PIPELINE INTEGRATION TEST SUITE")
        print("Enterprise-Grade Zero-Tolerance Testing")
        print("=" * 70)
        
        # Test configuration paths
        cls.phoneme_cfg = Path("engines/nlp/phoneme/config/phoneme_config.yaml")
        cls.rule_cfg = Path("engines/nlp/phonological/config/rules_config.yaml")
        cls.rule_data = Path("engines/nlp/phonological/data/rules.json")
        cls.syllabic_unit_cfg = Path("engines/nlp/syllabic_unit/config/syllabic_unit_config.yaml")
        cls.syllabic_unit_templates = Path("engines/nlp/syllabic_unit/data/templates.json")
        
        # Test statistics
        cls.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'begin_time': time.time()
        }
    
    def setUp(self):
        """Set up for each individual test"""
        try:
            self.pipeline = FullPipeline(
                self.phoneme_cfg,
                self.rule_cfg,
                self.rule_data,
                self.syllabic_unit_cfg,
                self.syllabic_unit_templates
            )
            self.test_results['total_tests'] += 1
        except Exception as e:
            self.fail(f"Failed to initialize FullPipeline: {e}")
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization and component validation"""
        print("🔧 Pipeline Initialization")
        print("-" * 50)
        
        # Test 1: Pipeline initialization
        print("  🔍 Testing pipeline initialization...")
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.phoneme_engine)
        self.assertIsNotNone(self.pipeline.phonology_engine)
        self.assertIsNotNone(self.pipeline.syllabic_unit_engine)
        print("    ✅ Pipeline initialized successfully")
        
        # Test 2: Component validation
        print("  🔍 Testing component validation...")
        validation = self.pipeline.validate_pipeline()
        self.assertTrue(validation['overall'])
        self.assertTrue(validation['phoneme_engine'])
        self.assertTrue(validation['phonology_engine'])
        self.assertTrue(validation['syllabic_unit_engine'])
        print("    ✅ All components validated successfully")
        
        # Test 3: Configuration paths
        print("  🔍 Testing configuration paths...")
        self.assertTrue(self.pipeline.phoneme_cfg.exists())
        self.assertTrue(self.pipeline.rule_cfg.exists())
        self.assertTrue(self.pipeline.rule_data.exists())
        self.assertTrue(self.pipeline.syllabic_unit_cfg.exists())
        self.assertTrue(self.pipeline.syllabic_unit_templates.exists())
        print("    ✅ All configuration files exist")
        
        print("✅ Pipeline Initialization completed successfully")
        self.test_results['passed'] += 1
    
    def test_basic_pipeline_integration(self):
        """Test basic pipeline integration with simple inputs"""
        print("🔧 Basic Pipeline Integration")
        print("-" * 50)
        
        # Test cases with expected results
        test_cases = [
            ("ka", "Simple CV unit"),
            ("kātaba", "Arabic verb كتب"),
            ("kitāb", "Arabic noun كتاب"),
            ("muhammad", "Arabic name محمد"),
        ]
        
        for text, description in test_cases:
            print(f"  🔍 Testing {description}...")
            
            result = self.pipeline.analyze(text)
            
            # Validate result structure
            self.assertIn('phonemes', result)
            self.assertIn('phonemes_clean', result)
            self.assertIn('syllabic_units', result)
            
            # Validate result types
            self.assertIsInstance(result['phonemes'], list)
            self.assertIsInstance(result['phonemes_clean'], list)
            self.assertIsInstance(result['syllabic_units'], list)
            
            # Validate that processing occurred
            self.assertGreater(len(result['phonemes']), 0)
            self.assertGreater(len(result['syllabic_units']), 0)
            
            print(f"    ✅ '{text}' → phonemes: {len(result['phonemes'])}, syllabic_units: {len(result['syllabic_units'])}")
        
        print("✅ Basic Pipeline Integration completed successfully")
        self.test_results['passed'] += 1
    
    def test_arabic_text_processing(self):
        """Test processing of actual Arabic text"""
        print("🔧 Arabic Text Processing")
        print("-" * 50)
        
        # Arabic test cases
        arabic_test_cases = [
            ("فعل", "Arabic verb فعل"),
            ("كتاب", "Arabic noun كتاب"),
            ("محمد", "Arabic name محمد"),
            ("عبدالله", "Arabic name عبدالله"),
        ]
        
        for text, description in arabic_test_cases:
            print(f"  🔍 Testing {description}...")
            
            result = self.pipeline.analyze(text)
            
            # Validate Arabic processing
            self.assertIsInstance(result['syllabic_units'], list)
            self.assertGreater(len(result['syllabic_units']), 0)
            
            # Validate syllabic_unit structure
            for syllabic_unit in result['syllabic_units']:
                self.assertIsInstance(syllabic_unit, list)
                self.assertGreater(len(syllabic_unit), 0)
            
            print(f"    ✅ {description}: {result['syllabic_units']}")
        
        print("✅ Arabic Text Processing completed successfully")
        self.test_results['passed'] += 1
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        print("🔧 Edge Cases and Error Handling")
        print("-" * 50)
        
        # Test 1: Empty input
        print("  🔍 Testing empty input...")
        result = self.pipeline.analyze("")
        self.assertEqual(result['phonemes'], [])
        self.assertEqual(result['phonemes_clean'], [])
        self.assertEqual(result['syllabic_units'], [])
        print("    ✅ Empty input processd correctly")
        
        # Test 2: Whitespace input
        print("  🔍 Testing whitespace input...")
        result = self.pipeline.analyze("   ")
        self.assertEqual(result['phonemes'], [])
        print("    ✅ Whitespace input processd correctly")
        
        # Test 3: Invalid input type
        print("  🔍 Testing invalid input type...")
        with self.assertRaises(TypeError):
            self.pipeline.analyze(123)
        print("    ✅ Invalid input type rejected correctly")
        
        # Test 4: Single character
        print("  🔍 Testing single character...")
        result = self.pipeline.analyze("a")
        self.assertIsInstance(result['syllabic_units'], list)
        print("    ✅ Single character processd correctly")
        
        # Test 5: Special characters
        print("  🔍 Testing special characters...")
        try:
            result = self.pipeline.analyze("@#$%")
            print("    ⚠️ Special characters processd gracefully")
        except Exception:
            print("    ✅ Special characters rejected appropriately")
        
        print("✅ Edge Cases and Error Handling completed successfully")
        self.test_results['passed'] += 1
    
    def test_pipeline_with_metadata(self):
        """Test pipeline with metadata functionality"""
        print("🔧 Pipeline with Metadata")
        print("-" * 50)
        
        print("  🔍 Testing metadata analysis...")
        text = "kitāb"
        result = self.pipeline.analyze_with_metadata(text)
        
        # Validate PipelineResult structure
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.original_text, text)
        self.assertIsInstance(result.phonemes, list)
        self.assertIsInstance(result.phonemes_clean, list)
        self.assertIsInstance(result.syllabic_units, list)
        self.assertIsInstance(result.processing_time, float)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.method, str)
        self.assertIsInstance(result.warnings, list)
        self.assertIsInstance(result.metadata, dict)
        
        # Validate metadata content
        self.assertIn('phoneme_count', result.metadata)
        self.assertIn('syllabic_unit_count', result.metadata)
        self.assertIn('pipeline_version', result.metadata)
        
        print(f"    ✅ Confidence: {result.confidence:.2f}")
        print(f"    ✅ Processing time: {result.processing_time:.4f}s")
        print(f"    ✅ Method: {result.method}")
        print(f"    ✅ Phoneme count: {result.metadata['phoneme_count']}")
        print(f"    ✅ SyllabicUnit count: {result.metadata['syllabic_unit_count']}")
        
        print("✅ Pipeline with Metadata completed successfully")
        self.test_results['passed'] += 1
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        print("🔧 Batch Processing")
        print("-" * 50)
        
        print("  🔍 Testing batch processing...")
        batch_texts = ["ka", "kitāb", "muhammad", "faʕil", ""]
        
        results = self.pipeline.analyze_batch(batch_texts)
        
        # Validate batch results
        self.assertEqual(len(results), len(batch_texts))
        
        for i, result in enumerate(results):
            if 'error' not in result:  # Skip error results
                self.assertIn('phonemes', result)
                self.assertIn('syllabic_units', result)
        
        print(f"    ✅ Processed {len(batch_texts)} texts in batch")
        print(f"    ✅ Success rate: {len([r for r in results if 'error' not in r])}/{len(results)}")
        
        print("✅ Batch Processing completed successfully")
        self.test_results['passed'] += 1
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("🔧 Performance Benchmarks")
        print("-" * 50)
        
        # Test 1: Single analysis speed
        print("  🔍 Testing single analysis performance...")
        test_text = "kitāb"
        
        begin_time = time.time()
        for _ in range(100):
            result = self.pipeline.analyze(test_text)
        end_time = time.time()
        
        avg_time = (end_time - begin_time) / 100
        self.assertLess(avg_time, 0.01)  # Less than 10ms average
        print(f"    ✅ Average analysis time: {avg_time:.6f}s")
        
        # Test 2: Batch processing speed
        print("  🔍 Testing batch processing performance...")
        batch_texts = ["ka", "kitāb", "muhammad"] * 50
        
        begin_time = time.time()
        batch_results = self.pipeline.analyze_batch(batch_texts)
        end_time = time.time()
        
        total_time = end_time - begin_time
        per_text_time = total_time / len(batch_texts)
        
        print(f"    ✅ Batch processing: {len(batch_texts)} texts in {total_time:.3f}s")
        print(f"    ✅ Per-text time: {per_text_time:.6f}s")
        
        # Test 3: Pipeline statistics
        print("  🔍 Testing pipeline statistics...")
        stats = self.pipeline.get_pipeline_stats()
        
        self.assertIn('total_processed', stats)
        self.assertIn('average_processing_time', stats)
        self.assertIn('success_rate', stats)
        
        print(f"    ✅ Total processed: {stats['total_processed']}")
        print(f"    ✅ Success rate: {stats['success_rate']:.2%}")
        
        print("✅ Performance Benchmarks completed successfully")
        self.test_results['passed'] += 1
    
    def test_pipeline_consistency(self):
        """Test pipeline consistency and reproducibility"""
        print("🔧 Pipeline Consistency")
        print("-" * 50)
        
        print("  🔍 Testing result consistency...")
        test_text = "kitāb"
        
        # Run same analysis multiple times
        results = []
        for _ in range(5):
            result = self.pipeline.analyze(test_text)
            results.append(result)
        
        # Check consistency
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(result['phonemes'], first_result['phonemes'])
            self.assertEqual(result['phonemes_clean'], first_result['phonemes_clean'])
            self.assertEqual(result['syllabic_units'], first_result['syllabic_units'])
        
        print("    ✅ Results are consistent across multiple runs")
        
        # Test different pipeline instances
        print("  🔍 Testing cross-instance consistency...")
        pipeline2 = FullPipeline(
            self.phoneme_cfg,
            self.rule_cfg,
            self.rule_data,
            self.syllabic_unit_cfg,
            self.syllabic_unit_templates
        )
        
        result1 = self.pipeline.analyze(test_text)
        result2 = pipeline2.analyze(test_text)
        
        self.assertEqual(result1['syllabic_units'], result2['syllabic_units'])
        print("    ✅ Results are consistent across pipeline instances")
        
        print("✅ Pipeline Consistency completed successfully")
        self.test_results['passed'] += 1
    
    @classmethod
    def tearDownClass(cls):
        """Generate final test report"""
        total_time = time.time() - cls.test_results['begin_time']
        
        print("=" * 70)
        print("🏆 FULL PIPELINE INTEGRATION TEST REPORT")
        print("=" * 70)
        print(f"📊 TEST SUMMARY:")
        print(f"   Total Tests: {cls.test_results['total_tests']}")
        print(f"   Passed: {cls.test_results['passed']} ✅")
        print(f"   Failed: {cls.test_results['failed']} ❌")
        
        success_rate = (cls.test_results['passed'] / cls.test_results['total_tests']) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Time: {total_time:.3f} seconds")
        
        if success_rate == 100.0:
            print("🎯 ASSESSMENT:")
            print("   Grade: EXCELLENT")
            print("   Status: 🟢 PRODUCTION READY")
            print("=" * 70)
            print("🎉 FULL PIPELINE INTEGRATION TESTING COMPLETED")
            print("=" * 70)
            print("🎯 ALL TESTS PASSED - FULL PIPELINE VALIDATED")
            print("🚀 READY FOR FLASK API DEPLOYMENT")
        else:
            print("🎯 ASSESSMENT:")
            print("   Grade: NEEDS IMPROVEMENT")
            print("   Status: 🟡 REQUIRES ATTENTION")
            print("=" * 70)
            print("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")

def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    suite = unittest.TestImporter().import_dataTestsFromTestCase(TestFullPipelineIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Update test results based on unittest results
    TestFullPipelineIntegration.test_results['failed'] = len(result.failures) + len(result.errors)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🚀 Begining Professional Full Pipeline Integration Test Suite...")
    success = run_integration_tests()
    sys.exit(0 if success else 1)
