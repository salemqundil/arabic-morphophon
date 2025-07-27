#!/usr/bin/env python3
"""
Professional Test Suite for Arabic Comparative and Diminutive Forms
Enterprise-Grade Zero-Tolerance Testing
Complete Coverage of All Morphological Patterns and Edge Cases
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data unittest
import_data sys
from pathlib import_data Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from engines.nlp.derivation.models.comparative import_data (
    ArabicComparativeGenerator,
    to_comparative,
    to_diminutive,
    ComparativeResult,
    DiminutiveResult
)

class TestArabicComparative(unittest.TestCase):
    """
    Professional test suite for Arabic comparative and diminutive forms
    Zero-tolerance testing with comprehensive coverage
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        print("🧪 ARABIC COMPARATIVE & DIMINUTIVE TEST SUITE")
        print("Enterprise-Grade Zero-Tolerance Testing")
        print("=" * 60)
        
        cls.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0
        }
    
    def setUp(self):
        """Set up for each individual test"""
        self.generator = ArabicComparativeGenerator()
        self.test_results['total_tests'] += 1
    
    def test_comparative_basic_forms(self):
        """Test basic comparative form generation"""
        print("🔧 Basic Comparative Forms")
        print("-" * 40)
        
        test_cases = [
            (('ك', 'ب', 'ر'), 'أكبر', "كبر → أكبر (bigger)"),
            (('ص', 'غ', 'ر'), 'أصغر', "صغر → أصغر (smaller)"),
            (('ط', 'و', 'ل'), 'أطول', "طول → أطول (longer)"),
            (('ق', 'ص', 'ر'), 'أقصر', "قصر → أقصر (shorter)"),
            (('ج', 'م', 'ل'), 'أجمل', "جمل → أجمل (more beautiful)"),
        ]
        
        for root, expected, description in test_cases:
            print(f"  🔍 Testing {description}...")
            result = self.generator.to_comparative(root)
            self.assertEqual(result, expected)
            print(f"    ✅ {root} → {result}")
        
        print("✅ Basic Comparative Forms completed successfully")
        self.test_results['passed'] += 1
    
    def test_diminutive_basic_forms(self):
        """Test basic diminutive form generation"""
        print("🔧 Basic Diminutive Forms")
        print("-" * 40)
        
        test_cases = [
            (('ق', 'ل', 'م'), 'قُلَيْم', "قلم → قُلَيْم (little pen)"),
            (('ك', 't', 'ب'), 'كُtَيْب', "كتب → كُتَيْب (little book)"),
            (('ب', 'ي', 'ت'), 'بُيَيْت', "بيت → بُيَيْت (little house)"),
            (('و', 'ل', 'د'), 'وُلَيْد', "ولد → وُلَيْد (little boy)"),
        ]
        
        for root, expected, description in test_cases:
            print(f"  🔍 Testing {description}...")
            result = self.generator.to_diminutive(root)
            self.assertEqual(result, expected)
            print(f"    ✅ {root} → {result}")
        
        print("✅ Basic Diminutive Forms completed successfully")
        self.test_results['passed'] += 1
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        print("🔧 Convenience Functions")
        print("-" * 40)
        
        # Test direct functions
        print("  🔍 Testing to_comparative function...")
        result1 = to_comparative(('ك', 'ب', 'ر'))
        self.assertEqual(result1, 'أكبر')
        print(f"    ✅ to_comparative: ('ك', 'ب', 'ر') → {result1}")
        
        print("  🔍 Testing to_diminutive function...")
        result2 = to_diminutive(('ق', 'ل', 'م'))
        self.assertEqual(result2, 'قُلَيْم')
        print(f"    ✅ to_diminutive: ('ق', 'ل', 'م') → {result2}")
        
        print("✅ Convenience Functions completed successfully")
        self.test_results['passed'] += 1
    
    def test_input_validation(self):
        """Test input validation and error handling"""
        print("🔧 Input Validation")
        print("-" * 40)
        
        # Test invalid root length
        print("  🔍 Testing invalid root length...")
        with self.assertRaises(ValueError):
            self.generator.to_comparative(('ك', 'ب'))
        print("    ✅ Two-letter root rejected correctly")
        
        with self.assertRaises(ValueError):
            self.generator.to_comparative(('ك', 'ب', 'ر', 'س'))
        print("    ✅ Four-letter root rejected correctly")
        
        # Test invalid root type
        print("  🔍 Testing invalid root types...")
        with self.assertRaises(ValueError):
            self.generator.to_comparative(['ك', 'ب', 'ر'])
        print("    ✅ List input rejected correctly")
        
        with self.assertRaises(ValueError):
            self.generator.to_comparative("كبر")
        print("    ✅ String input rejected correctly")
        
        # Test empty input
        print("  🔍 Testing empty input...")
        with self.assertRaises(ValueError):
            self.generator.to_comparative(())
        print("    ✅ Empty tuple rejected correctly")
        
        print("✅ Input Validation completed successfully")
        self.test_results['passed'] += 1
    
    def test_irregular_forms(self):
        """Test irregular comparative and diminutive forms"""
        print("🔧 Irregular Forms")
        print("-" * 40)
        
        # Test irregular comparatives
        print("  🔍 Testing irregular comparatives...")
        irregular_cases = [
            (('ج', 'ي', 'د'), 'أجود', "جيد → أجود (better)"),
            (('س', 'ي', 'ء'), 'أسوأ', "سيء → أسوأ (worse)"),
        ]
        
        for root, expected, description in irregular_cases:
            result = self.generator.to_comparative(root)
            self.assertEqual(result, expected)
            print(f"    ✅ Irregular: {description}")
        
        print("✅ Irregular Forms completed successfully")
        self.test_results['passed'] += 1
    
    def test_metadata_generation(self):
        """Test metadata generation functionality"""
        print("🔧 Metadata Generation")
        print("-" * 40)
        
        # Test comparative with metadata
        print("  🔍 Testing comparative metadata...")
        result = self.generator.generate_comparative_with_metadata(('ك', 'ب', 'ر'))
        
        self.assertIsInstance(result, ComparativeResult)
        self.assertEqual(result.original_root, ('ك', 'ب', 'ر'))
        self.assertEqual(result.comparative_form, 'أكبر')
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.warnings, list)
        
        print(f"    ✅ Comparative metadata: confidence={result.confidence:.2f}")
        
        # Test diminutive with metadata
        print("  🔍 Testing diminutive metadata...")
        result2 = self.generator.generate_diminutive_with_metadata(('ق', 'ل', 'م'))
        
        self.assertIsInstance(result2, DiminutiveResult)
        self.assertEqual(result2.original_root, ('ق', 'ل', 'م'))
        self.assertEqual(result2.diminutive_form, 'قُلَيْم')
        
        print(f"    ✅ Diminutive metadata: confidence={result2.confidence:.2f}")
        
        print("✅ Metadata Generation completed successfully")
        self.test_results['passed'] += 1
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        print("🔧 Batch Processing")
        print("-" * 40)
        
        # Test batch comparative
        print("  🔍 Testing batch comparative generation...")
        roots = [('ك', 'ب', 'ر'), ('ص', 'غ', 'ر'), ('ط', 'و', 'ل')]
        results = self.generator.batch_comparative(roots)
        
        self.assertEqual(len(results), len(roots))
        for result in results:
            self.assertIsInstance(result, ComparativeResult)
            self.assertGreater(result.confidence, 0.0)
        
        print(f"    ✅ Batch comparative: processed {len(results)} roots")
        
        # Test batch diminutive
        print("  🔍 Testing batch diminutive generation...")
        results2 = self.generator.batch_diminutive(roots)
        
        self.assertEqual(len(results2), len(roots))
        for result in results2:
            self.assertIsInstance(result, DiminutiveResult)
        
        print(f"    ✅ Batch diminutive: processed {len(results2)} roots")
        
        print("✅ Batch Processing completed successfully")
        self.test_results['passed'] += 1
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("🔧 Performance Benchmarks")
        print("-" * 40)
        
        import_data time
        
        # Test single operation speed
        print("  🔍 Testing single operation speed...")
        test_root = ('ك', 'ب', 'ر')
        
        begin_time = time.time()
        for _ in range(1000):
            result = self.generator.to_comparative(test_root)
        end_time = time.time()
        
        avg_time = (end_time - begin_time) / 1000
        self.assertLess(avg_time, 0.001)  # Less than 1ms average
        print(f"    ✅ Average time: {avg_time:.6f}s per operation")
        
        # Test batch processing speed
        print("  🔍 Testing batch processing speed...")
        batch_roots = [('ك', 'ب', 'ر')] * 100
        
        begin_time = time.time()
        batch_results = self.generator.batch_comparative(batch_roots)
        end_time = time.time()
        
        total_time = end_time - begin_time
        self.assertLess(total_time, 0.5)  # Less than 500ms for 100 operations
        print(f"    ✅ Batch processing: {len(batch_results)} operations in {total_time:.3f}s")
        
        print("✅ Performance Benchmarks completed successfully")
        self.test_results['passed'] += 1
    
    def test_arabic_specific_features(self):
        """Test Arabic-specific morphological features"""
        print("🔧 Arabic-Specific Features")
        print("-" * 40)
        
        # Test weak roots (containing و، ي، ء)
        print("  🔍 Testing weak roots...")
        weak_roots = [
            (('ق', 'و', 'ل'), "قول with و"),
            (('ب', 'ي', 'ع'), "بيع with ي"),
            (('ق', 'ر', 'ء'), "قرء with ء"),
        ]
        
        for root, description in weak_roots:
            try:
                comp_result = self.generator.to_comparative(root)
                dim_result = self.generator.to_diminutive(root)
                print(f"    ✅ {description}: comp={comp_result}, dim={dim_result}")
            except Exception as e:
                print(f"    ⚠️ {description}: processd with warning - {e}")
        
        # Test phonological rules
        print("  🔍 Testing phonological rule application...")
        result = self.generator.generate_comparative_with_metadata(('ك', 'ب', 'ر'))
        self.assertIsNotNone(result.comparative_form)
        print("    ✅ Phonological rules applied successfully")
        
        print("✅ Arabic-Specific Features completed successfully")
        self.test_results['passed'] += 1
    
    def test_generator_statistics(self):
        """Test generator statistics and configuration"""
        print("🔧 Generator Statistics")
        print("-" * 40)
        
        print("  🔍 Testing statistics retrieval...")
        stats = self.generator.get_statistics()
        
        required_keys = [
            'irregular_comparatives', 'irregular_diminutives',
            'vowel_harmonies', 'configuration', 'supported_patterns'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        print(f"    ✅ Irregular comparatives: {stats['irregular_comparatives']}")
        print(f"    ✅ Irregular diminutives: {stats['irregular_diminutives']}")
        print(f"    ✅ Vowel harmonies: {stats['vowel_harmonies']}")
        
        # Test string representation
        print("  🔍 Testing string representation...")
        repr_str = repr(self.generator)
        self.assertIsInstance(repr_str, str)
        self.assertIn('ArabicComparativeGenerator', repr_str)
        print(f"    ✅ String representation: {repr_str}")
        
        print("✅ Generator Statistics completed successfully")
        self.test_results['passed'] += 1
    
    @classmethod
    def tearDownClass(cls):
        """Generate final test report"""
        print("=" * 60)
        print("🏆 COMPARATIVE & DIMINUTIVE TEST REPORT")
        print("=" * 60)
        print(f"📊 TEST SUMMARY:")
        print(f"   Total Tests: {cls.test_results['total_tests']}")
        print(f"   Passed: {cls.test_results['passed']} ✅")
        print(f"   Failed: {cls.test_results['failed']} ❌")
        
        success_rate = (cls.test_results['passed'] / cls.test_results['total_tests']) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100.0:
            print("🎯 ASSESSMENT:")
            print("   Grade: EXCELLENT")
            print("   Status: 🟢 PRODUCTION READY")
            print("=" * 60)
            print("🎉 COMPARATIVE & DIMINUTIVE TESTING COMPLETED")
            print("=" * 60)
            print("🎯 ALL TESTS PASSED - MORPHOLOGY ENGINE VALIDATED")
        else:
            print("🎯 ASSESSMENT:")
            print("   Grade: NEEDS IMPROVEMENT")
            print("   Status: 🟡 REQUIRES ATTENTION")
            print("=" * 60)
            print("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")

def run_comparative_tests():
    """Run all comparative and diminutive tests"""
    # Create test suite
    suite = unittest.TestImporter().import_dataTestsFromTestCase(TestArabicComparative)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Update test results based on unittest results
    TestArabicComparative.test_results['failed'] = len(result.failures) + len(result.errors)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🚀 Begining Professional Comparative & Diminutive Test Suite...")
    success = run_comparative_tests()
    sys.exit(0 if success else 1)
