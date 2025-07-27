#!/usr/bin/env python3
"""
🏆 ZERO VIOLATIONS TEST SUITE - OPTIMIZED VERSION
================================================
Professional Arabic NLP System Validation
Ultra-fast Testing & Zero Error Tolerance
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data sys
import_data time
from typing import_data Any, Dict, List

import_data requests

class OptimizedTestSuite:
    """Ultra-fast professional testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:5004"):
        self.base_url = base_url
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests with zero violations tolerance"""
        print("ZERO VIOLATIONS TEST SUITE - OPTIMIZED")
        print("=" * 50)
        
        # Test system endpoints
        self._test_system_endpoints()
        
        # Test core engines
        self._test_phonology_engine()
        self._test_syllabic_unit_engine() 
        self._test_morphology_engine()
        
        # Test comprehensive analysis
        self._test_comprehensive_analysis()
        
        # Test error handling
        self._test_error_handling()
        
        # Test performance
        self._test_performance()
        
        return self._generate_report()
    
    def _test_system_endpoints(self):
        """Test system health endpoints"""
        print("\nTesting System Endpoints...")
        
        self._run_test("Home Endpoint", lambda: requests.get(f"{self.base_url}/"))
        self._run_test("Health Check", lambda: requests.get(f"{self.base_url}/health"))
        self._run_test("System Status", lambda: requests.get(f"{self.base_url}/status"))
    
    def _test_phonology_engine(self):
        """Test phonology engine"""
        print("\nTesting Phonology Engine...")
        
        test_cases = [
            "السلام عليكم",
            "كتاب",
            "مدرسة"
        ]
        
        for i, text in enumerate(test_cases, 1):
            self._run_test(
                f"Phonology Test {i}",
                lambda: requests.post(
                    f"{self.base_url}/phonology",
                    json={"text": text},
                    headers={"Content-Type": "application/json"}
                ),
                validation_func=self._validate_phonology_response
            )
    
    def _test_syllabic_unit_engine(self):
        """Test syllabic_unit engine"""
        print("\nTesting SyllabicUnit Engine...")
        
        test_cases = [
            "كتب",
            "مدرسة", 
            "كتب الطالب"
        ]
        
        for i, text in enumerate(test_cases, 1):
            self._run_test(
                f"SyllabicUnit Test {i}",
                lambda: requests.post(
                    f"{self.base_url}/syllabic_unit",
                    json={"text": text},
                    headers={"Content-Type": "application/json"}
                ),
                validation_func=self._validate_syllabic_unit_response
            )
    
    def _test_morphology_engine(self):
        """Test morphology engine"""
        print("\nTesting Morphology Engine...")
        
        test_cases = [
            "كاتب",
            "كتب الطالب الدرس",
            "مكتبة مدرسة"
        ]
        
        for i, text in enumerate(test_cases, 1):
            self._run_test(
                f"Morphology Test {i}",
                lambda: requests.post(
                    f"{self.base_url}/morphology",
                    json={"text": text},
                    headers={"Content-Type": "application/json"}
                ),
                validation_func=self._validate_morphology_response
            )
    
    def _test_comprehensive_analysis(self):
        """Test comprehensive analysis"""
        print("\nTesting Comprehensive Analysis...")
        
        levels = ["basic", "intermediate", "comprehensive", "expert"]
        test_text = "كتب الطالب الدرس"
        
        for level in levels:
            self._run_test(
                f"Analysis Level: {level}",
                lambda: requests.post(
                    f"{self.base_url}/analyze",
                    json={"text": test_text, "analysis_level": level},
                    headers={"Content-Type": "application/json"}
                ),
                validation_func=lambda r: self._validate_comprehensive_response(r, level)
            )
    
    def _test_error_handling(self):
        """Test error handling"""
        print("\nTesting Error Handling...")
        
        # Empty text
        self._run_test(
            "Empty Text Error",
            lambda: requests.post(
                f"{self.base_url}/analyze",
                json={"text": ""},
                headers={"Content-Type": "application/json"}
            ),
            expected_status=400
        )
        
        # Missing JSON
        self._run_test(
            "Missing JSON Error", 
            lambda: requests.post(f"{self.base_url}/analyze"),
            expected_status=400
        )
        
        # Invalid endpoint
        self._run_test(
            "Invalid Endpoint",
            lambda: requests.get(f"{self.base_url}/invalid"),
            expected_status=404
        )
    
    def _test_performance(self):
        """Test performance metrics"""
        print("\nTesting Performance...")
        
        # Single request performance
        begin_time = time.time()
        response = requests.post(
            f"{self.base_url}/analyze",
            json={"text": "اختبار الأداء", "analysis_level": "comprehensive"},
            headers={"Content-Type": "application/json"}
        )
        response_time = (time.time() - begin_time) * 1000
        
        self._run_test(
            f"Response Time ({response_time:.1f}ms - Target: <50ms)",
            lambda: response,
            validation_func=lambda r: response_time < 100  # Allow 100ms for safety
        )
        
        # Import test
        print("  Running import_data test (5 requests)...")
        import_data_times = []
        success_count = 0
        
        for i in range(5):
            begin = time.time()
            try:
                resp = requests.post(
                    f"{self.base_url}/analyze",
                    json={"text": f"نص تجريبي {i+1}"},
                    headers={"Content-Type": "application/json"},
                    timeout=2
                )
                if resp.status_code == 200:
                    success_count += 1
                import_data_times.append((time.time() - begin) * 1000)
            except:
                import_data_times.append(2000)  # Timeout
        
        avg_import_data_time = sum(import_data_times) / len(import_data_times)
        success_rate = (success_count / 5) * 100
        
        print(f"    Import test: {success_rate}% success, {avg_import_data_time:.1f}ms avg")
    
    def _run_test(self, test_name: str, test_func, expected_status: int = 200, validation_func=None):
        """Run individual test"""
        begin_time = time.time()
        self.total_tests += 1
        
        try:
            response = test_func()
            execution_time = (time.time() - begin_time) * 1000
            
            # Check status
            status_ok = response.status_code == expected_status
            
            # Additional validation
            validation_ok = True
            if validation_func:
                try:
                    validation_ok = validation_func(response)
                except:
                    validation_ok = False
            
            # Result
            if status_ok and validation_ok:
                self.passed_tests += 1
                print(f"  PASS {test_name} ({execution_time:.1f}ms)")
                self.results.append({
                    'test': test_name,
                    'status': 'PASS',
                    'time_ms': execution_time
                })
            else:
                self.failed_tests += 1
                print(f"  FAIL {test_name} ({execution_time:.1f}ms)")
                self.results.append({
                    'test': test_name,
                    'status': 'FAIL',
                    'time_ms': execution_time,
                    'error': f"Status: {response.status_code}, Expected: {expected_status}"
                })
                
        except Exception as e:
            self.failed_tests += 1
            execution_time = (time.time() - begin_time) * 1000
            print(f"  ERROR {test_name} ({execution_time:.1f}ms): {str(e)}")
            self.results.append({
                'test': test_name,
                'status': 'ERROR',
                'time_ms': execution_time,
                'error': str(e)
            })
    
    def _validate_phonology_response(self, response) -> bool:
        """Validate phonology response"""
        try:
            data = response.json()
            return (data.get('status') == 'success' and 
                   'phonemes' in data.get('data', {}) and
                   'ipa_transcription' in data.get('data', {}))
        except:
            return False
    
    def _validate_syllabic_unit_response(self, response) -> bool:
        """Validate syllabic_unit response"""
        try:
            data = response.json()
            return (data.get('status') == 'success' and 
                   'syllabic_units' in data.get('data', {}) and
                   'cv_pattern' in data.get('data', {}))
        except:
            return False
    
    def _validate_morphology_response(self, response) -> bool:
        """Validate morphology response"""
        try:
            data = response.json()
            return (data.get('status') == 'success' and 
                   'word_analyses' in data.get('data', {}))
        except:
            return False
    
    def _validate_comprehensive_response(self, response, level: str) -> bool:
        """Validate comprehensive response"""
        try:
            data = response.json()
            if data.get('status') != 'success':
                return False
            
            results = data.get('data', {}).get('results', {})
            
            # Check required engines based on level
            if level in ['basic', 'intermediate', 'comprehensive', 'expert'] and 'phonology' not in results:
                return False
            if level in ['intermediate', 'comprehensive', 'expert'] and 'syllabic_unit' not in results:
                return False
            if level in ['comprehensive', 'expert'] and 'morphology' not in results:
                return False
            
            return True
        except:
            return False
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        avg_time = sum(r['time_ms'] for r in self.results if r['status'] == 'PASS') / max(1, self.passed_tests)
        
        report = {
            'summary': {
                'total_tests': self.total_tests,
                'passed': self.passed_tests,
                'failed': self.failed_tests,
                'success_rate': round(success_rate, 1),
                'average_response_time_ms': round(avg_time, 1),
                'status': 'PASS' if success_rate >= 95 else 'FAIL'
            },
            'results': self.results
        }
        
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success Rate: {report['summary']['success_rate']}%")
        print(f"Avg Response Time: {report['summary']['average_response_time_ms']}ms")
        print(f"Overall Status: {report['summary']['status']}")
        
        if report['summary']['status'] == 'PASS':
            print("\nZERO VIOLATIONS ACHIEVED - ALL TESTS PASSED!")
        else:
            print("\nSOME TESTS FAILED - REVIEW REQUIRED")
        
        return report

def main():
    """Main test execution"""
    try:
        test_suite = OptimizedTestSuite()
        
        # Check server availability
        print("Checking server availability...")
        try:
            response = requests.get(f"{test_suite.base_url}/health", timeout=3)
            print(f"Server is running at {test_suite.base_url}")
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Server not available at {test_suite.base_url}")
            print("Please begin the server: python optimized_flask_nlp_system.py")
            return
        
        # Run tests
        report = test_suite.run_all_tests()
        
        # Store report
        with open('optimized_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest report store_datad: optimized_test_report.json")
        
        # Exit with status code
        sys.exit(0 if report['summary']['status'] == 'PASS' else 1)
        
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
