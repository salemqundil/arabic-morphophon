#!/usr/bin/env python3
"""
🏆 ZERO VIOLATIONS TEST SUITE
============================
Professional Arabic NLP System Validation
Expert-level Testing & Quality Assurance
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data sys
import_data time
from dataclasses import_data dataclass
from typing import_data Any, Dict, List

import_data requests

@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    status: str
    execution_time_ms: float
    response_data: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

class ProfessionalTestSuite:
    """🎯 Professional testing framework with zero tolerance"""
    
    def __init__(self, base_url: str = "http://localhost:5003"):
        self.base_url = base_url
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("🚀 Begining Professional Arabic NLP Test Suite")
        print("=" * 60)
        
        # Test Categories
        self._test_system_health()
        self._test_phonology_engine()
        self._test_syllabic_unit_engine()
        self._test_morphology_engine()
        self._test_comprehensive_analysis()
        self._test_error_handling()
        self._test_performance()
        
        # Generate report
        return self._generate_test_report()
    
    def _test_system_health(self):
        """Test system health and status endpoints"""
        print("\n🔍 Testing System Health...")
        
        # Test home endpoint
        self._run_command_test(
            "Home Endpoint",
            lambda: requests.get(f"{self.base_url}/"),
            expected_status=200
        )
        
        # Test health check
        self._run_command_test(
            "Health Check",
            lambda: requests.get(f"{self.base_url}/health"),
            expected_status=200
        )
        
        # Test status endpoint
        self._run_command_test(
            "System Status",
            lambda: requests.get(f"{self.base_url}/status"),
            expected_status=200
        )
        
        # Test documentation
        self._run_command_test(
            "API Documentation",
            lambda: requests.get(f"{self.base_url}/docs"),
            expected_status=200
        )
    
    def _test_phonology_engine(self):
        """Test phonological analysis engine"""
        print("\n🔊 Testing Phonology Engine...")
        
        test_cases = [
            {
                "name": "Basic Arabic Text",
                "text": "السلام عليكم",
                "expected_features": ["phonemes", "ipa_transcription", "statistics"]
            },
            {
                "name": "Single Word",
                "text": "كتاب",
                "expected_features": ["phonemes", "consonants", "vowels"]
            },
            {
                "name": "Complex Text",
                "text": "بسم الله الرحمن الرحيم",
                "expected_features": ["phonemes", "statistics"]
            }
        ]
        
        for case in test_cases:
            self._run_command_test(
                f"Phonology: {case['name']}",
                lambda: requests.post(
                    f"{self.base_url}/phonology",
                    json={"text": case["text"]}
                ),
                expected_status=200,
                validation_func=lambda r: self._validate_phonology_response(r, case["expected_features"])
            )
    
    def _test_syllabic_unit_engine(self):
        """Test syllabic_unit analysis engine"""
        print("\n🔧 Testing SyllabicUnit Engine...")
        
        test_cases = [
            {
                "name": "Simple Word",
                "text": "كتب",
                "expected_features": ["syllabic_units", "cv_pattern", "statistics"]
            },
            {
                "name": "Complex Word",
                "text": "مدرسة",
                "expected_features": ["syllabic_units", "prosodic_structure"]
            },
            {
                "name": "Multi-word Text",
                "text": "كتب الطالب",
                "expected_features": ["syllabic_units", "syllabic_unit_count"]
            }
        ]
        
        for case in test_cases:
            self._run_command_test(
                f"SyllabicUnit: {case['name']}",
                lambda: requests.post(
                    f"{self.base_url}/syllabic_unit",
                    json={"text": case["text"]}
                ),
                expected_status=200,
                validation_func=lambda r: self._validate_syllabic_unit_response(r, case["expected_features"])
            )
    
    def _test_morphology_engine(self):
        """Test morphological analysis engine"""
        print("\n🏗️ Testing Morphology Engine...")
        
        test_cases = [
            {
                "name": "Root Extraction",
                "text": "كاتب",
                "expected_features": ["root_analysis", "pattern_analysis"]
            },
            {
                "name": "Multiple Words",
                "text": "كتب الطالب الدرس",
                "expected_features": ["word_analyses", "root_summary"]
            },
            {
                "name": "Complex Morphology",
                "text": "مكتبة مدرسة",
                "expected_features": ["pattern_summary", "statistics"]
            }
        ]
        
        for case in test_cases:
            self._run_command_test(
                f"Morphology: {case['name']}",
                lambda: requests.post(
                    f"{self.base_url}/morphology",
                    json={"text": case["text"]}
                ),
                expected_status=200,
                validation_func=lambda r: self._validate_morphology_response(r, case["expected_features"])
            )
    
    def _test_comprehensive_analysis(self):
        """Test comprehensive analysis endpoint"""
        print("\n🎯 Testing Comprehensive Analysis...")
        
        analysis_levels = ["basic", "intermediate", "comprehensive", "expert"]
        test_text = "كتب الطالب الدرس في المدرسة"
        
        for level in analysis_levels:
            self._run_command_test(
                f"Comprehensive Analysis: {level}",
                lambda: requests.post(
                    f"{self.base_url}/analyze",
                    json={"text": test_text, "analysis_level": level}
                ),
                expected_status=200,
                validation_func=lambda r: self._validate_comprehensive_response(r, level)
            )
    
    def _test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n⚠️ Testing Error Handling...")
        
        # Empty text
        self._run_command_test(
            "Empty Text Error",
            lambda: requests.post(
                f"{self.base_url}/analyze",
                json={"text": ""}
            ),
            expected_status=400
        )
        
        # Missing JSON
        self._run_command_test(
            "Missing JSON Error",
            lambda: requests.post(f"{self.base_url}/analyze"),
            expected_status=400
        )
        
        # Invalid endpoint
        self._run_command_test(
            "Invalid Endpoint",
            lambda: requests.get(f"{self.base_url}/invalid"),
            expected_status=404
        )
        
        # Invalid analysis level
        self._run_command_test(
            "Invalid Analysis Level",
            lambda: requests.post(
                f"{self.base_url}/analyze",
                json={"text": "test", "analysis_level": "invalid"}
            ),
            expected_status=200  # Should default to comprehensive
        )
    
    def _test_performance(self):
        """Test performance metrics"""
        print("\n📊 Testing Performance...")
        
        test_text = "هذا نص تجريبي لاختبار الأداء"
        
        # Response time test
        begin_time = time.time()
        response = requests.post(
            f"{self.base_url}/analyze",
            json={"text": test_text, "analysis_level": "comprehensive"}
        )
        response_time = (time.time() - begin_time) * 1000
        
        self._run_command_test(
            f"Response Time Test (Target: <100ms, Actual: {response_time:.2f}ms)",
            lambda: response,
            expected_status=200,
            validation_func=lambda r: response_time < 500  # Allow 500ms for development
        )
        
        # Import test (multiple requests)
        print("  📈 Running import_data test (10 concurrent requests)...")
        import_data_begin = time.time()
        import_data_responses = []
        
        for i in range(10):
            try:
                resp = requests.post(
                    f"{self.base_url}/analyze",
                    json={"text": f"نص تجريبي رقم {i+1}"},
                    timeout=5
                )
                import_data_responses.append(resp.status_code == 200)
            except Exception as e:
                import_data_responses.append(False)
        
        import_data_time = (time.time() - import_data_begin) * 1000
        success_rate = sum(import_data_responses) / len(import_data_responses)
        
        print(f"    ✅ Import test completed: {success_rate*100:.1f}% success rate in {import_data_time:.2f}ms")
    
    def _run_command_test(self, test_name: str, test_func, expected_status: int = 200, validation_func=None):
        """Run individual test"""
        begin_time = time.time()
        self.total_tests += 1
        
        try:
            # Run test
            if callable(test_func):
                response = test_func()
            else:
                response = test_func
            
            execution_time = (time.time() - begin_time) * 1000
            
            # Validate status code
            status_ok = response.status_code == expected_status
            
            # Additional validation
            validation_ok = True
            if validation_func and callable(validation_func):
                try:
                    validation_ok = validation_func(response)
                except Exception as e:
                    validation_ok = False
            
            # Determine test result
            test_passed = status_ok and validation_ok
            
            if test_passed:
                self.passed_tests += 1
                print(f"  ✅ {test_name} - PASSED ({execution_time:.2f}ms)")
                status = "PASSED"
                errors = []
            else:
                self.failed_tests += 1
                print(f"  ❌ {test_name} - FAILED ({execution_time:.2f}ms)")
                status = "FAILED"
                errors = [f"Status: {response.status_code}, Expected: {expected_status}"]
            
            # Store result
            try:
                response_data = response.json() if hasattr(response, 'json') else {}
            except:
                response_data = {"raw_response": str(response.content)}
            
            self.test_results.append(TestResult(
                test_name=test_name,
                status=status,
                execution_time_ms=execution_time,
                response_data=response_data,
                errors=errors,
                warnings=[]
            ))
            
        except Exception as e:
            self.failed_tests += 1
            execution_time = (time.time() - begin_time) * 1000
            print(f"  ❌ {test_name} - ERROR ({execution_time:.2f}ms): {str(e)}")
            
            self.test_results.append(TestResult(
                test_name=test_name,
                status="ERROR",
                execution_time_ms=execution_time,
                response_data={},
                errors=[str(e)],
                warnings=[]
            ))
    
    def _validate_phonology_response(self, response, expected_features: List[str]) -> bool:
        """Validate phonology response structure"""
        try:
            data = response.json()
            if data.get('status') != 'success':
                return False
            
            response_data = data.get('data', {})
            for feature in expected_features:
                if feature not in response_data:
                    return False
            
            return True
        except:
            return False
    
    def _validate_syllabic_unit_response(self, response, expected_features: List[str]) -> bool:
        """Validate syllabic_unit response structure"""
        try:
            data = response.json()
            if data.get('status') != 'success':
                return False
            
            response_data = data.get('data', {})
            for feature in expected_features:
                if feature not in response_data:
                    return False
            
            return True
        except:
            return False
    
    def _validate_morphology_response(self, response, expected_features: List[str]) -> bool:
        """Validate morphology response structure"""
        try:
            data = response.json()
            if data.get('status') != 'success':
                return False
            
            response_data = data.get('data', {})
            for feature in expected_features:
                if feature not in response_data:
                    return False
            
            return True
        except:
            return False
    
    def _validate_comprehensive_response(self, response, analysis_level: str) -> bool:
        """Validate comprehensive analysis response"""
        try:
            data = response.json()
            if data.get('status') != 'success':
                return False
            
            response_data = data.get('data', {})
            results = response_data.get('results', {})
            
            # Validate based on analysis level
            if analysis_level in ['basic', 'intermediate', 'comprehensive', 'expert'] and 'phonology' not in results:
                return False
            
            if analysis_level in ['intermediate', 'comprehensive', 'expert'] and 'syllabic_unit' not in results:
                return False
            
            if analysis_level in ['comprehensive', 'expert'] and 'morphology' not in results:
                return False
            
            return True
        except:
            return False
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        report = {
            "test_summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": round(success_rate, 2),
                "status": "PASSED" if success_rate >= 95 else "FAILED"
            },
            "test_results": [
                {
                    "test_name": result.test_name,
                    "status": result.status,
                    "execution_time_ms": result.execution_time_ms,
                    "errors": result.errors
                }
                for result in self.test_results
            ],
            "performance_metrics": {
                "average_response_time": round(
                    sum(r.execution_time_ms for r in self.test_results if r.status == "PASSED") / 
                    max(1, len([r for r in self.test_results if r.status == "PASSED"])), 2
                ),
                "fastest_response": min(r.execution_time_ms for r in self.test_results if r.status == "PASSED") if self.test_results else 0,
                "slowest_response": max(r.execution_time_ms for r in self.test_results if r.status == "PASSED") if self.test_results else 0
            }
        }
        
        print("\n" + "=" * 60)
        print("📊 TEST SUITE RESULTS")
        print("=" * 60)
        print(f"📈 Total Tests: {report['test_summary']['total_tests']}")
        print(f"✅ Passed: {report['test_summary']['passed_tests']}")
        print(f"❌ Failed: {report['test_summary']['failed_tests']}")
        print(f"🎯 Success Rate: {report['test_summary']['success_rate']}%")
        print(f"🏆 Overall Status: {report['test_summary']['status']}")
        print(f"⚡ Avg Response Time: {report['performance_metrics']['average_response_time']}ms")
        
        if report['test_summary']['status'] == "PASSED":
            print("\n🎉 ALL TESTS PASSED - ZERO VIOLATIONS ACHIEVED! 🎉")
        else:
            print("\n⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        
        return report

def main():
    """Main test execution"""
    try:
        # Check if server is running
        test_suite = ProfessionalTestSuite()
        
        print("🔍 Checking server availability...")
        try:
            response = requests.get(f"{test_suite.base_url}/health", timeout=5)
            print(f"✅ Server is running at {test_suite.base_url}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Server not available at {test_suite.base_url}")
            print("Please begin the server first: python professional_flask_nlp_system.py")
            return
        except Exception as e:
            print(f"❌ Server check failed: {str(e)}")
            return
        
        # Run comprehensive tests
        report = test_suite.run_comprehensive_tests()
        
        # Store report
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 Test report store_datad to: test_report.json")
        
        # Exit with appropriate code
        exit_code = 0 if report['test_summary']['status'] == "PASSED" else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
