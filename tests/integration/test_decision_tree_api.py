#!/usr/bin/env python3
"""
🧪 DECISION TREE API TESTING SCRIPT
Test all decision tree endpoints and demonstrate Q&A flows
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data time
import_data requests
from typing import_data Dict, Any

def test_decision_tree_api():
    """Test all decision tree API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 TESTING DECISION TREE API")
    print("=" * 50)
    
    # Test cases for different decision tree paths
    test_cases = [
        {
            "name": "Arabic Text Analysis",
            "endpoint": "/api/analyze",
            "method": "POST",
            "data": {"text": "كتب الطالب", "level": "comprehensive"},
            "expected_decisions": [
                "Input validation",
                "Text normalization", 
                "Phonological rules",
                "SyllabicAnalysis",
                "Vector operations"
            ]
        },
        {
            "name": "Input Validation Test",
            "endpoint": "/api/validate",
            "method": "POST", 
            "data": {"text": "Hello مرحبا"},
            "expected_decisions": [
                "Data presence check",
                "Type validation",
                "Length validation",
                "Arabic content detection",
                "Security validation"
            ]
        },
        {
            "name": "Statistics Request",
            "endpoint": "/api/stats",
            "method": "GET",
            "data": None,
            "expected_decisions": [
                "Statistics compilation",
                "Performance metrics",
                "Decision tree stats"
            ]
        },
        {
            "name": "Decision Tree Structure",
            "endpoint": "/api/decision-tree",
            "method": "GET",
            "data": None,
            "expected_decisions": [
                "Structure compilation",
                "Q&A mapping",
                "Decision factor analysis"
            ]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🎯 Test: {test_case['name']}")
        print(f"Endpoint: {test_case['method']} {test_case['endpoint']}")
        
        try:
            begin_time = time.time()
            
            if test_case['method'] == 'GET':
                response = requests.get(f"{base_url}{test_case['endpoint']}")
            else:
                response = requests.post(
                    f"{base_url}{test_case['endpoint']}", 
                    json=test_case['data']
                )
            
            processing_time = time.time() - begin_time
            
            print(f"Status: {response.status_code}")
            print(f"Processing Time: {processing_time:.3f}s")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Success!")
                
                # Analyze decision tree execution
                decision_analysis = analyze_decision_execution(data, test_case)
                print(f"Decisions Rund: {decision_analysis['decision_count']}")
                print(f"Decision Path: {decision_analysis['path']}")
                
                results.append({
                    "test": test_case['name'],
                    "status": "success",
                    "processing_time": processing_time,
                    "decision_analysis": decision_analysis
                })
                
            else:
                print(f"❌ Failed: {response.status_code}")
                error_data = response.json() if response.content else {}
                print(f"Error: {error_data.get('error', 'Unknown error')}")
                
                results.append({
                    "test": test_case['name'],
                    "status": "failed",
                    "error": error_data.get('error', 'Unknown error')
                })
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Flask app not running")
            results.append({
                "test": test_case['name'],
                "status": "connection_error"
            })
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                "test": test_case['name'],
                "status": "exception",
                "error": str(e)
            })
    
    # Summary report
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    successful_tests = [r for r in results if r['status'] == 'success']
    failed_tests = [r for r in results if r['status'] != 'success']
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        avg_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
        print(f"Average Processing Time: {avg_time:.3f}s")
        
        total_decisions = sum(r['decision_analysis']['decision_count'] for r in successful_tests)
        print(f"Total Decisions Rund: {total_decisions}")
    
    return results

def analyze_decision_execution(response_data: Dict, test_case: Dict) -> Dict:
    """Analyze decision tree execution from API response"""
    decision_count = 0
    decision_path = []
    
    # Look for decision tree indicators in response
    if 'stages' in response_data:
        decision_count = len(response_data['stages'])
        decision_path = [stage[0] for stage in response_data['stages']]
    
    elif 'decision_tree_context' in response_data:
        questions = response_data['decision_tree_context'].get('questions_asked', [])
        decision_count = len(questions)
        decision_path = response_data['decision_tree_context'].get('decision_path', '').split(' → ')
    
    elif 'decision_tree_stats' in response_data:
        decision_count = response_data['decision_tree_stats'].get('total_decision_points', 0)
        decision_path = ['statistics_compilation']
    
    elif 'engine_operations' in response_data:
        decision_count = len(response_data.get('engine_operations', {}))
        decision_path = ['structure_analysis']
    
    else:
        # Look for other decision indicators
        if 'success' in response_data:
            decision_count = 1
            decision_path = ['basic_processing']
    
    return {
        "decision_count": decision_count,
        "path": " → ".join(decision_path),
        "expected_decisions": test_case.get('expected_decisions', []),
        "coverage": min(decision_count / len(test_case.get('expected_decisions', [1])), 1.0) if test_case.get('expected_decisions') else 1.0
    }

def test_individual_endpoints():
    """Test individual decision tree components"""
    print("\n🔬 INDIVIDUAL COMPONENT TESTS")
    print("=" * 40)
    
    # Test validation endpoint with different inputs
    validation_tests = [
        {"text": "كتاب", "expected": "valid"},
        {"text": "", "expected": "invalid_empty"},
        {"text": "x" * 1001, "expected": "invalid_long"},
        {"text": "<script>", "expected": "invalid_security"}
    ]
    
    for test in validation_tests:
        print(f"\nValidation Test: {test['expected']}")
        try:
            response = requests.post(
                "http://localhost:5000/api/validate",
                json={"text": test["text"]}
            )
            data = response.json()
            print(f"Valid: {data.get('valid', False)}")
            print(f"Errors: {len(data.get('errors', []))}")
            print(f"Warnings: {len(data.get('warnings', []))}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    try:
        results = test_decision_tree_api()
        test_individual_endpoints()
        
        print("\n🎯 Decision Tree API Testing Complete!")
        print("All major decision flows have been tested.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
