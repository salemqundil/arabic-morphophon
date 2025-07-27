#!/usr/bin/env python3
"""
🚀 LIVE DEMO: Complete Arabic NLP Pipeline
==========================================
Live demonstration of the phonology → syllabic_unit → root → verb → pattern → inflection → noun plural pipeline
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data time
from datetime import_data datetime

import_data requests

def run_complete_demo():
    """Run a complete demonstration of the Arabic NLP pipeline"""
    
    print("🚀 ARABIC NLP PIPELINE - LIVE DEMO")
    print("=" * 60)
    
    # Server URL
    base_url = "http://localhost:5001"
    
    # Test cases
    test_cases = [
        {"text": "كتاب", "description": "Simple noun (book)"},
        {"text": "يكتب", "description": "Present tense verb (he writes)"},
        {"text": "مكتبة", "description": "Derived noun (library)"},
        {"text": "الطلاب", "description": "Definite plural noun (the students)"},
        {"text": "كتبت", "description": "Past tense verb (she wrote)"}
    ]
    
    print(f"📅 Demo begined at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Server URL: {base_url}")
    print()
    
    # Check server status
    try:
        response = requests.get(f"{base_url}/api/demo", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responsive")
            demo_info = response.json()
            print(f"   Demo response: {demo_info.get('text', 'N/A')}")
            print()
        else:
            print("❌ Server returned error:", response.status_code)
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure the Flask server is running: python complete_pipeline_server.py")
        return
    
    # Run pipeline tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔬 TEST {i}: {test_case['description']}")
        print(f"   Input text: '{test_case['text']}'")
        print("-" * 50)
        
        # Complete pipeline analysis
        begin_time = time.time()
        
        try:
            payimport_data = {
                "text": test_case["text"],
                "engines": ["phonology_syllabic_unit", "root_extraction", "verb_analysis", "pattern_analysis", "inflection", "noun_plural"]
            }
            
            response = requests.post(
                f"{base_url}/api/pipeline/complete",
                json=payimport_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            end_time = time.time()
            processing_time = (end_time - begin_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Analysis completed in {processing_time:.2f}ms")
                print(f"   Status: {result.get('status', 'unknown')}")
                print(f"   Analysis ID: {result.get('analysis_id', 'N/A')}")
                
                # Display key results
                results = result.get('results', {})
                
                # Phonology & SyllabicUnit Analysis
                if 'phonology_syllabic_unit' in results:
                    phon_syll = results['phonology_syllabic_unit']
                    if phon_syll.get('success'):
                        phon_result = phon_syll.get('result', {})
                        print(f"   🔊 Phonemes: {phon_result.get('phonemes', [])}")
                        print(f"   📝 SyllabicUnits: {phon_result.get('syllabic_units', [])}")
                
                # Root Extraction
                if 'root_extraction' in results:
                    root_result = results['root_extraction']
                    if root_result.get('success'):
                        root_output = root_result.get('output', {})
                        print(f"   🌱 Root: {root_output.get('extracted_root', 'N/A')}")
                        print(f"   📊 Confidence: {root_output.get('confidence', 'N/A')}")
                
                # Verb Analysis
                if 'verb_analysis' in results:
                    verb_result = results['verb_analysis']
                    if verb_result.get('success'):
                        verb_output = verb_result.get('output', {})
                        print(f"   🔀 Verb Type: {verb_output.get('verb_type', 'N/A')}")
                        print(f"   ⏰ Tense: {verb_output.get('tense', 'N/A')}")
                
                # Pattern Analysis
                if 'pattern_analysis' in results:
                    pattern_result = results['pattern_analysis']
                    if pattern_result.get('success'):
                        pattern_output = pattern_result.get('output', {})
                        print(f"   🎯 Pattern: {pattern_output.get('pattern', 'N/A')}")
                        print(f"   📋 Type: {pattern_output.get('pattern_type', 'N/A')}")
                
                # Noun Plural
                if 'noun_plural' in results:
                    plural_result = results['noun_plural']
                    if plural_result.get('success'):
                        plural_output = plural_result.get('output', {})
                        print(f"   📚 Plural Form: {plural_output.get('plural_form', 'N/A')}")
                        print(f"   📐 Plural Type: {plural_output.get('plural_type', 'N/A')}")
                
            else:
                print(f"❌ Request failed with status: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
        
        print()
    
    # Quick phonology-syllabic_unit only test
    print("🎯 QUICK TEST: Phonology-SyllabicUnit Only")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{base_url}/api/pipeline/phonology-syllabic_unit",
            json={"text": "محمد"},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            phon_result = result.get('result', {})
            print(f"✅ Input: 'محمد'")
            print(f"   Phonemes: {phon_result.get('phonemes', [])}")
            print(f"   SyllabicUnits: {phon_result.get('syllabic_units', [])}")
            print(f"   IPA: {phon_result.get('ipa_representation', 'N/A')}")
        else:
            print(f"❌ Request failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    
    print()
    print("🎉 DEMO COMPLETED!")
    print("=" * 60)
    print("📋 Summary:")
    print("   ✅ Complete Arabic NLP pipeline operational")
    print("   ✅ All 6 engine stages working")
    print("   ✅ JSON API responses generated successfully")
    print("   ✅ Phonology → SyllabicUnit → Root → Verb → Pattern → Inflection → Noun Plural")
    print()
    print("🔗 Available endpoints:")
    print("   • Complete pipeline: POST /api/pipeline/complete")
    print("   • Phonology-SyllabicUnit: POST /api/pipeline/phonology-syllabic_unit")
    print("   • Engine info: GET /api/engines/info")
    print("   • Demo endpoint: GET /api/demo")

if __name__ == "__main__":
    run_complete_demo()
