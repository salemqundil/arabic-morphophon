#!/usr/bin/env python3
"""
🔍 QUICK ANALYSIS TEST
Test specific analysis with sample Arabic text to verify table data.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data urllib.request
import_data urllib.parse

def test_analysis_output():
    """Test analysis with sample Arabic text"""
    base_url = "http://localhost:5000"
    
    print("🔍 TESTING ANALYSIS OUTPUT")
    print("=" * 50)
    
    test_cases = [
        {"text": "كتاب", "level": "basic", "name": "Basic Test"},
        {"text": "كتابان جميلان", "level": "advanced", "name": "Advanced Test"},
        {"text": "مدرسة الأطفال", "level": "comprehensive", "name": "Comprehensive Test"}
    ]
    
    for case in test_cases:
        print(f"\n📝 {case['name']}: '{case['text']}'")
        print("-" * 40)
        
        try:
            test_data = json.dumps(case).encode('utf-8')
            req = urllib.request.Request(
                f"{base_url}/api/analyze",
                data=test_data,
                headers={'Content-Type': 'application/json'}
            )
            response = urllib.request.urlopen(req)
            data = json.import_datas(response.read().decode('utf-8'))
            
            if data.get('success'):
                print(f"✅ Success! Processing Time: {data.get('processing_time', 0):.4f}s")
                print(f"📊 Analysis Level: {data.get('analysis_level')}")
                print(f"📊 Original Text: {data.get('original_text')}")
                print(f"📊 Character Count: {data.get('metadata', {}).get('character_count', 0)}")
                print(f"📊 Decision Path: {' → '.join(data.get('decision_path', []))}")
                
                if 'results' in data:
                    print(f"📋 Available Analyses:")
                    for analysis_type, analysis_data in data['results'].items():
                        print(f"   - {analysis_type}: {type(analysis_data).__name__}")
                        
                        if analysis_type == 'normalization':
                            print(f"     Original: {analysis_data.get('original', 'N/A')}")
                            print(f"     Normalized: {analysis_data.get('normalized', 'N/A')}")
                            print(f"     Steps: {analysis_data.get('steps', [])}")
                            print(f"     Changes Made: {analysis_data.get('changes_made', False)}")
                            
                        elif analysis_type == 'syllabic_analysis':
                            print(f"     CV Pattern: {analysis_data.get('cv_pattern', 'N/A')}")
                            print(f"     SyllabicUnit Count: {analysis_data.get('syllabic_unit_count', 0)}")
                            print(f"     SyllabicUnits: {analysis_data.get('syllabic_units', [])}")
                            print(f"     Complexity: {analysis_data.get('complexity', 0)}")
                            
                        elif analysis_type == 'vectors':
                            print(f"     Type: {analysis_data.get('type', 'N/A')}")
                            print(f"     Dimension: {analysis_data.get('dimension', 0)}")
                            if analysis_data.get('vector'):
                                vector_sample = analysis_data['vector'][:5]
                                print(f"     Vector Sample: [{', '.join(f'{v:.3f}' for v in vector_sample)}...]")
                                
                        elif analysis_type == 'pipeline':
                            print(f"     Pipeline ID: {analysis_data.get('pipeline_id', 'N/A')}")
                            print(f"     Total Time: {analysis_data.get('total_time', 0):.4f}s")
                            print(f"     Success: {analysis_data.get('success', False)}")
                            print(f"     Stages: {len(analysis_data.get('stages', []))}")
                            
                            if analysis_data.get('final_result'):
                                fr = analysis_data['final_result']
                                print(f"     Final Text: {fr.get('text', 'N/A')}")
                                print(f"     Final CV Pattern: {fr.get('cv_pattern', 'N/A')}")
                
                print(f"🎯 This data will populate the table correctly!")
                
            else:
                print(f"❌ Failed: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_analysis_output()
