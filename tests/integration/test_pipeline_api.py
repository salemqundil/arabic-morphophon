#!/usr/bin/env python3
"""
Test the Complete Arabic NLP Pipeline API
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json

import_data requests

def test_pipeline():
    print('🎯 TESTING COMPLETE ARABIC NLP PIPELINE')
    print('=' * 60)

    # Test 1: Engine Info
    try:
        response = requests.get('http://localhost:5001/api/engines/info')
        if response.status_code == 200:
            data = response.json()
            print(f'✅ Engines Available: {data["total_engines"]}')
            print(f'📋 Pipeline Order: {" → ".join(data["pipeline_order"])}')
        else:
            print(f'❌ Engine info failed: {response.status_code}')
    except Exception as e:
        print(f'❌ Connection failed: {e}')

    # Test 2: Complete Pipeline
    print(f'\n🔍 COMPLETE PIPELINE TEST:')
    test_word = 'كتاب'

    try:
        data = {'text': test_word}
        response = requests.post('http://localhost:5001/api/pipeline/complete', json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Pipeline Success: {test_word}')
            meta = result['pipeline_metadata']
            print(f'⏱️ Total Time: {meta["total_processing_time_ms"]}ms')
            print(f'🔧 Engines Used: {len(meta["engines_used"])}')
            
            # Show results from each engine
            for engine_name, engine_result in result['results'].items():
                if engine_result.get('success', False):
                    print(f'   ✅ {engine_name}: Success')
                else:
                    print(f'   ❌ {engine_name}: Failed')
                    
        else:
            print(f'❌ Pipeline failed: {response.status_code}')
            print(response.text)
            
    except Exception as e:
        print(f'❌ Pipeline test failed: {e}')

    # Test 3: Demo endpoint
    print(f'\n🎭 DEMO TEST:')
    try:
        response = requests.get('http://localhost:5001/api/demo')
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Demo Success: {result["demo_text"]}')
            print(f'📊 Demo engines: {len(result["demo_result"]["results"])}')
        else:
            print(f'❌ Demo failed: {response.status_code}')
    except Exception as e:
        print(f'❌ Demo test failed: {e}')

    print(f'\n✅ PIPELINE TESTING COMPLETE!')

if __name__ == "__main__":
    test_pipeline()
