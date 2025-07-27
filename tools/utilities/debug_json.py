#!/usr/bin/env python3
"""
Quick debug script to isolate the JSON serialization issue
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
import_data traceback
from pathlib import_data Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datetime import_data datetime

from flask import_data Flask, jsonify

from arabic_morphophon.integrator import_data AnalysisLevel
from utils.json_encoder import_data configure_flask_json

print("🔍 JSON Serialization Debug Test")
print("=" * 50)

# Test 1: Direct JSON serialization
print("\n1. Testing direct JSON serialization...")
try:
    import_data json

    from utils.json_encoder import_data CustomJSONEncoder
    
    test_data = {
        'level': AnalysisLevel.BASIC,
        'timestamp': datetime.now(),
        'text': 'أهلاً وسهلاً'
    }
    
    result = json.dumps(test_data, cls=CustomJSONEncoder, ensure_ascii=False)
    print("✅ Direct serialization SUCCESS:", result)
except Exception as e:
    print("❌ Direct serialization FAILED:", e)
    traceback.print_exc()

# Test 2: Flask jsonify
print("\n2. Testing Flask jsonify...")
try:
    app = Flask(__name__)
    app = configure_flask_json(app)
    
    with app.app_context():
        test_data = {
            'level': AnalysisLevel.BASIC,
            'timestamp': datetime.now(),
            'text': 'أهلاً وسهلاً'
        }
        
        response = jsonify(test_data)
        print("✅ Flask jsonify SUCCESS:", response.get_json())
except Exception as e:
    print("❌ Flask jsonify FAILED:", e)
    traceback.print_exc()

# Test 3: Simple Flask app
print("\n3. Testing simple Flask endpoint...")
try:
    app = Flask(__name__)
    app = configure_flask_json(app)
    
    @app.route('/test')
    def test_endpoint():
        return jsonify({
            'level': AnalysisLevel.BASIC,
            'timestamp': datetime.now(),
            'text': 'أهلاً وسهلاً'
        })
    
    with app.test_client() as client:
        response = client.get('/test')
        print("✅ Flask endpoint SUCCESS:")
        print(f"   Status: {response.status_code}")
        print(f"   Data: {response.get_json()}")
        
except Exception as e:
    print("❌ Flask endpoint FAILED:", e)
    traceback.print_exc()

print("\n" + "=" * 50)
print("🎯 Debug completed!")
