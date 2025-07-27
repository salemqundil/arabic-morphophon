#!/usr/bin/env python3
"""
🧪 QUICK API TEST - AttributeError Fix Verification
Test that the Flask API is working after fixing the PHONEME_INVENTORY issue
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data requests
import_data json

def test_api_endpoints():
    """Test the main API endpoints to verify the fix worked"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Flask Decision Tree API after AttributeError fix...")
    print("=" * 60)
    
    # Test 1: Main interface (GET /)
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ GET / - Main interface: SUCCESS")
        else:
            print(f"❌ GET / - Main interface: FAILED ({response.status_code})")
    except Exception as e:
        print(f"❌ GET / - Main interface: ERROR - {e}")
    
    # Test 2: Stats endpoint (GET /api/stats)
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GET /api/stats - Statistics: SUCCESS")
            print(f"   📊 Phoneme Inventory: {data.get('phoneme_inventory', 'N/A')}")
        else:
            print(f"❌ GET /api/stats - Statistics: FAILED ({response.status_code})")
    except Exception as e:
        print(f"❌ GET /api/stats - Statistics: ERROR - {e}")
    
    # Test 3: Validation endpoint (POST /api/validate)
    try:
        test_data = {"text": "مرحبا"}
        response = requests.post(
            f"{base_url}/api/validate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_data)
        )
        if response.status_code == 200:
            print("✅ POST /api/validate - Input validation: SUCCESS")
        else:
            print(f"❌ POST /api/validate - Input validation: FAILED ({response.status_code})")
    except Exception as e:
        print(f"❌ POST /api/validate - Input validation: ERROR - {e}")
    
    # Test 4: Decision tree structure (GET /api/decision-tree)
    try:
        response = requests.get(f"{base_url}/api/decision-tree")
        if response.status_code == 200:
            data = response.json()
            print("✅ GET /api/decision-tree - Decision tree structure: SUCCESS")
            print(f"   🌳 Total Categories: {len(data.get('categories', []))}")
        else:
            print(f"❌ GET /api/decision-tree - Decision tree structure: FAILED ({response.status_code})")
    except Exception as e:
        print(f"❌ GET /api/decision-tree - Decision tree structure: ERROR - {e}")
    
    print("=" * 60)
    print("🎯 AttributeError fix verification complete!")

if __name__ == "__main__":
    test_api_endpoints()
