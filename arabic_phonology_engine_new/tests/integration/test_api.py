#!/usr/bin/env python3
"""
API End-to-End Test
"""

import requests
import json
import time


def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"

    print("🌐 Testing API End-to-End...")

    # Wait for server to be ready
    time.sleep(2)

    try:
        # Test health endpoint
        print("  Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("  ✅ Health endpoint working")
            print(f"     Response: {response.json()}")
        else:
            print(f"  ❌ Health endpoint failed: {response.status_code}")
            return False

        # Test root endpoint
        print("  Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("  ✅ Root endpoint working")
            print(f"     Response: {response.json()}")
        else:
            print(f"  ❌ Root endpoint failed: {response.status_code}")
            return False

        # Test analyze endpoint
        print("  Testing analyze endpoint...")
        test_data = {"root": ["K", "T", "B"]}
        response = requests.post(
            f"{base_url}/analyze",
            json=test_data,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            print("  ✅ Analyze endpoint working")
            result = response.json()
            if result.get("success"):
                print(f"     Analysis result: {result['result']['root']}")
            else:
                print(f"     Analysis failed: {result}")
                return False
        else:
            print(f"  ❌ Analyze endpoint failed: {response.status_code}")
            print(f"     Response: {response.text}")
            return False

        return True

    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_api()
    if success:
        print("
🎉 All API tests passed!")
    else:
        print("
❌ Some API tests failed!")
