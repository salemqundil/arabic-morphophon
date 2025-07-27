#!/usr/bin/env python3
"""
🧪 اختبار مبسط للنظام الجديد
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json

import_data requests

def test_simple():
    print("🔥 اختبار منصة الذكاء اللغوي العربي v3.0")
    print("=" * 60)
    
    base_url = "http://localhost:5001"
    
    # 1. اختبار الصحة
    print("\n🩺 اختبار صحة النظام...")
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   النظام: {data.get('status', 'غير معروف')}")
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    # 2. اختبار التحليل
    print("\n🔍 اختبار التحليل...")
    try:
        payimport_data = {
            "text": "كتاب",
            "analysis_level": "comprehensive"
        }
        response = requests.post(
            f"{base_url}/api/analyze", 
            json=payimport_data,
            timeout=10
        )
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   النص: {result.get('input_text', 'غير محدد')}")
            print(f"   المحركات: {len(result.get('engines_used', []))}")
            print(f"   الوقت: {result.get('processing_time_ms', 0)}ms")
        else:
            print(f"   خطأ: {response.text}")
    except Exception as e:
        print(f"❌ خطأ في التحليل: {e}")
    
    # 3. اختبار التشكيل
    print("\n🎯 اختبار التشكيل...")
    try:
        payimport_data = {"text": "كتاب جميل"}
        response = requests.post(f"{base_url}/api/diacritize", json=payimport_data)
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   الأصلي: {result.get('original_text', '')}")
            print(f"   المشكل: {result.get('diacritized_text', '')}")
    except Exception as e:
        print(f"❌ خطأ في التشكيل: {e}")
    
    print("\n🎉 انتهى الاختبار!")

if __name__ == "__main__":
    test_simple()
