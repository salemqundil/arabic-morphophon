#!/usr/bin/env python3
"""
🧪 Frontend-Backend Connection Test
تطبيق بسيط لاختبار الاتصال بين الواجهة والخادم
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data time

import_data requests

def test_backend_connection():
    """Test all backend endpoints"""
    
    print("🔥 اختبار الاتصال بين الواجهة والخادم")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    # 1. Test Health Endpoint
    print("\n🩺 اختبار صحة النظام...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   النظام: {health_data.get('status', 'غير معروف')}")
            print(f"   الطابع الزمني: {health_data.get('timestamp', 'غير محدد')}")
        else:
            print(f"❌ خطأ: {response.text}")
    except Exception as e:
        print(f"❌ فشل الاتصال: {e}")
        return False
    
    # 2. Test Analysis Endpoint
    print("\n🔍 اختبار تحليل النص...")
    try:
        payimport_data = {
            "text": "كتاب جميل",
            "analysis_level": "comprehensive"
        }
        
        response = requests.post(
            f"{base_url}/api/analyze",
            json=payimport_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   النص المدخل: {result.get('input_text', 'غير محدد')}")
            print(f"   عدد المحركات: {len(result.get('engines_used', []))}")
            print(f"   وقت المعالجة: {result.get('processing_time_ms', 0)}ms")
            
            # Display some results
            if 'results' in result:
                results = result['results']
                if 'phonology' in results:
                    phonology = results['phonology']
                    print(f"   🔊 الأصوات: {len(phonology.get('phonemes', []))}")
                
                if 'morphology' in results:
                    morphology = results['morphology']
                    print(f"   🌱 الجذور: {morphology.get('roots', [])}")
        else:
            print(f"❌ خطأ في التحليل: {response.text}")
            
    except Exception as e:
        print(f"❌ فشل تحليل النص: {e}")
    
    # 3. Test Diacritization
    print("\n🎯 اختبار التشكيل...")
    try:
        payimport_data = {"text": "كتاب جميل"}
        
        response = requests.post(
            f"{base_url}/api/diacritize",
            json=payimport_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   النص الأصلي: {result.get('original_text', '')}")
            print(f"   النص المشكل: {result.get('diacritized_text', '')}")
            print(f"   درجة الثقة: {result.get('confidence', 0)}")
        else:
            print(f"❌ خطأ في التشكيل: {response.text}")
            
    except Exception as e:
        print(f"❌ فشل التشكيل: {e}")
    
    # 4. Test Weight Extraction
    print("\n⚖️ اختبار استخراج الأوزان...")
    try:
        payimport_data = {"text": "كاتب مكتبة"}
        
        response = requests.post(
            f"{base_url}/api/weight",
            json=payimport_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            weights = result.get('weights', [])
            print(f"   عدد الكلمات: {len(weights)}")
            for weight in weights:
                word = weight.get('word', '')
                analysis = weight.get('weight_analysis', {})
                pattern = analysis.get('pattern', 'غير محدد')
                print(f"   📋 {word}: {pattern}")
        else:
            print(f"❌ خطأ في الأوزان: {response.text}")
            
    except Exception as e:
        print(f"❌ فشل استخراج الأوزان: {e}")
    
    # 5. Test Stats
    print("\n📊 اختبار الإحصائيات...")
    try:
        response = requests.get(f"{base_url}/api/stats", timeout=5)
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            stats = response.json()
            print(f"   إجمالي الطلبات: {stats.get('total_requests', 0)}")
            print(f"   معدل النجاح: {stats.get('success_rate', 0)}%")
            print(f"   متوسط الاستجابة: {stats.get('average_response_time', 0):.3f}s")
        else:
            print(f"❌ خطأ في الإحصائيات: {response.text}")
            
    except Exception as e:
        print(f"❌ فشل جلب الإحصائيات: {e}")
    
    # 6. Test Feedback
    print("\n📝 اختبار التغذية الراجعة...")
    try:
        payimport_data = {
            "word": "كتاب",
            "correct_weight": "فِعال",
            "user_id": "test_user"
        }
        
        response = requests.post(
            f"{base_url}/api/feedback",
            json=payimport_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   الحالة: {result.get('status', 'غير محدد')}")
            print(f"   إجمالي التغذية: {result.get('total_feedback', 0)}")
        else:
            print(f"❌ خطأ في التغذية: {response.text}")
            
    except Exception as e:
        print(f"❌ فشل التغذية الراجعة: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 انتهى اختبار الاتصال!")
    print(f"🌐 الخادم متاح على: {base_url}")
    print(f"📚 التوثيق: {base_url}/docs")
    print(f"🖥️ الواجهة: file:///c:/Users/Administrator/new%20engine/frontend.html")
    
    return True

if __name__ == "__main__":
    test_backend_connection()
