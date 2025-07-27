#!/usr/bin/env python3
"""
🧪 اختبار سريع للجيل الثالث من منصة الذكاء اللغوي العربي
================================================================
Quick test for Arabic NLP Expert Engine v3.0
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data time
from datetime import_data datetime

import_data requests

def test_arabic_nlp_v3():
    """🔬 اختبار شامل للنظام الجديد"""
    
    print("🔥 بدء اختبار منصة الذكاء اللغوي العربي v3.0")
    print("=" * 70)
    
    base_url = "http://localhost:5001"
    
    # 1. اختبار الصفحة الرئيسية
    print("\n🏠 اختبار الصفحة الرئيسية...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ النجاح! الإصدار: {data.get('version', 'غير محدد')}")
            print(f"   المحركات المتوفرة: {len(data.get('features', []))}")
        else:
            print(f"❌ فشل: {response.status_code}")
    except Exception as e:
        print(f"❌ خطأ في الاتصال: {e}")
        return False
    
    # 2. اختبار صحة النظام
    print("\n🩺 اختبار صحة النظام...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ حالة النظام: {health.get('status', 'غير معروف')}")
            print(f"   المحركات النشطة: {health.get('available_engines', 0)}")
        else:
            print(f"❌ فشل فحص الصحة: {response.status_code}")
    except Exception as e:
        print(f"❌ خطأ في فحص الصحة: {e}")
    
    # 3. اختبار التحليل الأساسي
    print("\n🔍 اختبار التحليل الأساسي...")
    test_cases = [
        {"text": "كتاب", "description": "كلمة بسيطة"},
        {"text": "يكتب الطالب", "description": "جملة بسيطة"},
        {"text": "المكتبة العربية", "description": "عبارة مركبة"},
        {"text": "الذكاء الاصطناعي", "description": "مصطلح تقني"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   🧪 الاختبار {i}: {test_case['description']}")
        print(f"      النص: '{test_case['text']}'")
        
        try:
            begin_time = time.time()
            
            payimport_data = {
                "text": test_case["text"],
                "analysis_level": "comprehensive"
            }
            
            response = requests.post(
                f"{base_url}/api/analyze",
                json=payimport_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            end_time = time.time()
            processing_time = (end_time - begin_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ تم التحليل في {processing_time:.2f}ms")
                print(f"      📊 المحركات المستخدمة: {len(result.get('engines_used', []))}")
                
                # عرض نتائج مختصرة
                if 'results' in result:
                    results = result['results']
                    if 'phonology' in results:
                        phonemes = results['phonology'].get('phonemes', [])
                        print(f"      🔊 الأصوات: {len(phonemes)} صوت")
                    
                    if 'morphology' in results:
                        roots = results['morphology'].get('roots', [])
                        print(f"      🌱 الجذور: {roots}")
                    
                    if 'transformer_layer' in results:
                        confidence = results['transformer_layer'].get('confidence_scores', {})
                        print(f"      🧠 ثقة النموذج: {confidence.get('overall', 'غير محدد')}")
            else:
                print(f"      ❌ فشل: {response.status_code}")
                print(f"      📄 الاستجابة: {response.text[:200]}...")
                
        except Exception as e:
            print(f"      ❌ خطأ: {e}")
    
    # 4. اختبار التشكيل
    print("\n🎯 اختبار التشكيل...")
    try:
        payimport_data = {"text": "كتاب جميل"}
        response = requests.post(
            f"{base_url}/api/diacritize",
            json=payimport_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            original = result.get('original_text', '')
            diacritized = result.get('diacritized_text', '')
            confidence = result.get('confidence', 0)
            print(f"   ✅ النص الأصلي: {original}")
            print(f"   ✅ النص المشكل: {diacritized}")
            print(f"   📊 درجة الثقة: {confidence}")
        else:
            print(f"   ❌ فشل التشكيل: {response.status_code}")
    except Exception as e:
        print(f"   ❌ خطأ في التشكيل: {e}")
    
    # 5. اختبار استخراج الأوزان
    print("\n⚖️ اختبار استخراج الأوزان...")
    try:
        payimport_data = {"text": "كاتب مكتبة"}
        response = requests.post(
            f"{base_url}/api/weight",
            json=payimport_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            weights = result.get('weights', [])
            print(f"   ✅ عدد الكلمات المحللة: {len(weights)}")
            for weight in weights:
                word = weight.get('word', '')
                analysis = weight.get('weight_analysis', {})
                pattern = analysis.get('pattern', 'غير محدد')
                print(f"   📋 {word}: {pattern}")
        else:
            print(f"   ❌ فشل استخراج الأوزان: {response.status_code}")
    except Exception as e:
        print(f"   ❌ خطأ في استخراج الأوزان: {e}")
    
    # 6. اختبار التغذية الراجعة
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
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ تم حفظ التغذية الراجعة")
            print(f"   📊 إجمالي التغذية الراجعة: {result.get('total_feedback', 0)}")
            print(f"   💬 الرسالة: {result.get('message', '')}")
        else:
            print(f"   ❌ فشل حفظ التغذية الراجعة: {response.status_code}")
    except Exception as e:
        print(f"   ❌ خطأ في التغذية الراجعة: {e}")
    
    # 7. اختبار الإحصائيات
    print("\n📊 اختبار الإحصائيات...")
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✅ إجمالي الطلبات: {stats.get('total_requests', 0)}")
            print(f"   ⏱️ متوسط وقت الاستجابة: {stats.get('average_response_time', 0):.3f}s")
            print(f"   📈 معدل النجاح: {stats.get('success_rate', 0)}%")
        else:
            print(f"   ❌ فشل جلب الإحصائيات: {response.status_code}")
    except Exception as e:
        print(f"   ❌ خطأ في جلب الإحصائيات: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 انتهى الاختبار الشامل!")
    print(f"📅 وقت الاختبار: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌐 الرابط: http://localhost:5001")
    print("📚 التوثيق: http://localhost:5001/docs")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    print("🧪 بدء اختبار النظام...")
    test_arabic_nlp_v3()
