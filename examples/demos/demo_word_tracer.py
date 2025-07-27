#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Demo Script for Arabic Word Tracer
سكريبت عرض توضيحي لمتتبع الكلمات العربية

This script demonstrates the capabilities of the Arabic Word Tracer
by testing various Arabic words and showing the analysis results.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data sys
import_data time

import_data requests


def test_word_tracer():
    """اختبار متتبع الكلمات العربية"""
    
    base_url = "http://localhost:5001"
    
    # Test words with different characteristics
    test_words = [
        {
            "word": "كتاب",
            "description": "كلمة بسيطة - اسم ثلاثي مجرد",
            "expected_features": ["جذر ثلاثي", "اسم", "مفرد"]
        },
        {
            "word": "يدرس", 
            "description": "فعل مضارع",
            "expected_features": ["فعل", "مضارع", "مفرد"]
        },
        {
            "word": "مدرسة",
            "description": "اسم مكان بتاء مربوطة",
            "expected_features": ["اسم مكان", "مؤنث", "مفرد"]
        },
        {
            "word": "والطلاب",
            "description": "كلمة مركبة مع واو العطف وأل التعريف",
            "expected_features": ["بادئة", "معرف", "جمع"]
        },
        {
            "word": "استخراج",
            "description": "مصدر من الفعل العاشر",
            "expected_features": ["مصدر", "مزيد", "استفعال"]
        }
    ]
    
    print("🔍 اختبار متتبع الكلمات العربية")
    print("=" * 60)
    print(f"🌐 الخادم: {base_url}")
    print()
    
    # Test server connectivity
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            print("✅ الخادم متاح ويعمل بشكل صحيح")
        else:
            print(f"⚠️ الخادم يستجيب لكن مع خطأ: {response.status_code}")
    except Exception as e:
        print(f"❌ لا يمكن الوصول للخادم: {e}")
        print("تأكد من تشغيل الخادم أولاً باستخدام:")
        print("python run_tracer.py --mock-engines")
        return False
    
    print()
    
    # Test each word
    for i, test_case in enumerate(test_words, 1):
        word = test_case["word"]
        description = test_case["description"]
        
        print(f"📝 اختبار {i}: {word}")
        print(f"   الوصف: {description}")
        
        try:
            # Send analysis request
            response = requests.post(
                f"{base_url}/api/trace",
                json={"word": word},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display key results
                print(f"   ✅ نجح التحليل")
                print(f"   ⏱️  وقت المعالجة: {data.get('metadata', {}).get('processing_time_ms', 'غير محدد')} مللي ثانية")
                
                # Show linguistic levels analyzed
                levels = data.get('linguistic_levels', {})
                successful_levels = [level for level, data in levels.items() if data.get('status') == 'success']
                print(f"   📊 المستويات المحللة: {len(successful_levels)}/{len(levels)}")
                
                # Show trace summary if available
                if 'trace_summary' in data:
                    summary = data['trace_summary']
                    complexity = summary.get('word_complexity_score', 0) * 100
                    confidence = summary.get('analysis_confidence', 0) * 100
                    print(f"   🎯 درجة التعقيد: {complexity:.0f}%")
                    print(f"   🔍 مستوى الثقة: {confidence:.0f}%")
                    
                    characteristics = summary.get('dominant_characteristics', [])
                    if characteristics:
                        print(f"   🏷️  الخصائص المهيمنة: {', '.join(characteristics)}")
                
                # Show some specific analysis details
                if 'phonemes' in levels and levels['phonemes'].get('status') == 'success':
                    phoneme_count = levels['phonemes'].get('phoneme_count', 0)
                    print(f"   🔊 عدد الأصوات: {phoneme_count}")
                
                if 'syllabic_units' in levels and levels['syllabic_units'].get('status') == 'success':
                    syllabic_unit_count = levels['syllabic_units'].get('syllabic_unit_count', 0)
                    cv_pattern = levels['syllabic_units'].get('cv_pattern', 'غير محدد')
                    print(f"   📝 عدد المقاطع: {syllabic_unit_count}")
                    print(f"   📋 نمط CV: {cv_pattern}")
                
                if 'root' in levels and levels['root'].get('status') == 'success':
                    root = levels['root'].get('identified_root', 'غير محدد')
                    root_type = levels['root'].get('root_type', 'غير محدد')
                    print(f"   🌳 الجذر: {root} ({root_type})")
                
            else:
                print(f"   ❌ فشل التحليل - رمز الخطأ: {response.status_code}")
                print(f"   📄 رسالة الخطأ: {response.text[:100]}...")
                
        except Exception as e:
            print(f"   💥 خطأ في الطلب: {e}")
        
        print()
        time.sleep(1)  # Small delay between requests
    
    # Test engine status
    print("🔧 فحص حالة المحركات:")
    try:
        response = requests.get(f"{base_url}/api/engines")
        if response.status_code == 200:
            engines = response.json()
            for name, info in engines.items():
                status = info.get('status', 'unknown')
                engine_type = info.get('type', 'unknown')
                status_icon = "✅" if status == "active" else "❌"
                print(f"   {status_icon} {name}: {status} ({engine_type})")
        else:
            print(f"   ❌ لا يمكن الحصول على حالة المحركات: {response.status_code}")
    except Exception as e:
        print(f"   💥 خطأ في فحص المحركات: {e}")
    
    print()
    print("🎉 انتهى الاختبار!")
    print("🌐 لاستخدام الواجهة التفاعلية، افتح:")
    print(f"   {base_url}")
    
    return True

def main():
    """الدالة الرئيسية"""
    print("🚀 بدء اختبار متتبع الكلمات العربية...")
    print()
    
    success = test_word_tracer()
    
    if success:
        print("✅ نجح الاختبار!")
        sys.exit(0)
    else:
        print("❌ فشل الاختبار!")
        sys.exit(1)

if __name__ == "__main__":
    main()
