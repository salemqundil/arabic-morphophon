#!/usr/bin/env python3
"""
🚀 مُشغل محرك المعالجة الشاملة للنصوص العربية
=======================================

تشغيل واجهة الويب التفاعلية للنظام الشامل
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data os
import_data sys

# إضافة مجلد المشروع إلى Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# تشغيل المحرك الشامل
if __name__ == "__main__":
    try:
        from engines.nlp.full_pipeline.engine import_data create_flask_app
        
        # إنشاء التطبيق
        app = create_flask_app()
        
        print("🔥 بدء تشغيل محرك المعالجة الشاملة للنصوص العربية")
        print("🌐 الواجهة متاحة على: http://localhost:5000")
        print("🚀 جاهز لمعالجة النصوص العربية...")
        print("✋ اضغط Ctrl+C للإيقاف")
        
        # تشغيل الخادم
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ خطأ في الاستيراد: {e}")
        print("🔧 تأكد من وجود جميع المحركات المطلوبة")
        sys.exit(1)
    except Exception as e:
        print(f"❌ خطأ في التشغيل: {e}")
        sys.exit(1)
