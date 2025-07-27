#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Launch Script for Arabic Word Tracer
سكريبت تشغيل متتبع الكلمات العربية

This script provides an easy way to launch the Arabic Word Tracer with proper
configuration and error handling.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data argparse
import_data logging
import_data os
import_data sys
from pathlib import_data Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """إعداد نظام التسجيل"""
    logs_dir = current_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        processrs=[
            logging.FileProcessr(logs_dir / 'arabic_tracer.log', encoding='utf-8'),
            logging.StreamProcessr(sys.stdout)
        ]
    )

def check_dependencies():
    """فحص التبعيات المطلوبة"""
    required_packages = ['flask', 'flask_cors']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import_data__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ حزم مفقودة: {', '.join(missing_packages)}")
        print("يرجى تثبيت الحزم المطلوبة باستخدام:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_sample_data():
    """إنشاء بيانات عينة للاختبار"""
    try:
        # Create a simple sample data structure
        sample_dir = current_dir / 'sample_data'
        sample_dir.mkdir(exist_ok=True)
        
        # Sample Arabic words for testing
        sample_words = [
            "كتاب", "يكتب", "مكتبة", "كتابة", "مكتوب",
            "درس", "يدرس", "مدرسة", "دراسة", "مدروس",
            "علم", "يعلم", "معلم", "تعليم", "معلوم"
        ]
        
        sample_file = sample_dir / 'test_words.txt'
        with open(sample_file, 'w', encoding='utf-8') as f:
            for word in sample_words:
                f.write(f"{word}\n")
        
        print(f"✅ تم إنشاء بيانات عينة في: {sample_file}")
        
    except Exception as e:
        print(f"⚠️ تحذير: لم يتم إنشاء بيانات العينة: {e}")

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(
        description='🔍 متتبع الكلمات العربية - Arabic Word Tracer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='عنوان الخادم (افتراضي: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000,
        help='منفذ الخادم (افتراضي: 5000)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='تفعيل وضع التطوير'
    )
    
    parser.add_argument(
        '--mock-engines', 
        action='store_true',
        help='استخدام محركات محاكاة للاختبار'
    )
    
    parser.add_argument(
        '--setup-only', 
        action='store_true',
        help='إعداد فقط بدون تشغيل الخادم'
    )

    args = parser.parse_args()

    # إعداد التسجيل
    setup_logging()
    logger = logging.getLogger(__name__)

    print("🔍 متتبع الكلمات العربية")
    print("=" * 50)

    # فحص التبعيات
    print("🔧 فحص التبعيات...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ جميع التبعيات متوفرة")

    # إنشاء بيانات العينة
    print("📁 إعداد بيانات العينة...")
    create_sample_data()

    # إعداد متغيرات البيئة
    if args.mock_engines:
        os.environ['USE_MOCK_ENGINES'] = 'true'
        print("🎭 تم تفعيل وضع المحركات التجريبية")

    if args.debug:
        os.environ['FLASK_ENV'] = 'development'
        print("🐛 تم تفعيل وضع التطوير")

    if args.setup_only:
        print("✅ تم الإعداد بنجاح!")
        print("لتشغيل الخادم، استخدم:")
        print(f"python {__file__} --host {args.host} --port {args.port}")
        return

    # تشغيل التطبيق
    try:
        print("🚀 بدء تشغيل الخادم...")
        print(f"🌐 الرابط: http://{args.host}:{args.port}")
        print("📱 يمكن الوصول للواجهة من أي متصفح")
        print("⏹️  للإيقاف: اضغط Ctrl+C")
        print("-" * 50)

        # استيراد وتشغيل التطبيق
        from arabic_word_tracer_app import_data app
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reimport_dataer=False  # تجنب مشاكل الاستيراد
        )

    except ImportError as e:
        logger.error(f"خطأ في استيراد التطبيق: {e}")
        print("❌ خطأ في استيراد التطبيق")
        print("تأكد من وجود ملف arabic_word_tracer_app.py")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم بواسطة المستخدم")
        sys.exit(0)

    except Exception as e:
        logger.error(f"خطأ في تشغيل الخادم: {e}")
        print(f"❌ خطأ في تشغيل الخادم: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
