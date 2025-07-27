"""
Advanced Arabic Morphophonological Analysis Web Application
تطبيق ويب متقدم للتحليل الصرفي الصوتي العربي
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data logging
import_data os
import_data sys
import_data time
from datetime import_data datetime
from threading import_data Thread
from typing import_data Any, Dict, List, Union

# Flask import_datas
from flask import_data Flask, jsonify, render_template, request, send_from_directory, session
from flask_socketio import_data SocketIO, emit, join_room, leave_room

# Import our advanced Arabic processing engine
try:
    import_data arabic_morphophon.integrator as integrator_module
    from arabic_morphophon.models.patterns import_data PatternType
    from arabic_morphophon.models.roots import_data ArabicRoot

    # Dynamic access to avoid type conflicts
    AnalysisLevel = integrator_module.AnalysisLevel  # type: ignore
    MorphophonologicalEngine = integrator_module.MorphophonologicalEngine  # type: ignore
    OutputFormat = integrator_module.OutputFormat  # type: ignore

    print("✅ تم تحميل محرك التحليل الصرفي الصوتي بنجاح")
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"❌ خطأ في تحميل المحرك: {e}")
    print("🔄 استخدام النظام الأساسي...")
    ENGINE_AVAILABLE = False

# Fallback classes for when the engine is not available
if not ENGINE_AVAILABLE:

    class MorphophonologicalEngine:
        def analyze(self, text: str, level: Any = None) -> Dict[str, Any]:
            return {
                "original_text": text,
                "analysis_level": "basic",
                "confidence_score": 0.5,
                "identified_roots": [],
                "detected_patterns": [],
                "phonological_output": text,
                "applied_rules": [],
                "syllabic_unit_structure": [],
                "syllabic_unit_count": len(text.split()),
                "stress_pattern": "",
                "morphological_breakdown": {},
                "pos_tags": [],
                "grammatical_features": {},
                "processing_time": 0.1,
                "warnings": ["استخدام النظام الأساسي"],
                "errors": [],
                "timestamp": datetime.now().isoformat(),
            }

        def store_data_result(self, result: Dict[str, Any], format_type: Any = None) -> str:
            return json.dumps(result, ensure_ascii=False, indent=2)

        def get_statistics(self) -> Dict[str, Any]:
            return {"total_analyses": 0, "cache_hits": 0}

        def clear_cache(self) -> None:
            pass

    class AnalysisLevel:
        BASIC = "basic"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
        COMPREHENSIVE = "comprehensive"

    class OutputFormat:
        JSON = "json"

def safe_get_attr(
    obj: Union[Dict[str, Any], Any], attr: str, default: Any = None
) -> Any:
    """Safely get attribute from object or dict"""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)

def convert_result_to_dict(result: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """Convert analysis result to dictionary format"""
    if isinstance(result, dict):
        return result

    # Convert object attributes to dictionary
    return {
        "original_text": safe_get_attr(result, "original_text", ""),
        "analysis_level": get_analysis_level_value(
            safe_get_attr(result, "analysis_level", "basic")
        ),
        "confidence_score": safe_get_attr(result, "confidence_score", 0.0),
        "identified_roots": safe_get_attr(result, "identified_roots", []),
        "detected_patterns": safe_get_attr(result, "detected_patterns", []),
        "phonological_output": safe_get_attr(result, "phonological_output", ""),
        "applied_rules": safe_get_attr(result, "applied_rules", []),
        "syllabic_unit_structure": safe_get_attr(result, "syllabic_unit_structure", []),
        "syllabic_unit_count": safe_get_attr(result, "syllabic_unit_count", 0),
        "stress_pattern": safe_get_attr(result, "stress_pattern", ""),
        "morphological_breakdown": safe_get_attr(result, "morphological_breakdown", {}),
        "pos_tags": safe_get_attr(result, "pos_tags", []),
        "grammatical_features": safe_get_attr(result, "grammatical_features", {}),
        "processing_time": safe_get_attr(result, "processing_time", 0.0),
        "warnings": safe_get_attr(result, "warnings", []),
        "errors": safe_get_attr(result, "errors", []),
        "timestamp": safe_get_attr(result, "timestamp", datetime.now().isoformat()),
    }

def get_analysis_level_value(level: Any) -> str:
    """Get analysis level value as string"""
    if hasattr(level, "value"):
        return level.value
    return str(level)

def create_batch_result_dict(
    result: Union[Dict[str, Any], Any], index: int
) -> Dict[str, Any]:
    """Create batch result dictionary"""
    result_dict = convert_result_to_dict(result)
    errors = safe_get_attr(result, "errors", [])

    return {
        "index": index,
        "original_text": result_dict["original_text"],
        "confidence_score": result_dict["confidence_score"],
        "processing_time": result_dict["processing_time"],
        "success": len(errors) == 0,
        "summary": {
            "roots_count": len(result_dict["identified_roots"]),
            "patterns_count": len(result_dict["detected_patterns"]),
            "syllabic_units_count": result_dict["syllabic_unit_count"],
        },
    }

def create_websocket_result_dict(result: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """Create WebSocket result dictionary"""
    result_dict = convert_result_to_dict(result)

    return {
        "original_text": result_dict["original_text"],
        "confidence_score": result_dict["confidence_score"],
        "identified_roots": result_dict["identified_roots"][:5],  # أول 5 جذور
        "detected_patterns": result_dict["detected_patterns"][:5],  # أول 5 أوزان
        "phonological_output": result_dict["phonological_output"],
        "syllabic_unit_count": result_dict["syllabic_unit_count"],
        "processing_time": result_dict["processing_time"],
        "warnings": result_dict["warnings"],
        "errors": result_dict["errors"],
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    processrs=[logging.FileProcessr("app.log"), logging.StreamProcessr()],
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "arabic_morphophon_2024")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize the morphophonological engine
def create_engine():
    """Create engine instance with proper fallback"""
    if ENGINE_AVAILABLE:
        return MorphophonologicalEngine()
    else:
        # Use the fallback class defined below
        class FallbackEngine:
            def analyze(self, text: str, level: Any = None) -> Dict[str, Any]:
                return {
                    "original_text": text,
                    "analysis_level": "basic",
                    "confidence_score": 0.5,
                    "identified_roots": [],
                    "detected_patterns": [],
                    "phonological_output": text,
                    "applied_rules": [],
                    "syllabic_unit_structure": [],
                    "syllabic_unit_count": len(text.split()),
                    "stress_pattern": "",
                    "morphological_breakdown": {},
                    "pos_tags": [],
                    "grammatical_features": {},
                    "processing_time": 0.1,
                    "warnings": ["استخدام النظام الأساسي"],
                    "errors": [],
                    "timestamp": datetime.now().isoformat(),
                }

            def store_data_result(
                self, result: Dict[str, Any], format_type: Any = None
            ) -> str:
                return json.dumps(result, ensure_ascii=False, indent=2)

            def get_statistics(self) -> Dict[str, Any]:
                return {"total_analyses": 0, "cache_hits": 0}

            def clear_cache(self) -> None:
                pass

        return FallbackEngine()

morphophon_engine = create_engine()

# Application statistics
app_stats = {
    "total_requests": 0,
    "successful_analyses": 0,
    "errors": 0,
    "begin_time": datetime.now(),
    "unique_sessions": set(),
    "analysis_history": [],
}

# Cache for recent analyses
analysis_cache = {}
MAX_CACHE_SIZE = 100

def log_analysis(session_id, text, result, analysis_time):
    """تسجيل تحليل في السجل"""
    truncated_text = f"{text[:50]}..." if len(text) > 50 else text
    log_entry = {
        "session_id": session_id[:8],  # جزء من معرف الجلسة
        "timestamp": datetime.now().isoformat(),
        "text": truncated_text,
        "confidence": getattr(result, "confidence_score", 0),
        "processing_time": analysis_time,
        "success": len(getattr(result, "errors", [])) == 0,
    }

    app_stats["analysis_history"].append(log_entry)

    # الاحتفاظ بآخر 100 تحليل فقط
    if len(app_stats["analysis_history"]) > 100:
        app_stats["analysis_history"] = app_stats["analysis_history"][-100:]

def get_analysis_level(level_str: str) -> Any:
    """Get analysis level safely"""
    if ENGINE_AVAILABLE:
        level_map = {
            "basic": AnalysisLevel.BASIC,
            "intermediate": AnalysisLevel.INTERMEDIATE,
            "advanced": AnalysisLevel.ADVANCED,
            "comprehensive": AnalysisLevel.COMPREHENSIVE,
        }
        return level_map.get(level_str, AnalysisLevel.INTERMEDIATE)
    else:
        # Use fallback strings
        return level_str

@app.route("/")
def index():
    """الصفحة الرئيسية"""
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    """API لتحليل النص"""
    try:
        app_stats["total_requests"] += 1

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "لم يتم توفير نص للتحليل"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "النص فارغ"}), 400

        # مستوى التحليل المطلوب
        analysis_level = get_analysis_level(data.get("level", "intermediate"))
        output_format = data.get("format", "json")

        # فحص الذاكرة المؤقتة
        cache_key = f"{text}_{analysis_level}"
        if cache_key in analysis_cache:
            logger.info(f"نتيجة من الذاكرة المؤقتة للنص: {text[:30]}...")
            cached_result = analysis_cache[cache_key]
            return jsonify(
                {
                    "success": True,
                    "cached": True,
                    "result": cached_result,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # تنفيذ التحليل
        begin_time = time.time()
        result = morphophon_engine.analyze(text, analysis_level)
        analysis_time = time.time() - begin_time

        # تحويل النتيجة إلى قاموس
        result_dict = convert_result_to_dict(result)

        # حفظ في الذاكرة المؤقتة
        if len(analysis_cache) >= MAX_CACHE_SIZE:
            # حذف أقدم عنصر
            oldest_key = next(iter(analysis_cache))
            del analysis_cache[oldest_key]
        analysis_cache[cache_key] = result_dict

        # تسجيل الإحصاءات
        session_id = session.get("session_id", "unknown")
        log_analysis(session_id, text, result, analysis_time)

        if len(result_dict.get("errors", [])) == 0:
            app_stats["successful_analyses"] += 1
        else:
            app_stats["errors"] += 1

        logger.info(f"تم تحليل النص: '{text[:30]}...' في {analysis_time:.3f} ثانية")

        return jsonify(
            {
                "success": True,
                "cached": False,
                "result": result_dict,
                "engine_stats": morphophon_engine.get_statistics(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        app_stats["errors"] += 1
        logger.error(f"خطأ في تحليل النص: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )

@app.route("/api/batch-analyze", methods=["POST"])
def batch_analyze():
    """تحليل متعدد للنصوص"""
    try:
        data = request.get_json()
        if not data or "texts" not in data:
            return jsonify({"error": "لم يتم توفير نصوص للتحليل"}), 400

        texts = data["texts"]
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "قائمة النصوص فارغة"}), 400

        if len(texts) > 10:  # حد أقصى للحماية
            return jsonify({"error": "الحد الأقصى 10 نصوص في المرة الواحدة"}), 400

        analysis_level = get_analysis_level(data.get("level", "intermediate"))

        results = []
        total_time = 0

        for i, text in enumerate(texts):
            if not text.strip():
                continue

            try:
                begin_time = time.time()
                result = morphophon_engine.analyze(text.strip(), analysis_level)
                analysis_time = time.time() - begin_time
                total_time += analysis_time

                # تحويل النتيجة
                result_dict = create_batch_result_dict(result, i)
                results.append(result_dict)

            except Exception as e:
                results.append(
                    {
                        "index": i,
                        "original_text": text,
                        "error": str(e),
                        "success": False,
                    }
                )

        logger.info(f"تم تحليل {len(results)} نصوص في {total_time:.3f} ثانية")

        return jsonify(
            {
                "success": True,
                "results": results,
                "total_processing_time": total_time,
                "processed_count": len(results),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"خطأ في التحليل المتعدد: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/stats")
def get_statistics():
    """إحصاءات التطبيق"""
    uptime = datetime.now() - app_stats["begin_time"]
    engine_stats = morphophon_engine.get_statistics()

    return jsonify(
        {
            "app_stats": {
                "total_requests": app_stats["total_requests"],
                "successful_analyses": app_stats["successful_analyses"],
                "errors": app_stats["errors"],
                "success_rate": app_stats["successful_analyses"]
                / max(app_stats["total_requests"], 1),
                "uptime_seconds": uptime.total_seconds(),
                "unique_sessions": len(app_stats["unique_sessions"]),
                "cache_size": len(analysis_cache),
            },
            "engine_stats": engine_stats,
            "recent_analyses": app_stats["analysis_history"][-10:],  # آخر 10 تحاليل
            "timestamp": datetime.now().isoformat(),
        }
    )

@app.route("/api/clear-cache", methods=["POST"])
def clear_cache():
    """مسح الذاكرة المؤقتة"""
    global analysis_cache
    analysis_cache.clear()
    morphophon_engine.clear_cache()

    return jsonify(
        {
            "success": True,
            "message": "تم مسح الذاكرة المؤقتة",
            "timestamp": datetime.now().isoformat(),
        }
    )

# WebSocket Events
@socketio.on("connect")
def process_connect():
    """عند الاتصال"""
    session_id = session.get("session_id")
    if not session_id:
        session_id = f"session_{datetime.now().timestamp()}"
        session["session_id"] = session_id

    app_stats["unique_sessions"].add(session_id)
    join_room(session_id)

    emit(
        "connection_response",
        {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message": "تم الاتصال بنجاح",
        },
    )

    logger.info(f"اتصال جديد: {session_id}")

@socketio.on("disconnect")
def process_disconnect():
    """عند قطع الاتصال"""
    session_id = session.get("session_id")
    if session_id:
        leave_room(session_id)

    logger.info(f"تم قطع الاتصال: {session_id}")

@socketio.on("analyze_request")
def process_analyze_request(data):
    """معالجة طلب التحليل عبر WebSocket"""
    try:
        session_id = session.get("session_id")
        text = data.get("text", "").strip()

        if not text:
            emit("analyze_error", {"error": "النص فارغ"})
            return

        analysis_level = get_analysis_level(data.get("level", "intermediate"))

        # إرسال تحديث البداية
        emit("analyze_progress", {"status": "بدء التحليل", "progress": 10})

        # تنفيذ التحليل
        begin_time = time.time()
        result = morphophon_engine.analyze(text, analysis_level)
        analysis_time = time.time() - begin_time

        # تحويل النتيجة
        result_dict = create_websocket_result_dict(result)

        # تسجيل الإحصاءات
        log_analysis(session_id, text, result, analysis_time)

        # إرسال النتيجة
        emit(
            "analyze_complete",
            {
                "success": True,
                "result": result_dict,
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info(f"تحليل WebSocket للجلسة {session_id}: '{text[:30]}...'")

    except Exception as e:
        emit(
            "analyze_error", {"error": str(e), "timestamp": datetime.now().isoformat()}
        )
        logger.error(f"خطأ في تحليل WebSocket: {e}")

# Error processrs
@app.errorprocessr(404)
def not_found(error):
    return jsonify({"error": "الصفحة غير موجودة"}), 404

@app.errorprocessr(500)
def internal_error(error):
    return jsonify({"error": "خطأ داخلي في الخادم"}), 500

@app.errorprocessr(413)
def request_entity_too_large(error):
    return jsonify({"error": "الملف كبير جداً"}), 413

# Background cleanup task
def cleanup_task():
    """مهمة تنظيف دورية"""
    while True:
        try:
            time.sleep(3600)  # كل ساعة

            # تنظيف السجلات القديمة
            if len(app_stats["analysis_history"]) > 200:
                app_stats["analysis_history"] = app_stats["analysis_history"][-100:]

            # تنظيف الذاكرة المؤقتة القديمة
            if len(analysis_cache) > MAX_CACHE_SIZE * 2:
                keys_to_remove = list(analysis_cache.keys())[:MAX_CACHE_SIZE]
                for key in keys_to_remove:
                    del analysis_cache[key]

            logger.info("تم تنفيذ مهمة التنظيف الدورية")

        except Exception as e:
            logger.error(f"خطأ في مهمة التنظيف: {e}")

# Begin background cleanup thread
cleanup_thread = Thread(target=cleanup_task, daemon=True)
cleanup_thread.begin()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"

    logger.info(f"🚀 بدء تطبيق التحليل الصرفي الصوتي العربي على المنفذ {port}")
    logger.info(f"🔧 وضع التصحيح: {'مفعل' if debug else 'معطل'}")

    # Print initial engine stats
    try:
        engine_stats = morphophon_engine.get_statistics()
        logger.info(f"📊 إحصاءات المحرك: {engine_stats}")
    except Exception as e:
        logger.warning(f"تعذر الحصول على إحصاءات المحرك: {e}")

    socketio.run(
        app, host="0.0.0.0", port=port, debug=debug, allow_unsafe_werkzeug=True
    )
