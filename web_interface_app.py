#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌐 واجهة الويب المتقدمة للنظام المحسن للمتجه الرقمي العربي
Advanced Web Interface for Enhanced Arabic Digital Vector System

🎯 المميزات:
- واجهة مستخدم تفاعلية حديثة
- تحليل فوري للكلمات العربية
- عرض بصري للمتجهات والتحليلات
- مراقبة الأداء في الوقت الفعلي
- تصدير النتائج بصيغ متعددة
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import asyncio
import json
import io
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import base64
import numpy as np

# استيراد النظام المحسن
from enhanced_system_architecture import EnhancedArabicVectorSystem

# إنشاء التطبيق
app = Flask(__name__)
CORS(app)

# النظام المحسن العام
enhanced_system = EnhancedArabicVectorSystem()

# ============== الصفحات الأساسية ==============


@app.route("/")
def index():
    """الصفحة الرئيسية"""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """لوحة المراقبة"""
    return render_template("dashboard.html")


@app.route("/analysis")
def analysis():
    """صفحة التحليل التفصيلي"""
    return render_template("analysis.html")


@app.route("/batch")
def batch():
    """صفحة التحليل المجمع"""
    return render_template("batch.html")


# ============== API Endpoints ==============


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """تحليل كلمة واحدة"""
    try:
        data = request.json
        word = data.get("word", "").strip()
        optimization_level = data.get("optimization_level", "balanced")

        if not word:
            return jsonify({"error": "لم يتم تقديم كلمة للتحليل"}), 400

        # تشغيل التحليل
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                enhanced_system.analyze_word_enhanced(word, optimization_level)
            )

            # تحويل النتيجة إلى JSON
            response = {
                "word": word,
                "success": True,
                "digital_vector": result.digital_vector,
                "vector_dimensions": len(result.digital_vector),
                "quality_indicators": result.quality_indicators,
                "performance_metrics": {
                    "analysis_speed": result.performance_metrics.analysis_speed,
                    "accuracy_score": result.performance_metrics.accuracy_score,
                    "cache_hit_rate": result.performance_metrics.cache_hit_rate,
                    "parallel_efficiency": result.performance_metrics.parallel_efficiency,
                },
                "feature_importance": result.feature_importance,
                "linguistic_breakdown": result.linguistic_breakdown,
                "timestamp": datetime.now().isoformat(),
            }

            return jsonify(response)

        finally:
            loop.close()

    except Exception as e:
        return jsonify({"error": f"خطأ في التحليل: {str(e)}", "success": False}), 500


@app.route("/api/batch_analyze", methods=["POST"])
def api_batch_analyze():
    """تحليل مجموعة من الكلمات"""
    try:
        data = request.json
        words = data.get("words", [])
        optimization_level = data.get("optimization_level", "balanced")

        if not words or not isinstance(words, list):
            return jsonify({"error": "لم يتم تقديم قائمة صحيحة من الكلمات"}), 400

        # تنظيف الكلمات
        clean_words = [word.strip() for word in words if word.strip()]

        if not clean_words:
            return jsonify({"error": "لا توجد كلمات صالحة للتحليل"}), 400

        # تشغيل التحليل المجمع
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(
                enhanced_system.batch_analyze(clean_words, optimization_level)
            )

            # تحويل النتائج إلى JSON
            batch_response = {
                "total_words": len(clean_words),
                "successful_analyses": len(results),
                "success_rate": len(results) / len(clean_words) if clean_words else 0,
                "results": [],
                "summary": {
                    "average_accuracy": 0.0,
                    "average_vector_dimensions": 0,
                    "total_processing_time": 0.0,
                },
                "timestamp": datetime.now().isoformat(),
            }

            total_accuracy = 0.0
            total_dimensions = 0
            total_speed = 0.0

            for i, result in enumerate(results):
                word_result = {
                    "word": clean_words[i] if i < len(clean_words) else f"word_{i}",
                    "digital_vector": result.digital_vector,
                    "vector_dimensions": len(result.digital_vector),
                    "quality_indicators": result.quality_indicators,
                    "performance_metrics": {
                        "analysis_speed": result.performance_metrics.analysis_speed,
                        "accuracy_score": result.performance_metrics.accuracy_score,
                    },
                }

                batch_response["results"].append(word_result)

                # تجميع الإحصائيات
                total_accuracy += result.quality_indicators.get("accuracy", 0.0)
                total_dimensions += len(result.digital_vector)
                total_speed += result.performance_metrics.analysis_speed

            # حساب المتوسطات
            if results:
                batch_response["summary"]["average_accuracy"] = total_accuracy / len(
                    results
                )
                batch_response["summary"]["average_vector_dimensions"] = (
                    total_dimensions // len(results)
                )
                batch_response["summary"]["average_processing_speed"] = (
                    total_speed / len(results)
                )

            return jsonify(batch_response)

        finally:
            loop.close()

    except Exception as e:
        return (
            jsonify({"error": f"خطأ في التحليل المجمع: {str(e)}", "success": False}),
            500,
        )


@app.route("/api/statistics")
def api_statistics():
    """إحصائيات النظام"""
    try:
        stats = enhanced_system.get_system_statistics()
        return jsonify(
            {
                "success": True,
                "statistics": stats,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return (
            jsonify({"error": f"خطأ في جلب الإحصائيات: {str(e)}", "success": False}),
            500,
        )


@app.route("/api/export/<format>")
def api_export(format):
    """تصدير النتائج"""
    try:
        # جمع البيانات من قاعدة بيانات النتائج
        results_data = enhanced_system.results_database

        if not results_data:
            return jsonify({"error": "لا توجد بيانات للتصدير"}), 400

        if format.lower() == "csv":
            return export_to_csv(results_data)
        elif format.lower() == "json":
            return export_to_json(results_data)
        elif format.lower() == "excel":
            return export_to_excel(results_data)
        else:
            return jsonify({"error": "صيغة التصدير غير مدعومة"}), 400

    except Exception as e:
        return jsonify({"error": f"خطأ في التصدير: {str(e)}", "success": False}), 500


@app.route("/api/visualization/vector/<word>")
def api_visualization_vector(word):
    """تصور المتجه لكلمة معينة"""
    try:
        # البحث عن نتيجة الكلمة في قاعدة البيانات
        word_result = None
        for entry in enhanced_system.results_database:
            if entry["word"] == word:
                word_result = entry["result"]
                break

        if not word_result:
            return jsonify({"error": "الكلمة غير موجودة في قاعدة البيانات"}), 404

        # إنشاء الرسم البياني
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # رسم المتجه
        vector = word_result.digital_vector
        ax1.plot(vector, marker="o", markersize=2)
        ax1.set_title(f"المتجه الرقمي للكلمة: {word}", fontsize=14)
        ax1.set_xlabel("مؤشر الميزة")
        ax1.set_ylabel("قيمة الميزة")
        ax1.grid(True, alpha=0.3)

        # رسم أهمية الميزات
        if word_result.feature_importance:
            features = list(word_result.feature_importance.keys())
            values = list(word_result.feature_importance.values())

            ax2.bar(features, values)
            ax2.set_title("أهمية الميزات اللغوية")
            ax2.set_ylabel("مستوى الأهمية")
            ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # تحويل إلى base64
        img_buffer = io.BytesIO()
        plt.store_datafig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        return jsonify(
            {
                "success": True,
                "image": f"data:image/png;base64,{img_base64}",
                "word": word,
                "vector_info": {
                    "dimensions": len(vector),
                    "min_value": float(min(vector)),
                    "max_value": float(max(vector)),
                    "mean_value": float(np.mean(vector)),
                    "std_value": float(np.std(vector)),
                },
            }
        )

    except Exception as e:
        return jsonify({"error": f"خطأ في التصور: {str(e)}", "success": False}), 500


@app.route("/api/comparison")
def api_comparison():
    """مقارنة بين عدة كلمات"""
    try:
        words = request.args.getlist("words")

        if len(words) < 2:
            return jsonify({"error": "يجب تقديم كلمتين على الأقل للمقارنة"}), 400

        # جمع بيانات الكلمات
        words_data = []
        for word in words:
            for entry in enhanced_system.results_database:
                if entry["word"] == word:
                    words_data.append(
                        {
                            "word": word,
                            "vector": entry["result"].digital_vector,
                            "accuracy": entry["result"].quality_indicators.get(
                                "accuracy", 0.0
                            ),
                            "dimensions": len(entry["result"].digital_vector),
                        }
                    )
                    break

        if len(words_data) < 2:
            return jsonify({"error": "لم يتم العثور على بيانات كافية للمقارنة"}), 400

        # حساب المقارنات
        comparison_matrix = []
        for i, data1 in enumerate(words_data):
            row = []
            for j, data2 in enumerate(words_data):
                if i == j:
                    similarity = 1.0
                else:
                    # حساب التشابه الكوساينية
                    vec1 = np.array(data1["vector"])
                    vec2 = np.array(data2["vector"])

                    # توحيد الأطوال
                    min_len = min(len(vec1), len(vec2))
                    vec1 = vec1[:min_len]
                    vec2 = vec2[:min_len]

                    # حساب التشابه
                    dot_product = np.dot(vec1, vec2)
                    magnitude1 = np.linalg.norm(vec1)
                    magnitude2 = np.linalg.norm(vec2)

                    if magnitude1 > 0 and magnitude2 > 0:
                        similarity = float(dot_product / (magnitude1 * magnitude2))
                    else:
                        similarity = 0.0

                row.append(similarity)
            comparison_matrix.append(row)

        return jsonify(
            {
                "success": True,
                "words": [data["word"] for data in words_data],
                "comparison_matrix": comparison_matrix,
                "word_details": words_data,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": f"خطأ في المقارنة: {str(e)}", "success": False}), 500


# ============== دوال التصدير ==============


def export_to_csv(results_data):
    """تصدير إلى CSV"""
    output = io.StringIO()
    writer = csv.writer(output)

    # رؤوس الأعمدة
    headers = [
        "الكلمة",
        "أبعاد المتجه",
        "مقياس الدقة",
        "سرعة التحليل",
        "مستوى الثقة",
        "الطابع الزمني",
    ]
    writer.writerow(headers)

    # البيانات
    for entry in results_data:
        result = entry["result"]
        row = [
            entry["word"],
            len(result.digital_vector),
            result.quality_indicators.get("accuracy", 0.0),
            result.performance_metrics.analysis_speed,
            result.quality_indicators.get("reliability", 0.0),
            entry["timestamp"],
        ]
        writer.writerow(row)

    # إنشاء الاستجابة
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        downimport_data_name=f'arabic_vector_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    )


def export_to_json(results_data):
    """تصدير إلى JSON"""
    export_data = {
        "export_info": {
            "total_entries": len(results_data),
            "export_timestamp": datetime.now().isoformat(),
            "system_version": "Enhanced Arabic Vector System v1.0",
        },
        "results": [],
    }

    for entry in results_data:
        result = entry["result"]
        export_data["results"].append(
            {
                "word": entry["word"],
                "timestamp": entry["timestamp"],
                "digital_vector": result.digital_vector,
                "vector_dimensions": len(result.digital_vector),
                "quality_indicators": result.quality_indicators,
                "performance_metrics": {
                    "analysis_speed": result.performance_metrics.analysis_speed,
                    "accuracy_score": result.performance_metrics.accuracy_score,
                    "cache_hit_rate": result.performance_metrics.cache_hit_rate,
                },
                "feature_importance": result.feature_importance,
            }
        )

    json_data = json.dumps(export_data, ensure_ascii=False, indent=2)

    return send_file(
        io.BytesIO(json_data.encode("utf-8")),
        mimetype="application/json",
        as_attachment=True,
        downimport_data_name=f'arabic_vector_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    )


def export_to_excel(results_data):
    """تصدير إلى Excel"""
    # إنشاء DataFrame
    data_for_df = []

    for entry in results_data:
        result = entry["result"]
        data_for_df.append(
            {
                "الكلمة": entry["word"],
                "أبعاد المتجه": len(result.digital_vector),
                "مقياس الدقة": result.quality_indicators.get("accuracy", 0.0),
                "مقياس الاتساق": result.quality_indicators.get("consistency", 0.0),
                "مقياس الموثوقية": result.quality_indicators.get("reliability", 0.0),
                "سرعة التحليل": result.performance_metrics.analysis_speed,
                "كفاءة الذاكرة": result.performance_metrics.cache_hit_rate,
                "الطابع الزمني": entry["timestamp"],
            }
        )

    df = pd.DataFrame(data_for_df)

    # حفظ في memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="تحليل المتجهات العربية", index=False)

    output.seek(0)

    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        downimport_data_name=f'arabic_vector_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
    )


# ============== تشغيل التطبيق ==============

if __name__ == "__main__":
    print("🌐 بدء تشغيل واجهة الويب المتقدمة للنظام المحسن")
    print("=" * 60)
    print("📍 الرابط: http://localhost:5000")
    print("🎯 المميزات المتاحة:")
    print("   - تحليل فوري للكلمات العربية")
    print("   - تحليل مجمع للكلمات")
    print("   - عرض بصري للمتجهات")
    print("   - مراقبة الأداء")
    print("   - تصدير النتائج")
    print("   - مقارنة الكلمات")

    app.run(host="0.0.0.0", port=5000, debug=True)
