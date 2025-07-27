#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Arabic Demonstrative Pronouns Evaluation System
=======================================================
نظام التقييم المتقدم لأسماء الإشارة العربية

Comprehensive evaluation, benchmarking, and analysis system for Arabic
demonstrative pronouns classification with advanced metrics and visualizations.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0 - ADVANCED EVALUATION
Date: 2025-07-24
Encoding: UTF 8
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


import torch  # noqa: F401
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)  # noqa: F401
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve,
    auc,
)  # noqa: F401
import pandas as pd  # noqa: F401
import json  # noqa: F401
import time  # noqa: F401
from typing import Dict, List, Tuple, Any
from arabic_demonstrative_pronouns_deep_model import (  # noqa: F401
    DemonstrativePronounInference,
    DEMONSTRATIVE_PRONOUNS,
    create_synthetic_data,
    DemonstrativePhoneticProcessor,
)
import warnings  # noqa: F401

warnings.filterwarnings('ignore')

# إعداد matplotlib للغة العربية
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DemonstrativeEvaluationSystem:
    """نظام التقييم الشامل لأسماء الإشارة"""

    def __init__(self):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.inference = DemonstrativePronounInference()
    self.processor = DemonstrativePhoneticProcessor()
    self.results = {}

        # بيانات الاختبار الأساسية
    self.test_data = {
    "هذا": [["هَا", "ذَا"], ["هَـ", "ذَا"], ["هَ", "ذَا"]],
    "هذه": [["هَا", "ذِهِ"], ["هَا", "ذِه"], ["هَـ", "ذِهِ"]],
    "ذلك": [["ذَا", "لِكَ"], ["ذَالِ", "كَ"], ["ذَ", "لِكَ"]],
    "تلك": [["تِل", "كَ"], ["تِ", "لِكَ"], ["تِلْ", "كَ"]],
    "هذان": [["هَا", "ذَا", "نِ"], ["هَا", "ذَان"], ["هَـ", "ذَا", "نِ"]],
    "هذين": [["هَا", "ذَيْ", "نِ"], ["هَا", "ذَين"], ["هَـ", "ذَيْ", "نِ"]],
    "هاتان": [["هَا", "تَا", "نِ"], ["هَا", "تَان"], ["هَـ", "تَا", "نِ"]],
    "هاتين": [["هَا", "تَيْ", "نِ"], ["هَا", "تَين"], ["هَـ", "تَيْ", "نِ"]],
    "هؤلاء": [["هَا", "ؤُ", "لَا", "ءِ"], ["هَا", "ؤُلاء"], ["هَـ", "ؤُ", "لاء"]],
    "أولئك": [["أُو", "لَا", "ئِ", "كَ"], ["أُولَا", "ئِكَ"], ["أُو", "لَائِكَ"]],
    "هنا": [["هُ", "نَا"], ["هُنَا"], ["هُـ", "نَا"]],
    "هناك": [["هُ", "نَا", "كَ"], ["هُنَا", "كَ"], ["هُـ", "نَاكَ"]],
    "هاهنا": [["هَا", "هُ", "نَا"], ["هَاهُنَا"], ["هَـ", "هُنَا"]],
    "هنالك": [["هُ", "نَا", "لِ", "كَ"], ["هُنَا", "لِكَ"], ["هُـ", "نَالِكَ"]],
    "ذانك": [["ذَا", "نِ", "كَ"], ["ذَانِكَ"], ["ذَـ", "نِكَ"]],
    "تانك": [["تَا", "نِ", "كَ"], ["تَانِكَ"], ["تَـ", "نِكَ"]],
    }

    def generate_comprehensive_test_data(
    self, num_samples: int = 500
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """إنشاء بيانات اختبار شاملة"""

    X = []
    y = []
    syllables_list = []

        # البيانات الأساسية
        for pronoun, variations in self.test_data.items():
            for variation in variations:
    features = self.processor.extract_features(variation)
    X.append(features)
    y.append(list(DEMONSTRATIVE_PRONOUNS.values()).index(pronoun))
    syllables_list.append(variation)

        # بيانات اصطناعية إضافية
    X_synthetic, y_synthetic = create_synthetic_data(self.processor, num_samples)
    X.extend(X_synthetic)
    y.extend(y_synthetic)

        # إضافة المقاطع الاصطناعية
        for _ in range(len(X_synthetic)):
    syllables_list.append(["synthetic"])

    return X, y, syllables_list

    def evaluate_model_performance(self, model_type: str = 'gru') -> Dict[str, Any]:
    """تقييم أداء النموذج بشكل شامل"""

    print(f"🔍 تقييم نموذج {model_type.upper()}...")

        # إنشاء بيانات الاختبار
    X_test, y_test, syllables_test = self.generate_comprehensive_test_data(300)

        # التنبؤات
    predictions = []
    confidences = []
    inference_times = []

        for i, features in enumerate(X_test):
    start_time = time.time()

            # استخدام المقاطع الحقيقية أو الاصطناعية
            if syllables_test[i] != ["synthetic"]:
    syllables = syllables_test[i]
            else:
                # تحويل الخصائص إلى مقاطع تقريبية
    syllables = ["مق", "طع"]  # placeholder

            try:
    prediction = self.inference.predict_syllables(syllables, model_type)
    confidence_result = self.inference.predict_with_confidence(
    syllables, model_type
    )

    pred_id = list(DEMONSTRATIVE_PRONOUNS.values()).index(prediction)
    predictions.append(pred_id)
    confidences.append(confidence_result['confidence'])

            except Exception as e:
    print(f"خطأ في التنبؤ {i: {e}}")
    predictions.append(0)  # قيمة افتراضية
    confidences.append(0.0)

    inference_times.append(time.time() - start_time)

        # حساب المقاييس
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, predictions, average='weighted'
    )

        # مصفوفة الالتباس
    cm = confusion_matrix(y_test, predictions)

        # تقرير التصنيف
        class_names = list(DEMONSTRATIVE_PRONOUNS.values())
        class_report = classification_report(
    y_test, predictions, target_names=class_names, output_dict=True
    )

    results = {
    'model_type': model_type,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': cm,
    'classification_report': class_report,
    'avg_confidence': np.mean(confidences),
    'avg_inference_time': np.mean(inference_times),
    'predictions': predictions,
    'true_labels': y_test,
    'confidences': confidences,
    }

    return results

    def benchmark_all_models(self) -> Dict[str, Dict[str, Any]]:
    """قياس أداء جميع النماذج"""

    print("📊 بدء قياس الأداء الشامل...")

        # تدريب النماذج أولاً
    print("🚀 تدريب النماذج...")
    self.inference.train_all_models(num_samples=1000)

        # تقييم كل نموذج
    models = ['lstm', 'gru', 'transformer']
    results = {}

        for model_type in models:
            try:
    results[model_type] = self.evaluate_model_performance(model_type)
    print(
    f"✅ {model_type.upper():} دقة {results[model_type]['accuracy']:.3f}}"
    )  # noqa: E501
            except Exception as e:
    print(f"❌ خطأ في تقييم {model_type: {e}}")
    results[model_type] = None

    self.results = results
    return results

    def create_performance_visualization(self):  # type: ignore[no-untyped def]
    """إنشاء مخططات الأداء"""

        if not self.results:
    print("❌ لا توجد نتائج للتصور. يجب تشغيل benchmark_all_models أولاً.")
    return

        # إعداد المخططات
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('تقييم أداء نماذج أسماء الإشارة العربية', fontsize=16, y=0.95)

        # 1. مقارنة الدقة
    models = []
    accuracies = []
    f1_scores = []

        for model_type, result in self.results.items():
            if result:
    models.append(model_type.upper())
    accuracies.append(result['accuracy'])
    f1_scores.append(result['f1_score'])

    axes[0, 0].bar(
    models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8
    )
    axes[0, 0].set_title('دقة النماذج')
    axes[0, 0].set_ylabel('الدقة')
    axes[0, 0].set_ylim(0, 1)

        # إضافة القيم على الأعمدة
        for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 2. مقارنة F1 Score
    axes[0, 1].bar(
    models, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8
    )
    axes[0, 1].set_title('F1 Score النماذج')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1)

        for i, v in enumerate(f1_scores):
    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 3. مصفوفة الالتباس للنموذج الأفضل
    best_model = max(
    self.results.items(), key=lambda x: x[1]['accuracy'] if x[1] else 0
    )
        if best_model[1]:
    cm = best_model[1]['confusion_matrix']
            class_names = list(DEMONSTRATIVE_PRONOUNS.values())[: cm.shape[0]]

    sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names[: min(10, len(class_names))],
    yticklabels=class_names[: min(10, len(class_names))],
    ax=axes[1, 0],
    )
    axes[1, 0].set_title(f'مصفوفة الالتباس - {best_model[0].upper()}')
    axes[1, 0].set_xlabel('التنبؤ')
    axes[1, 0].set_ylabel('الحقيقة')

        # 4. توزيع الثقة
        if best_model[1]:
    confidences = best_model[1]['confidences']
    axes[1, 1].hist(
    confidences, bins=20, color='#96CEB4', alpha=0.8, edgecolor='black'
    )
    axes[1, 1].set_title(f'توزيع الثقة - {best_model[0].upper()}')
    axes[1, 1].set_xlabel('درجة الثقة')
    axes[1, 1].set_ylabel('العدد')
    axes[1, 1].axvline(
    np.mean(confidences),
    color='red',
    linestyle='- ',
    label=f'المتوسط: {np.mean(confidences):.3f}',
    )
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(
    'demonstrative_pronouns_evaluation.png', dpi=300, bbox_inches='tight'
    )
    plt.show()

    print("💾 تم حفظ مخططات التقييم في: demonstrative_pronouns_evaluation.png")

    def generate_detailed_report(self) -> str:
    """إنشاء تقرير مفصل عن الأداء"""

        if not self.results:
    return "❌ لا توجد نتائج متاحة"

    report = "📊 تقرير تقييم نماذج أسماء الإشارة العربية\n"
    report += "=" * 60 + "\n\n"

        # ملخص عام
    report += "📈 ملخص الأداء:\n"
    report += " " * 20 + "\n"

        for model_type, result in self.results.items():
            if result:
    report += f"🔹 {model_type.upper()}:\n"
    report += f"   • الدقة: {result['accuracy']:.3f}\n"
    report += f"   • الدقة المرجحة: {result['precision']:.3f}\n"
    report += f"   • الاستدعاء: {result['recall']:.3f}\n"
    report += f"   • F1-Score: {result['f1_score']:.3f}\n"
    report += f"   • متوسط الثقة: {result['avg_confidence']:.3f}\n"
    report += f"   • وقت الاستنتاج: {result['avg_inference_time']*1000:.2f} ميلي ثانية\n\n"

        # أفضل نموذج
        if any(self.results.values()):
    best_model = max(
    [(k, v) for k, v in self.results.items() if v],
    key=lambda x: x[1]['accuracy'],
    )

    report += f"🏆 أفضل نموذج: {best_model[0].upper()}\n"
    report += f"   دقة: {best_model[1]['accuracy']:.3f}\n\n"

        # تحليل التصنيف
    report += "📋 تحليل مفصل لكل فئة:\n"
    report += " " * 30 + "\n"

        if best_model[1]:
            class_report = best_model[1]['classification_report']
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and class_name not in [
    'accuracy',
    'macro avg',
    'weighted avg',
    ]:
    report += f"🔸 {class_name}:\n"
    report += f"   دقة: {metrics['precision']:.3f}, "
    report += f"استدعاء: {metrics['recall']:.3f}, "
    report += f"F1: {metrics['f1 score']:.3f}\n"

        # توصيات
    report += "\n💡 التوصيات:\n"
    report += " " * 15 + "\n"

        if any(self.results.values()):
    gru_acc = (
    self.results.get('gru', {}).get('accuracy', 0)
                if self.results.get('gru')
                else 0
    )
    lstm_acc = (
    self.results.get('lstm', {}).get('accuracy', 0)
                if self.results.get('lstm')
                else 0
    )

            if gru_acc > 0.9:
    report += "✅ نموذج GRU يظهر أداءً ممتازاً ويُنصح باستخدامه في الإنتاج\n"
            elif lstm_acc > 0.8:
    report += "✅ نموذج LSTM يظهر أداءً جيداً ومناسب للاستخدام\n"
            else:
    report += "⚠️ يُنصح بتحسين البيانات أو ضبط المعايير\n"

    return report

    def save_results(self, filename: str = "demonstrative_evaluation_results.json"):  # type: ignore[no-untyped def]
    """حفظ النتائج في ملف"""

        if not self.results:
    print("❌ لا توجد نتائج للحفظ")
    return

        # تحويل النتائج إلى تنسيق قابل للحفظ
    serializable_results = {}

        for model_type, result in self.results.items():
            if result:
    serializable_results[model_type] = {
    'accuracy': float(result['accuracy']),
    'precision': float(result['precision']),
    'recall': float(result['recall']),
    'f1_score': float(result['f1_score']),
    'avg_confidence': float(result['avg_confidence']),
    'avg_inference_time': float(result['avg_inference_time']),
    'confusion_matrix': result['confusion_matrix'].tolist(),
    'classification_report': result['classification_report'],
    }

        with open(filename, 'w', encoding='utf 8') as f:
    json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"💾 تم حفظ النتائج في: {filename}")


def main():  # type: ignore[no-untyped def]
    """التشغيل الرئيسي لنظام التقييم"""

    print("📊 نظام التقييم المتقدم لأسماء الإشارة العربية")
    print("=" * 55)

    # إنشاء نظام التقييم
    evaluator = DemonstrativeEvaluationSystem()

    # قياس الأداء
    print("🚀 بدء القياس الشامل...")
    evaluator.benchmark_all_models()

    # إنشاء التصورات
    print("\n📈 إنشاء المخططات...")
    evaluator.create_performance_visualization()

    # إنشاء التقرير
    print("\n📄 إنشاء التقرير المفصل...")
    report = evaluator.generate_detailed_report()
    print(report)

    # حفظ النتائج
    evaluator.save_results()

    # حفظ التقرير
    with open("demonstrative_evaluation_report.txt", "w", encoding="utf 8") as f:
    f.write(report)

    print("\n✅ اكتمل التقييم الشامل!")
    print("📁 الملفات المحفوظة:")
    print("   • demonstrative_pronouns_evaluation.png")
    print("   • demonstrative_evaluation_results.json")
    print("   • demonstrative_evaluation_report.txt")


if __name__ == "__main__":
    main()
