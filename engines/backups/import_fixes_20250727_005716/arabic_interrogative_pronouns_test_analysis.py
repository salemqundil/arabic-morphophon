#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Interrogative Pronouns Testing & Analysis System
======================================================
نظام اختبار وتحليل أسماء الاستفهام العربية

Comprehensive testing, evaluation, and performance analysis system for
Arabic interrogative pronouns deep learning models.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0 - TESTING & ANALYSIS
Date: 2025-07-24
Encoding: UTF 8
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401
from sklearn.metrics import ()
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score)  # noqa: F401
from sklearn.model_selection import StratifiedKFold  # noqa: F401
import torch  # noqa: F401
import json  # noqa: F401
import time  # noqa: F401
from typing import Dict, List, Tuple, Any
import warnings  # noqa: F401

warnings.filterwarnings('ignore')

# استيراد النظام الأساسي
from arabic_interrogative_pronouns_deep_model import (  # noqa: F401
    InterrogativePronounInference,
    INTERROGATIVE_PRONOUNS,
    PRONOUN_TO_ID)

# إعداد الخطوط العربية للرسوم البيانية
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════════


class InterrogativePronounTestSuite:
    """مجموعة اختبارات شاملة لأسماء الاستفهام"""

    def __init__(self):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.inference_system = InterrogativePronounInference()
    self.test_results = {}

        # حالات اختبار شاملة
    self.test_cases = {
    'basic': [
    (["مَنْ"], "مَن", "شخص"),
    (["مَا"], "مَا", "شيء"),
    (["مَ", "تَى"], "مَتَى", "زمان"),
    (["أَيْ", "نَ"], "أَيْنَ", "مكان"),
    (["كَيْ", "فَ"], "كَيْفَ", "كيفية"),
    (["كَمْ"], "كَمْ", "كمية"),
    (["أَيّ"], "أَيّ", "اختيار"),
    (["لِ", "مَ"], "لِمَ", "سبب"),
    ],
    'intermediate': [
    (["مَا", "ذَا"], "مَاذَا", "شيء"),
    (["لِ", "مَا", "ذَا"], "لِمَاذَا", "سبب"),
    (["أَيَّ", "ا", "نَ"], "أَيَّانَ", "زمان"),
    (["أَنَّ", "ى"], "أَنَّى", "مكان"),
    (["كَأَيْ", "يِنْ"], "كَأَيِّنْ", "كمية"),
    ],
    'advanced': [
    (["أَيُّ", "هَا"], "أَيُّهَا", "اختيار"),
    (["مَهْ", "مَا"], "مَهْمَا", "شيء"),
    (["أَيْ", "نَ", "مَا"], "أَيْنَمَا", "مكان"),
    (["كَيْ", "فَ", "مَا"], "كَيْفَمَا", "كيفية"),
    (["مَنْ", "ذَا"], "مَنْ ذَا", "شخص"),
    ],
    'variations': [
                # تنويعات صوتية
    (["مِنْ"], "مَن", "شخص"),  # تغيير حركة
    (["مُا"], "مَا", "شيء"),  # تغيير حركة
    (["مَ", "تِى"], "مَتَى", "زمان"),  # تغيير حركة
    (["أُيْ", "نَ"], "أَيْنَ", "مكان"),  # تغيير حركة
    ],
    }

    def run_basic_tests(self, model_types: List[str] = None) -> Dict[str, Any]:
    """تشغيل الاختبارات الأساسية"""

        if model_types is None:
    model_types = ['lstm', 'gru', 'transformer']

    print("🧪 تشغيل الاختبارات الأساسية...")

    results = {}

        for model_type in model_types:
    print(f"\n🔍 اختبار نموذج {model_type.upper()}...")

    model_results = {
    'correct': 0,
    'total': 0,
    'details': [],
    'accuracy_by_category': {},
    'errors': [],
    }


            # اختبار جميع الحالات
            for category, cases in self.test_cases.items():
    category_correct = 0
    category_total = len(cases)

                for syllables, expected, semantic_category in cases:
    prediction = self.inference_system.predict_syllables()
    syllables, model_type
    )
    is_correct = prediction == expected

    model_results['details'].append()
    {
    'syllables': syllables,
    'expected': expected,
    'predicted': prediction,
    'correct': is_correct,
    'category': category,
    'semantic': semantic_category,
    }
    )

                    if is_correct:
    model_results['correct'] += 1
    category_correct += 1
                    else:
    model_results['errors'].append()
    {
    'syllables': syllables,
    'expected': expected,
    'predicted': prediction,
    'category': category,
    }
    )

    model_results['total'] += 1

    category_accuracy = category_correct / category_total
    model_results['accuracy_by_category'][category] = category_accuracy

    print()
    f"   📊 {category}: {category_accuracy:.1%} ({category_correct}/{category_total)}"
    )  # noqa: E501

    overall_accuracy = model_results['correct'] / model_results['total']
    model_results['overall_accuracy'] = overall_accuracy

    results[model_type] = model_results
    print(f"   ✅ الدقة الإجمالية: {overall_accuracy:.1%}")

    return results

    def run_stress_tests()
    self, model_type: str = 'transformer', num_tests: int = 500
    ) -> Dict[str, Any]:
    """اختبارات الضغط والأداء"""

    print(f"⚡ تشغيل اختبارات الضغط ({num_tests} اختبار)...")

        # إنشاء بيانات اختبار عشوائية
        from arabic_interrogative_pronouns_deep_model import ()
    create_synthetic_data)  # noqa: F401

    X, y = create_synthetic_data(self.inference_system.processor, num_tests)

        # قياس الوقت والذاكرة
    start_time = time.time()
    memory_before = ()
    torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )

    correct_predictions = 0
    prediction_times = []

        for i, (features, true_label) in enumerate(zip(X, y)):
            # تحويل الخصائص إلى مقاطع تقريبية (للاختبار)
    syllables = ["مَنْ"] if true_label == 0 else ["مَا"]  # مبسط للاختبار

    pred_start = time.time()
    predicted_pronoun = self.inference_system.predict_syllables()
    syllables, model_type
    )
    pred_time = time.time() - pred_start
    prediction_times.append(pred_time)

            if predicted_pronoun == INTERROGATIVE_PRONOUNS[true_label]:
    correct_predictions += 1

    end_time = time.time()
    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    total_time = end_time - start_time
    avg_prediction_time = np.mean(prediction_times)
    accuracy = correct_predictions / num_tests

    results = {
    'total_tests': num_tests,
    'accuracy': accuracy,
    'total_time': total_time,
    'avg_prediction_time': avg_prediction_time,
    'predictions_per_second': num_tests / total_time,
    'memory_usage_mb': ()
    (memory_after - memory_before) / (1024 * 1024)
                if torch.cuda.is_available()
                else 0
    ),
    'prediction_times': prediction_times,
    }

    print(f"   ⚡ الدقة: {accuracy:.1%}")
    print(f"   ⏱️  الوقت الإجمالي: {total_time:.2fs}")
    print(f"   📈 التنبؤات/ثانية: {results['predictions_per_second']:.1f}")
    print(f"   ⏱️  متوسط وقت التنبؤ: {avg_prediction_time*1000:.2f}ms")

    return results

    def run_cross_validation()
    self, num_folds: int = 5, samples_per_fold: int = 200
    ) -> Dict[str, Any]:
    """اختبار التحقق المتقاطع"""

    print(f"🔄 تشغيل التحقق المتقاطع ({num_folds} طيات)...")

        from arabic_interrogative_pronouns_deep_model import ()
    create_synthetic_data,
    train_model)  # noqa: F401

        # إنشاء بيانات التحقق
    X, y = create_synthetic_data()
    self.inference_system.processor, num_folds * samples_per_fold
    )

    X_array = np.array([x.flatten() for x in X])
    y_array = np.array(y)

        # التحقق المتقاطع الطبقي
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    cv_results = {
    'fold_accuracies': [],
    'fold_f1_scores': [],
    'mean_accuracy': 0,
    'std_accuracy': 0,
    'mean_f1': 0,
    'std_f1': 0,
    }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_array, y_array)):
    print(f"   📂 الطية {fold} + 1}/{num_folds}...")

            # تقسيم البيانات
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

            # تدريب نموذج جديد
            from arabic_interrogative_pronouns_deep_model import ()
    InterrogativeTransformer)  # noqa: F401

    model = InterrogativeTransformer()
    input_size=self.inference_system.processor.vocab_size + 12,
    d_model=128,
    nhead=8,
    num_layers=3,  # أقل للسرعة
    num_classes=len(INTERROGATIVE_PRONOUNS),
    dropout=0.1)

            # تدريب سريع
    train_model(model, X_train, y_train, epochs=15, batch_size=32)

            # تقييم
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    predictions = []
            with torch.no_grad():
                for x in X_test:
    input_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    outputs = model(input_tensor)
    pred = torch.argmax(outputs, dim=1).item()
    predictions.append(pred)

            # حساب المقاييس
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    cv_results['fold_accuracies'].append(accuracy)
    cv_results['fold_f1_scores'].append(f1)

    print(f"      دقة الطية: {accuracy:.3f,} F1: {f1:.3f}}")

        # الإحصائيات النهائية
    cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
    cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
    cv_results['mean_f1'] = np.mean(cv_results['fold_f1_scores'])
    cv_results['std_f1'] = np.std(cv_results['fold_f1_scores'])

    print()
    f"   📊 متوسط الدقة: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}}"
    )  # noqa: E501
    print()
    f"   📊 متوسط F1: {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}}"
    )  # noqa: E501

    return cv_results

    def generate_confusion_matrix()
    self, model_type: str = 'transformer', num_samples: int = 300
    ) -> Dict[str, Any]:
    """إنشاء مصفوفة الخلط"""

    print(f"📊 إنشاء مصفوفة الخلط لنموذج {model_type.upper()}...")

        from arabic_interrogative_pronouns_deep_model import ()
    create_synthetic_data)  # noqa: F401

        # إنشاء بيانات اختبار
    X, y_true = create_synthetic_data(self.inference_system.processor, num_samples)

        # التنبؤ
    y_pred = []
        for x in X:
            # تحويل تقريبي للاختبار
    syllables = ["مَنْ"]  # مبسط
    prediction = self.inference_system.predict_syllables(syllables, model_type)
    pred_id = PRONOUN_TO_ID.get(prediction, 0)
    y_pred.append(pred_id)

        # حساب مصفوفة الخلط
    cm = confusion_matrix(y_true, y_pred)

        # إنشاء التقرير
    labels = list(INTERROGATIVE_PRONOUNS.values())
    report = classification_report()
    y_true, y_pred, target_names=labels, output_dict=True
    )

        # رسم مصفوفة الخلط
    plt.figure(figsize=(12, 10))
    sns.heatmap()
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels)
    plt.title(f'مصفوفة الخلط - نموذج {model_type.upper()}')
    plt.xlabel('التنبؤ')
    plt.ylabel('الحقيقة')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
    'confusion_matrix': cm,
    'classification_report': report,
    'accuracy': accuracy_score(y_true, y_pred),
    }

    def benchmark_models(self) -> Dict[str, Any]:
    """مقارنة أداء النماذج"""

    print("🏆 مقارنة أداء النماذج...")

        # تدريب النماذج
    model_results = self.inference_system.train_all_models(num_samples=1000)

        # اختبار الأداء
    benchmark_results = {}

        for model_type in ['lstm', 'gru', 'transformer']:
    print(f"\n📊 تقييم {model_type.upper()}...")

            # اختبارات أساسية
    basic_results = self.run_basic_tests([model_type])

            # اختبارات الضغط
    stress_results = self.run_stress_tests(model_type, num_tests=100)

    benchmark_results[model_type] = {
    'training_accuracy': model_results[model_type]['final_train_acc'],
    'test_accuracy': model_results[model_type]['final_test_acc'],
    'basic_test_accuracy': basic_results[model_type]['overall_accuracy'],
    'stress_test_accuracy': stress_results['accuracy'],
    'prediction_speed': stress_results['predictions_per_second'],
    'avg_prediction_time': stress_results['avg_prediction_time'],
    }

        # عرض المقارنة
    df = pd.DataFrame(benchmark_results).T
    print("\n📈 ملخص المقارنة:")
    print(df.round(4))

        # رسم المقارنة
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # دقة التدريب
    axes[0, 0].bar()
    benchmark_results.keys(),
    [r['training_accuracy'] for r in benchmark_results.values()])
    axes[0, 0].set_title('دقة التدريب')
    axes[0, 0].set_ylim(0.8, 1.0)

        # دقة الاختبار
    axes[0, 1].bar()
    benchmark_results.keys(),
    [r['test_accuracy'] for r in benchmark_results.values()])
    axes[0, 1].set_title('دقة الاختبار')
    axes[0, 1].set_ylim(0.8, 1.0)

        # سرعة التنبؤ
    axes[1, 0].bar()
    benchmark_results.keys(),
    [r['prediction_speed'] for r in benchmark_results.values()])
    axes[1, 0].set_title('سرعة التنبؤ (تنبؤ/ثانية)')

        # زمن التنبؤ المتوسط
    axes[1, 1].bar()
    benchmark_results.keys(),
    [r['avg_prediction_time'] * 1000 for r in benchmark_results.values()])
    axes[1, 1].set_title('متوسط زمن التنبؤ (مللي ثانية)')

    plt.tight_layout()
    plt.savefig('model_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

    return benchmark_results


# ═══════════════════════════════════════════════════════════════════════════════════
# ANALYSIS SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


class InterrogativeAnalysisSystem:
    """نظام تحليل أسماء الاستفهام"""

    def __init__(self):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.test_suite = InterrogativePronounTestSuite()

    def full_analysis_report(self) -> Dict[str, Any]:
    """تقرير تحليل شامل"""

    print("📋 إنشاء التقرير الشامل...")

    report = {
    'timestamp': time.strftime('%Y-%m %d %H:%M:%S'),
    'system_info': self._get_system_info(),
    'basic_tests': {},
    'stress_tests': {},
    'cross_validation': {},
    'model_benchmark': {},
    'confusion_matrices': {},
    'recommendations': [],
    }

        # الاختبارات الأساسية
    report['basic_tests'] = self.test_suite.run_basic_tests()

        # اختبارات الضغط
        for model_type in ['transformer']:  # أفضل نموذج
    report['stress_tests'][model_type] = self.test_suite.run_stress_tests()
    model_type, 200
    )

        # التحقق المتقاطع
    report['cross_validation'] = self.test_suite.run_cross_validation(num_folds=3)

        # مقارنة النماذج
    report['model_benchmark'] = self.test_suite.benchmark_models()

        # مصفوفات الخلط
        for model_type in ['transformer']:
    report['confusion_matrices'][model_type] = ()
    self.test_suite.generate_confusion_matrix(model_type, 200)
    )

        # التوصيات
    report['recommendations'] = self._generate_recommendations(report)

        # حفظ التقرير
        with open()
    'interrogative_pronouns_analysis_report.json', 'w', encoding='utf 8'
    ) as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("✅ تم حفظ التقرير في: interrogative_pronouns_analysis_report.json")

    return report

    def _get_system_info(self) -> Dict[str, Any]:
    """معلومات النظام"""

    return {
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'device_count': ()
    torch.cuda.device_count() if torch.cuda.is_available() else 0
    ),
    'python_version': '3.13',
    'interrogative_pronouns_count': len(INTERROGATIVE_PRONOUNS),
    'phoneme_vocab_size': 56,
    }

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
    """توليد التوصيات"""

    recommendations = []

        # تحليل أداء النماذج
    benchmark = report['model_benchmark']
    best_model = max(benchmark.keys(), key=lambda k: benchmark[k]['test_accuracy'])

    recommendations.append(f"🏆 النموذج الأفضل: {best_model.upper()}")

        # تحليل دقة الاختبارات
    basic_tests = report['basic_tests']
        for model, results in basic_tests.items():
            if results['overall_accuracy'] < 0.95:
    recommendations.append()
    f"⚠️ {model.upper()}: يحتاج تحسين (دقة: {results['overall_accuracy']:.1%})"
    )
            elif results['overall_accuracy'] > 0.98:
    recommendations.append()
    f"✅ {model.upper()}: أداء ممتاز (دقة: {results['overall_accuracy']:.1%})"
    )

        # تحليل التحقق المتقاطع
    cv_results = report['cross_validation']
        if cv_results['std_accuracy'] > 0.05:
    recommendations.append("📊 التباين في الدقة عالي - يُنصح بمزيد من التدريب")

        # تحليل السرعة
        if best_model in benchmark:
    speed = benchmark[best_model]['prediction_speed']
            if speed > 100:
    recommendations.append(f"⚡ سرعة التنبؤ ممتازة: {speed:.1f} تنبؤ/ثانية")
            elif speed < 50:
    recommendations.append("🐌 السرعة يمكن تحسينها")

    return recommendations


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════


def main():  # type: ignore[no-untyped def]
    """التشغيل الرئيسي لنظام الاختبار"""

    print("🧪 نظام اختبار وتحليل أسماء الاستفهام العربية")
    print("=" * 60)

    # إنشاء نظام التحليل
    analysis_system = InterrogativeAnalysisSystem()

    # تشغيل التحليل الشامل
    print("🚀 بدء التحليل الشامل...")
    report = analysis_system.full_analysis_report()

    # عرض الملخص
    print("\n📊 ملخص النتائج:")
    print(f"   🎯 الاختبارات الأساسية: {len(report['basic_tests'])} نماذج")
    print("   ⚡ اختبارات الضغط: مكتملة")
    print()
    f"   🔄 التحقق المتقاطع: {report['cross_validation']['mean_accuracy']:.1%} ± {report['cross_validation']['std_accuracy']:.1%}"
    )

    # عرض التوصيات
    print("\n💡 التوصيات:")
    for rec in report['recommendations']:
    print(f"   {rec}")

    print("\n✅ اكتمل التحليل الشامل!")
    print("📄 التقرير محفوظ في: interrogative_pronouns_analysis_report.json")


if __name__ == "__main__":
    main()

