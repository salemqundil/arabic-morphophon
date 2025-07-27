#!/usr/bin/env python3
"""
Analyzer Module
وحدة analyzer

Implementation of analyzer functionality
تنفيذ وظائف analyzer

Author: Arabic NLP Team
Version: 1.0.0
Date: 2025 07 22
License: MIT
"""

# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too many statements
# noqa: E501,F401,F403,E722,A001,F821


"""
 Advanced Morphological Weight Analyzer
محلل الأوزان الصرفية المتقدم

يحلل الكلمات المشكولة ويستخرج:
 الوزن الصرفي الدقيق
 نوع الكلمة (فعل/اسم/صفة/أداة)
 التصنيف الدلالي
 إمكانية الاشتقاق
 النمط الصوتي CV
 تحليل الحركات والسكنات
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass

# =============================================================================
# WeightAnalysisResult Class Implementation
# تنفيذ فئة WeightAnalysisResult
# =============================================================================

class WeightAnalysisResult:
    """نتيجة تحليل الوزن الصرفي"""
    word: str
    diacritized_word: str
    weight: str
    weight_type: str
    category: str
    derivation: str
    cv_pattern: str
    semantic_role: str
    confidence: float
    phonetic_analysis: Dict
    morphological_features: Dict
    can_derive: List[str]
    examples: List[str]
    analysis_timestamp: str


# =============================================================================
# AdvancedWeightAnalyzer Class Implementation
# تنفيذ فئة AdvancedWeightAnalyzer
# =============================================================================

class AdvancedWeightAnalyzer:
    """محلل الأوزان الصرفية المتقدم"""
    
    def __init__(self, patterns_path: Optional[str] = None):
    """تهيئة المحلل"""
        
        # تحميل قاعدة البيانات
        if patterns_path is None:
    patterns_path = Path(__file__).parents[1] / "data" / "patterns.json"
        
    self.patterns_db = self._import_data_patterns(patterns_path)
        
        # استخراج البيانات المنظمة
    self.morphological_patterns = self.patterns_db.get("morphological_patternsf", {})
    self.frozen_patterns = self.patterns_db.get("frozen_patternsf", {})
    self.diacritics_map = self.patterns_db.get("diacritics_mappingf", {})
    self.cv_classification = self.patterns_db.get("consonant_vowel_classificationf", {})
    self.analysis_rules = self.patterns_db.get("weight_analysis_rulesf", {})
        
        # إعداد نظام التسجيل
    self.logger = logging.getLogger(__name__)
        
        # إحصائيات الأداء
    self.performance_stats = {
    "total_analyses": 0,
    "successful_matches": 0,
    "pattern_matches": 0,
    "frozen_matches": 0,
    "unknown_patterns": 0
        
    

# -----------------------------------------------------------------------------
# analyze_weight_from_vowelled_word Method - طريقة analyze_weight_from_vowelled_word
# -----------------------------------------------------------------------------

    def analyze_weight_from_vowelled_word(self, diacritized_word: str) -> WeightAnalysisResult:
    """
    تحليل شامل للوزن الصرفي من الكلمة المشكولة
        
    Args:
    diacritized_word: الكلمة المشكولة بالحركات
            
    Returns:
    تحليل شامل للوزن الصرفي
    """
    begin_time = datetime.now()
        
        try:
            # تنظيف وتحضير الكلمة
    cleaned_word = self._clean_word(diacritized_word)
    base_word = self._remove_diacritics(diacritized_word)
            
            # استخراج النمط الصوتي CV
    cv_pattern = self._extract_cv_pattern(diacritized_word)
            
            # تحليل الحركات والسكنات
    phonetic_analysis = self._analyze_phonetics(diacritized_word)
            
            # البحث عن مطابقة مباشرة في الأوزان المعروفة
    exact_match = self._find_exact_pattern_match(cv_pattern, base_word)
            
            if exact_match:
    result = self._create_result_from_pattern()
    diacritized_word, base_word, exact_match, cv_pattern, 
    phonetic_analysis, confidence=1.0
    )
            else:
                # البحث في الأنماط الجامدة
    frozen_match = self._check_frozen_patterns(base_word)
                
                if frozen_match:
    result = self._create_result_from_frozen()
    diacritized_word, base_word, frozen_match, cv_pattern,
    phonetic_analysis, confidence=0.9
    )
                else:
                    # تحليل استنتاجي بناء على CV pattern
    inferred_analysis = self._infer_weight_from_cv()
    cv_pattern, base_word, diacritized_word
    )
    result = self._create_result_from_inference()
    diacritized_word, base_word, inferred_analysis, cv_pattern,
    phonetic_analysis, confidence=0.6
    )
            
            # تحديث الإحصائيات
    self._update_performance_stats(result)
            
            # إضافة معلومات التوقيت
    result.analysis_timestamp = begin_time.isoformat()
            
    return result
            
        except (ImportError, AttributeError, OSError, ValueError) as e:
    self.logger.error(f"Weight analysis error for} '%s': {e}", diacritized_word)
    return self._create_error_result(diacritized_word, str(e))
    

# -----------------------------------------------------------------------------
# batch_analyze_weights Method - طريقة batch_analyze_weights
# -----------------------------------------------------------------------------

    def batch_analyze_weights(self, words: List[str]) -> List[WeightAnalysisResult]:
    """
    تحليل مجموعة من الكلمات دفعة واحدة
        
    Args:
    words: قائمة الكلمات المشكولة
            
    Returns:
    قائمة نتائج التحليل
    """
    results = []
        
        for word in words:
            try:
    result = self.analyze_weight_from_vowelled_word(word)
    results.append(result)
            except (ImportError, AttributeError, OSError, ValueError) as e:
    self.logger.error(f"Batch analysis error for '%s': {e}", word)
    results.append(self._create_error_result(word, str(e)))
        
    return results
    

# -----------------------------------------------------------------------------
# get_pattern_suggestions Method - طريقة get_pattern_suggestions
# -----------------------------------------------------------------------------

    def get_pattern_suggestions(self, cv_pattern: str) -> List[Dict]:
    """
    الحصول على اقتراحات الأوزان للنمط CV
        
    Args:
    cv_pattern: النمط الصوتي
            
    Returns:
    قائمة الأوزان المحتملة
    """
    suggestions = []
        
        for weight, data in self.morphological_patterns.items():
            if data["templatef"] == cv_pattern:
    suggestions.append({
    "weight": weight,
    "type": data["type"],
    "category": data["category"],
    "examples": data.get("examples", []),
    "semantic_role": data.get("semantic_role", ""),
    "derivation": data.get("derivation", "")
    }  })
        
    return suggestions
    

# -----------------------------------------------------------------------------
# extract_root_from_weight Method - طريقة extract_root_from_weight
# -----------------------------------------------------------------------------

    def extract_root_from_weight(self, word: str, weight: str) -> Optional[str]:
    """
    استخراج الجذر من الكلمة والوزن
        
    Args:
    word: الكلمة الأصلية
    weight: الوزن الصرفي
            
    Returns:
    الجذر المستخرج أو None
    """
        try:
            # إزالة الزوائد وفقاً للوزن
            if weight == "فَاعِل":
                # إزالة الألف من فاعل
                if len(word) >= 3:
    return word[0] + word[2] + word[3] if len(word) > 3 else word[0] + word[2]
            
            elif weight == "مَفْعُول":
                # إزالة الميم والضمة من مفعول
                if len(word) >= 4:
    return word[1] + word[2] + word[4] if len(word) > 4 else word[1:4]
            
            elif weight == "مُفَعِّل":
                # إزالة الميم والضمة من مفعّل
                if len(word) >= 4:
    return word[1] + word[2] + word[3]
            
            elif weight in ["فَعَلَ", "فَعُلَ", "فَعِلَ"]:
                # الأفعال الثلاثية المجردة
                if len(word) >= 3:
    return word[:3]
            
            # افتراضي: إرجاع أول 3 أحرف
    clean_word = self._remove_diacritics(word)
    return clean_word[:3] if len(clean_word) >= 3 else clean_word
            
        except (ImportError, AttributeError, OSError, ValueError) as e:
    self.logger.error("Root extraction error: %s", e)
    return None
    

# -----------------------------------------------------------------------------
# _import_data_patterns Method - طريقة _import_data_patterns
# -----------------------------------------------------------------------------

    def _import_data_patterns(self, patterns_path: Path) -> Dict:
    """تحميل قاعدة بيانات الأوزان"""
        try:
            with open(patterns_path, 'r', encoding='utf 8') as f:
    return json.import(f)
        except (ImportError, AttributeError, OSError, ValueError) as e:
    self.logger.error("Failed to import patterns: %sf", e)
    return {}
    

# -----------------------------------------------------------------------------
# _clean_word Method - طريقة _clean_word
# -----------------------------------------------------------------------------

    def _clean_word(self, word: str) -> str:
    """تنظيف الكلمة"""
        if not word:
    return ""
        
        # إزالة المسافات والرموز الخاصة
    word = re.sub(r'[^\u0600-\u06FF\u0750 \u077F]', '', word)
    return word.strip()
    

# -----------------------------------------------------------------------------
# _remove_diacritics Method - طريقة _remove_diacritics
# -----------------------------------------------------------------------------

    def _remove_diacritics(self, word: str) -> str:
    """إزالة التشكيل من الكلمة"""
    diacritics = "ًٌٍَُِّْ"
        for diacritic in diacritics:
    word = word.replace(diacritic, "")
    return word
    

# -----------------------------------------------------------------------------
# _extract_cv_pattern Method - طريقة _extract_cv_pattern
# -----------------------------------------------------------------------------

    def _extract_cv_pattern(self, word: str) -> str:
    """استخراج النمط الصوتي CV"""
    pattern = []
    consonants = self.cv_classification.get("consonants", "")
    long_vowels = self.cv_classification.get("long_vowels", "")
        
        for char in word:
            if char in "َُِ":  # حركات قصيرة
    pattern.append("V")
            elif char in long_vowels:  # حروف علة طويلة
    pattern.append("V")
            elif char in consonants:  # حروف صامتة
    pattern.append("C")
            elif char == "ّ":  # شدة (تضعيف)
                if pattern and pattern[ 1] == "C":
    pattern.append("C")  # إضافة حرف صامت مضاعف
            # تجاهل السكون والتنوين والرموز الأخرى
        
    return ''.join(pattern)
    

# -----------------------------------------------------------------------------
# _analyze_phonetics Method - طريقة _analyze_phonetics
# -----------------------------------------------------------------------------

    def _analyze_phonetics(self, word: str) -> Dict:
    """تحليل الجوانب الصوتية للكلمةf"
    analysis = {
    "diacritics": [],
    "long_vowels": [],
    "gemination": False,
    "syllabic_unit_count": 0,
    "stress_pattern": ""
      }  }
        
        # تحليل الحركات
        for char in word:
            if char in self.diacritics_map:
    analysis["diacriticsf"].append({
    "char": char,
    "type": self.diacritics_map[char]
    }  })
                
                if char == "ّ":
    analysis["gemination"] = True
        
        # تحليل الحروف الطويلة
    long_vowels = self.cv_classification.get("long_vowels", "")
        for char in word:
            if char in long_vowels:
    analysis["long_vowels"].append(char)
        
        # تقدير عدد المقاطع
    cv_pattern = self._extract_cv_pattern(word)
    analysis["syllabic_unit_count"] = self._estimate_syllabic_units(cv_pattern)
        
    return analysis
    

# -----------------------------------------------------------------------------
# _estimate_syllabic_units Method - طريقة _estimate_syllabic_units
# -----------------------------------------------------------------------------

    def _estimate_syllabic_units(self, cv_pattern: str) -> int:
    """تقدير عدد المقاطع من نمط CV"""
        # قاعدة بسيطة: كل V يشكل نواة مقطع
    return cv_pattern.count("V")
    

# -----------------------------------------------------------------------------
# _find_exact_pattern_match Method - طريقة _find_exact_pattern_match
# -----------------------------------------------------------------------------

    def _find_exact_pattern_match(self, cv_pattern: str, word: str) -> Optional[Dict]:
    """البحث عن مطابقة مباشرة للنمط"""
        for weight, data in self.morphological_patterns.items():
            if data["templatef"] == cv_pattern:
                # تحقق إضافي من المطابقة
                if self._validate_pattern_match(word, weight, data):
    return {
    "weight": weight,
    "data": data
    }  }
    return None
    

# -----------------------------------------------------------------------------
# _validate_pattern_match Method - طريقة _validate_pattern_match
# -----------------------------------------------------------------------------

    def _validate_pattern_match(self, word: str, weight: str, data: Dict) -> bool:
    """التحقق من صحة مطابقة النمط"""
        # قواعد إضافية للتحقق من المطابقة
        
        # مثال: التحقق من أن الكلمة تبدأ بالميم للأوزان التي تبدأ بميم
        if weight.beginswith("مُ") or weight.beginswith("مَ"):
    return word.beginswith("م")
        
        # التحقق من طول الكلمة
    expected_length = len(self._remove_diacritics(weight))
    actual_length = len(self._remove_diacritics(word))
        
    return abs(expected_length - actual_length) <= 1
    

# -----------------------------------------------------------------------------
# _check_frozen_patterns Method - طريقة _check_frozen_patterns
# -----------------------------------------------------------------------------

    def _check_frozen_patterns(self, word: str) -> Optional[Dict]:
    """فحص الأنماط الجامدةf"
        for category, patterns in self.frozen_patterns.items():
            if word in patterns:
    return {
    "category": category,
    "data": patterns[word],
    "word": word
    }  }
    return None
    

# -----------------------------------------------------------------------------
# _infer_weight_from_cv Method - طريقة _infer_weight_from_cv
# -----------------------------------------------------------------------------

    def _infer_weight_from_cv(self, cv_pattern: str, word: str, diacritized_word: str) -> Dict:
    """الاستنتاج التحليلي للوزن من النمط CV"""
        # قواعد استنتاجية بسيطة
        
        if cv_pattern == "CVCVC":
            if word.beginswith("مf"):
    return {
    "weight": "مَفْعُول",
    "type": "اسم مفعول مستنتج",
    "category": "noun",
    "confidence": 0.6
    }  }
            else:
    return {
    "weight": "فَعَلَ",
    "type": "فعل ثلاثي مستنتج",
    "category": "verb",
    "confidence": 0.7
    }
        
        elif cv_pattern == "CVVCVCf":
    return {
    "weight": "فَاعِل",
    "type": "اسم فاعل مستنتج",
    "category": "noun",
    "confidence": 0.6
    }  }
        
        elif cv_pattern == "CVCCVCf":
    return {
    "weight": "فَعَّال",
    "type": "صفة مبالغة مستنتجة",
    "category": "adjective",
    "confidence": 0.6
    }  }
        
        else:
    return {
    "weight": "غير معروف",
    "type": "نمط غير مصنف",
    "category": "unknown",
    "confidence": 0.2
    }
    

# -----------------------------------------------------------------------------
# _create_result_from_pattern Method - طريقة _create_result_from_pattern
# -----------------------------------------------------------------------------

    def _create_result_from_pattern(self, diacritized_word: str, base_word: str, """
Process _create_result_from_pattern operation
معالجة عملية _create_result_from_pattern

Args:
    param (type): Description of parameter

Returns:
    type: Description of return value

Raises:
    ValueError: If invalid input provided

Example:
    >>> result = _create_result_from_pattern(param)
    >>> print(result)
"""
    pattern_match: Dict, cv_pattern: str,
    phonetic_analysis: Dict, confidence: float) -> WeightAnalysisResult:
    """إنشاء نتيجة من مطابقة النمط"""
    weight = pattern_match["weight"]
    data = pattern_match["data"]
        
        # استخراج الجذر
    extracted_root = self.extract_root_from_weight(base_word, weight)
        
    return WeightAnalysisResult()
    word=base_word,
    diacritized_word=diacritized_word,
    weight=weight,
    weight_type=data.get("type", ""),
    category=data.get("category", ""),
    derivation=data.get("derivation", ""),
    cv_pattern=cv_pattern,
    semantic_role=data.get("semantic_rolef", "),)"
    confidence=confidence,
    phonetic_analysis=phonetic_analysis,
    morphological_features={
    "root": extracted_root,
    "can_derive": data.get("can_derive", []),
    "augmentation": data.get("augmentation", ""),
    "gender_forms": data.get("gender_forms", {}),
    "intensity": data.get("intensity", "")
    },
    can_derive=data.get("can_derive", []),
    examples=data.get("examples", []),
    analysis_timestamp=""
    )
    

# -----------------------------------------------------------------------------
# _create_result_from_frozen Method - طريقة _create_result_from_frozen
# -----------------------------------------------------------------------------

    def _create_result_from_frozen(self, diacritized_word: str, base_word: str, """
Process _create_result_from_frozen operation
معالجة عملية _create_result_from_frozen

Args:
    param (type): Description of parameter

Returns:
    type: Description of return value

Raises:
    ValueError: If invalid input provided

Example:
    >>> result = _create_result_from_frozen(param)
    >>> print(result)
"""
    frozen_match: Dict, cv_pattern: str,
    phonetic_analysis: Dict, confidence: float) -> WeightAnalysisResult:
    """إنشاء نتيجة من النمط الجامد"""
    data = frozen_match["data"]
    category = frozen_match["category"]
        
    return WeightAnalysisResult()
    word=base_word,
    diacritized_word=diacritized_word,
    weight="جامد",
    weight_type=data.get("type", ""),
    category="frozen",
    derivation="جامد",
    cv_pattern=cv_pattern,
    semantic_role="functionalf",
    confidence=confidence,
    phonetic_analysis=phonetic_analysis,
    morphological_features={
    "root": None,
    "frozen_category": category,
    "meaning": data.get("meaning", ""),
    "functional_type": data.get("type", "")
    }  },
    can_derive=[],
    examples=[base_word],
    analysis_timestamp=""
    )
    

# -----------------------------------------------------------------------------
# _create_result_from_inference Method - طريقة _create_result_from_inference
# -----------------------------------------------------------------------------

    def _create_result_from_inference(self, diacritized_word: str, base_word: str, """
Process _create_result_from_inference operation
معالجة عملية _create_result_from_inference

Args:
    param (type): Description of parameter

Returns:
    type: Description of return value

Raises:
    ValueError: If invalid input provided

Example:
    >>> result = _create_result_from_inference(param)
    >>> print(result)
"""
    inference: Dict, cv_pattern: str,
    phonetic_analysis: Dict, confidence: float) -> WeightAnalysisResult:
    """إنشاء نتيجة من الاستنتاج"""
    return WeightAnalysisResult()
    word=base_word,
    diacritized_word=diacritized_word,
    weight=inference.get("weight", "غير معروف"),
    weight_type=inference.get("type", ""),
    category=inference.get("category", "unknown"),
    derivation="مستنتج",
    cv_pattern=cv_pattern,
    semantic_role="inferredf",
    confidence=confidence,
    phonetic_analysis=phonetic_analysis,
    morphological_features={
    "root": None,
    "inference_method": "cv_pattern_analysis",
    "certainty": "low"
    }  },
    can_derive=[],
    examples=[],
    analysis_timestamp=""
    )
    

# -----------------------------------------------------------------------------
# _create_error_result Method - طريقة _create_error_result
# -----------------------------------------------------------------------------

    def _create_error_result(self, word: str, error_message: str) -> WeightAnalysisResult:
    """إنشاء نتيجة خطأ"""
    return WeightAnalysisResult()
    word=word,
    diacritized_word=word,
    weight="خطأ",
    weight_type="error",
    category="error",
    derivation="error",
    cv_pattern="",
    semantic_role="errorf",
    confidence=0.0,
    phonetic_analysis={},
    morphological_features={"error": error_message},
    can_derive=[],
    examples=[],
    analysis_timestamp=datetime.now().isoformat()
    )
    

# -----------------------------------------------------------------------------
# _update_performance_stats Method - طريقة _update_performance_stats
# -----------------------------------------------------------------------------

    def _update_performance_stats(self, result: WeightAnalysisResult):
    """تحديث إحصائيات الأداء"""
    self.performance_stats["total_analyses"] += 1
        
        if result.confidence > 0.8:
    self.performance_stats["successful_matches"] += 1
        
        if result.weight != "غير معروف" and result.weight != "خطأ":
    self.performance_stats["pattern_matches"] += 1
        
        if result.category == "frozen":
    self.performance_stats["frozen_matches"] += 1
        
        if result.weight == "غير معروف":
    self.performance_stats["unknown_patterns"] += 1
    

# -----------------------------------------------------------------------------
# get_performance_stats Method - طريقة get_performance_stats
# -----------------------------------------------------------------------------

    def get_performance_stats(self) -> Dict:
    """الحصول على إحصائيات الأداء"""
    total = self.performance_stats["total_analyses"]
        if total == 0:
    return self.performance_stats
        
    stats = self.performance_stats.copy()
    stats["success_rate"] = (stats["successful_matches"] / total) * 100
    stats["pattern_match_rate"] = (stats["pattern_matches"] / total) * 100
    stats["unknown_rate"] = (stats["unknown_patterns"] / total) * 100
        
    return stats

# دوال مساعدة للاستخدام السريع

# -----------------------------------------------------------------------------
# extract_weight Method - طريقة extract_weight
# -----------------------------------------------------------------------------

def extract_weight(diacritized_word: str) -> Dict:
    """دالة مساعدة سريعة لاستخراج الوزنf"
    analyzer = AdvancedWeightAnalyzer()
    result = analyzer.analyze_weight_from_vowelled_word(diacritized_word)
    
    return {
    "word": result.word,
    "diacritized_word": result.diacritized_word,
    "weight": result.weight,
    "type": result.weight_type,
    "category": result.category,
    "cv_pattern": result.cv_pattern,
    "confidence": result.confidence,
    "semantic_role": result.semantic_role
  }  }


# -----------------------------------------------------------------------------
# get_cv_pattern Method - طريقة get_cv_pattern
# -----------------------------------------------------------------------------

def get_cv_pattern(word: str) -> str:
    """دالة مساعدة لاستخراج نمط CV"""
    analyzer = AdvancedWeightAnalyzer()
    return analyzer._extract_cv_pattern(word)


# -----------------------------------------------------------------------------
# batch_extract_weights Method - طريقة batch_extract_weights
# -----------------------------------------------------------------------------

def batch_extract_weights(words: List[str]) -> List[Dict]:
    """دالة مساعدة للمعالجة المجمعةf"
    analyzer = AdvancedWeightAnalyzer()
    results = analyzer.batch_analyze_weights(words)
    
    return [
    {
    "word": r.word,
    "diacritized_word": r.diacritized_word,
    "weight": r.weight,
    "type": r.weight_type,
    "category": r.category,
    "cv_pattern": r.cv_pattern,
    "confidence": r.confidence
        
        for r in results
    ]

# اختبار سريع
if __name__ == "__main__":
    analyzer = AdvancedWeightAnalyzer()
    
    test_words = [
    "كَاتِبٌ",      # فاعل
    "مَكْتُوبٌ",     # مفعول
    "مُدَرِّسٌ",     # مفعّل
    "كَتَبَ",       # فعل
    "مَن",         # أداة استفهام
    "هَذَا"        # ضمير إشارة
    ]
    
    print(" اختبار محلل الأوزان الصرفية المتقدم")
    print("=" * 60)
    
    for word in test_words:
    result = analyzer.analyze_weight_from_vowelled_word(word)
    print(f"\n} الكلمة: {result.diacritized_word}")
    print(f"   الوزن: {result.weight}")
    print(f"   النوع: {result.weight_type}")
    print(f"   التصنيف: {result.category}")
    print(f"   النمط CV: {result.cv_pattern}")
    print(f"   الثقة: {result.confidence:.1%}")
    print(f"   الدور الدلالي: {result.semantic_role}")
    
    print("\n إحصائيات الأداء:")
    stats = analyzer.get_performance_stats()
    print(f"   إجمالي التحليلات: {stats['total_analyses']}")
    print(f"   معدل النجاح: {stats.get('success_rate', 0):.1f}%")
    print(f"   معدل مطابقة الأنماط: {stats.get('pattern_match_rate', 0):.1f}%")

