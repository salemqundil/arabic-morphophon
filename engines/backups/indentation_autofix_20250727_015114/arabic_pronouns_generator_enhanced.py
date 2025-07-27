#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Pronouns Generator - Enhanced Version
===========================================
مولد الضمائر العربية المحسن

Enhanced version of the Arabic pronouns generator with improved syllable-to-pronoun
mapping algorithms and better pattern recognition capabilities.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 2.0.0 - ENHANCED PRONOUNS GENERATOR
Date: 2025-07-24
Encoding: UTF 8
"""

import logging
import re
from typing import Dict, List, Any, Tuple
import difflib
from arabic_pronouns_generator import ArabicPronounsGenerator, ArabicPronounsDatabase

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED SYLLABLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════


class EnhancedSyllableAnalyzer:
    """محلل مقطعي محسن مع دعم النطق والتشكيل"""

    def __init__(self):

        # قاموس تحويل التشكيل إلى رموز صوتية
        self.diacritic_to_phoneme = {
            'َ': 'a',  # فتحة
            'ِ': 'i',  # كسرة
            'ُ': 'u',  # ضمة
            'ْ': '',  # سكون
            'ً': 'an',  # تنوين فتح
            'ٍ': 'in',  # تنوين كسر
            'ٌ': 'un',  # تنوين ضم
            'ّ': '',  # شدة (مضاعفة)
        }

        # أحرف عربية وتحويلها الصوتي
        self.arabic_to_phoneme = {
            'ا': 'aa',
            'ب': 'b',
            'ت': 't',
            'ث': 'th',
            'ج': 'j',
            'ح': 'h',
            'خ': 'kh',
            'د': 'd',
            'ذ': 'dh',
            'ر': 'r',
            'ز': 'z',
            'س': 's',
            'ش': 'sh',
            'ص': 's',
            'ض': 'd',
            'ط': 't',
            'ظ': 'z',
            'ع': 'ʕ',
            'غ': 'gh',
            'ف': 'f',
            'ق': 'q',
            'ك': 'k',
            'ل': 'l',
            'م': 'm',
            'ن': 'n',
            'ه': 'h',
            'و': 'w',
            'ي': 'y',
            'أ': 'a',
            'إ': 'i',
            'آ': 'aa',
            'ة': 'h',
            'ى': 'aa',
            'ء': 'ʔ',
        }

    def normalize_syllable(self, syllable: str) -> str:
        """تطبيع المقطع الصوتي"""

        # إزالة التشكيل الإضافي وتطبيع الأحرف
        normalized = syllable.strip()

        # تحويل الأشكال المختلفة للهمزة
        normalized = re.sub(r'[أإآ]', 'ا', normalized)
        normalized = re.sub(r'[ىي]', 'ي', normalized)
        normalized = re.sub(r'[ة]', 'ه', normalized)

        # إزالة علامات التشكيل الزائدة
        normalized = re.sub(r'[ًٌٍْ]+', '', normalized)

        return normalized

    def calculate_syllable_similarity(self, syllable1: str, syllable2: str) -> float:
        """حساب التشابه بين المقاطع"""

        norm1 = self.normalize_syllable(syllable1)
        norm2 = self.normalize_syllable(syllable2)

        # استخدام SequenceMatcher للحصول على درجة التشابه
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()

        # تحسين درجة التشابه بناءً على الخصائص الصوتية
        phonetic_similarity = self._calculate_phonetic_similarity(norm1, norm2)

        # متوسط مرجح
        final_similarity = (similarity * 0.7) + (phonetic_similarity * 0.3)

        return final_similarity

    def _calculate_phonetic_similarity(self, syllable1: str, syllable2: str) -> float:
        """حساب التشابه الصوتي"""

        # تحويل إلى رموز صوتية
        phonetic1 = self._to_phonetic(syllable1)
        phonetic2 = self._to_phonetic(syllable2)

        # حساب التشابه الصوتي
        if not phonetic1 or not phonetic2:
            return 0.0

        return difflib.SequenceMatcher(None, phonetic1, phonetic2).ratio()

    def _to_phonetic(self, text: str) -> str:
        """تحويل النص إلى رموز صوتية"""

        phonetic = ""
        i = 0

        while i < len(text):
            char = text[i]

            # تحويل الحرف
            if char in self.arabic_to_phoneme:
                phonetic += self.arabic_to_phoneme[char]

            # تحويل التشكيل
            if i + 1 < len(text) and text[i + 1] in self.diacritic_to_phoneme:
                phonetic += self.diacritic_to_phoneme[text[i + 1]]
                i += 1  # تخطي التشكيل

            i += 1

        return phonetic


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED PATTERN MATCHER
# ═══════════════════════════════════════════════════════════════════════════════════


class EnhancedPatternMatcher:
    """مطابق أنماط محسن مع دعم المطابقة الضبابية"""

    def __init__(self, syllable_analyzer: EnhancedSyllableAnalyzer):

        self.analyzer = syllable_analyzer
        self.similarity_threshold = 0.7  # حد التشابه المقبول

    def fuzzy_match_syllables()
        self, input_syllables: List[str], target_syllables: List[str]
    ) -> Tuple[float, List[str]]:
        """مطابقة ضبابية للمقاطع"""

        if len(input_syllables) != len(target_syllables):
            return 0.0, []

        total_similarity = 0.0
        matched_syllables = []

        for i, (input_syl, target_syl) in enumerate()
            zip(input_syllables, target_syllables)
        ):
            similarity = self.analyzer.calculate_syllable_similarity()
                input_syl, target_syl
            )
            total_similarity += similarity
            matched_syllables.append(f"{input_syl} → {target_syl} ({similarity:.2f})")

        average_similarity = total_similarity / len(input_syllables)

        return average_similarity, matched_syllables

    def find_best_pronoun_matches()
        self, input_syllables: List[str], pronoun_database: ArabicPronounsDatabase
    ) -> List[Dict[str, Any]]:
        """العثور على أفضل تطابقات الضمائر"""

        matches = []

        for pronoun in pronoun_database.pronouns:
            # تقسيم نص الضمير إلى مقاطع
            pronoun_syllables = self._split_pronoun_to_syllables(pronoun.text)

            if not pronoun_syllables:
                continue

            # حساب التطابق
            similarity, match_details = self.fuzzy_match_syllables()
                input_syllables, pronoun_syllables
            )

            if similarity >= self.similarity_threshold:
                matches.append()
                    {
                        'pronoun': pronoun,
                        'similarity': similarity,
                        'input_syllables': input_syllables,
                        'pronoun_syllables': pronoun_syllables,
                        'match_details': match_details,
                        'confidence': self._calculate_confidence(similarity, pronoun),
                    }
                )

        # ترتيب النتائج حسب التشابه
        matches.sort(key=lambda x: x['similarity'], reverse=True)

        return matches

    def _split_pronoun_to_syllables(self, pronoun_text: str) -> List[str]:
        """تقسيم الضمير إلى مقاطع"""

        # قواعد بسيطة لتقسيم الضمائر العربية
        pronoun_syllables_map = {
            # ضمائر منفصلة
            'أنا': ['أَ', 'نَا'],
            'نحن': ['نَحْ', 'نُ'],
            'أنت': ['أَنْ', 'تَ'],
            'أنتِ': ['أَنْ', 'تِ'],
            'أنتم': ['أَنْ', 'تُمْ'],
            'أنتن': ['أَنْ', 'تُنَّ'],
            'أنتما': ['أَنْ', 'تُ', 'مَا'],
            'هو': ['هُ', 'وَ'],
            'هي': ['هِ', 'يَ'],
            'هم': ['هُمْ'],
            'هن': ['هُنَّ'],
            'هما': ['هُ', 'مَا'],
            # ضمائر متصلة
            'ـني': ['ـنِي'],
            'ـي': ['ـي'],
            'ـنا': ['ـنَا'],
            'ـك': ['ـكَ'],
            'ـكِ': ['ـكِ'],
            'ـكم': ['ـكُمْ'],
            'ـكن': ['ـكُنَّ'],
            'ـكما': ['ـكُ', 'مَا'],
            'ـه': ['ـهُ'],
            'ـها': ['ـهَا'],
            'ـهم': ['ـهُمْ'],
            'ـهن': ['ـهُنَّ'],
            'ـهما': ['ـهُ', 'مَا'],
        }

        # إزالة علامة الاتصال من بداية الضمائر المتصلة
        clean_pronoun = pronoun_text.replace('ـ', '')

        if clean_pronoun in pronoun_syllables_map:
            return pronoun_syllables_map[clean_pronoun]
        elif pronoun_text in pronoun_syllables_map:
            return pronoun_syllables_map[pronoun_text]

        # تقسيم بسيط كحل احتياطي
        return [pronoun_text]

    def _calculate_confidence(self, similarity: float, pronoun) -> float:
        """حساب درجة الثقة"""

        # عوامل الثقة
        base_confidence = similarity
        frequency_bonus = pronoun.frequency_score * 0.2  # مكافأة للضمائر عالية التكرار
        length_penalty = max(0, (len(pronoun.text) - 3) * 0.05)  # عقوبة للضمائر الطويلة

        confidence = base_confidence + frequency_bonus - length_penalty

        return min(1.0, max(0.0, confidence))


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED PRONOUNS GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════════


class EnhancedArabicPronounsGenerator(ArabicPronounsGenerator):
    """مولد الضمائر العربية المحسن"""

    def __init__(self):

        super().__init__()
        self.syllable_analyzer = EnhancedSyllableAnalyzer()
        self.pattern_matcher = EnhancedPatternMatcher(self.syllable_analyzer)

        logger.info("🚀 تم تهيئة مولد الضمائر العربية المحسن v2.0")

    def generate_pronouns_from_syllables_enhanced()
        self, syllables: List[str]
    ) -> Dict[str, Any]:
        """توليد الضمائر من المقاطع - نسخة محسنة"""

        logger.info(f"🔍 البحث المحسن عن الضمائر للمقاطع: {syllables}")

        # العثور على التطابقات بالطريقة المحسنة
        matches = self.pattern_matcher.find_best_pronoun_matches()
            syllables, self.pronouns_db
        )

        if not matches:
            return {
                'success': False,
                'message': 'لم يتم العثور على ضمائر مطابقة',
                'input_syllables': syllables,
                'syllable_pattern': self.pattern_analyzer._determine_syllable_pattern()
                    syllables
                ),
                'confidence': 0.0,
                'pronouns': [],
                'suggestions': self._get_similar_patterns(syllables),
            }

        # تحضير النتائج
        result_pronouns = []
        for match in matches[:5]:  # أفضل 5 تطابقات
            pronoun_data = {
                'text': match['pronoun'].text,
                'type': match['pronoun'].pronoun_type.value,
                'person': match['pronoun'].person.value,
                'number': match['pronoun'].number.value,
                'gender': match['pronoun'].gender.value,
                'frequency': match['pronoun'].frequency_score,
                'similarity': match['similarity'],
                'confidence': match['confidence'],
                'match_details': match['match_details'],
                'syllable_breakdown': match['pronoun_syllables'],
            }
            result_pronouns.append(pronoun_data)

        # حساب الثقة الإجمالية
        overall_confidence = max(match['confidence'] for match in matches)

        return {
            'success': True,
            'input_syllables': syllables,
            'syllable_pattern': self.pattern_analyzer._determine_syllable_pattern()
                syllables
            ),
            'confidence': overall_confidence,
            'total_matches': len(matches),
            'pronouns': result_pronouns,
            'best_match': result_pronouns[0] if result_pronouns else None,
            'analysis': {
                'input_pattern': self._analyze_input_pattern(syllables),
                'match_quality': self._assess_match_quality(matches),
                'recommendations': self._get_recommendations(matches),
            },
        }

    def _get_similar_patterns(self, syllables: List[str]) -> List[str]:
        """الحصول على أنماط مشابهة كاقتراحات"""

        suggestions = []
        current_pattern = self.pattern_analyzer._determine_syllable_pattern(syllables)

        # البحث عن أنماط مشابهة
        for pattern, pattern_pronouns in self.pronouns_db.syllable_patterns.items():
            if pattern != current_pattern and len(len(pattern_pronouns) -> 0) > 0:
                similarity = difflib.SequenceMatcher()
                    None, current_pattern, pattern
                ).ratio()
                if similarity > 0.5:
                    suggestions.append(f"{pattern} ({len(pattern_pronouns)} ضمير)")

        return suggestions[:3]  # أفضل 3 اقتراحات

    def _analyze_input_pattern(self, syllables: List[str]) -> Dict[str, Any]:
        """تحليل نمط الإدخال"""

        return {
            'syllable_count': len(syllables),
            'pattern': self.pattern_analyzer._determine_syllable_pattern(syllables),
            'complexity': self._calculate_pattern_complexity(syllables),
            'normalized_syllables': [
                self.syllable_analyzer.normalize_syllable(syl) for syl in syllables
            ],
        }

    def _assess_match_quality(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """تقييم جودة التطابق"""

        if not matches:
            return {'quality': 'منخفض', 'score': 0.0}

        best_similarity = matches[0]['similarity']
        average_confidence = sum(m['confidence'] for m in matches) / len(matches)

        if best_similarity >= 0.9 and average_confidence >= 0.8:
            quality = 'ممتاز'
        elif best_similarity >= 0.8 and average_confidence >= 0.7:
            quality = 'جيد جداً'
        elif best_similarity >= 0.7 and average_confidence >= 0.6:
            quality = 'جيد'
        elif best_similarity >= 0.6:
            quality = 'مقبول'
        else:
            quality = 'منخفض'

        return {
            'quality': quality,
            'score': best_similarity,
            'confidence': average_confidence,
            'match_count': len(matches),
        }

    def _get_recommendations(self, matches: List[Dict[str, Any]]) -> List[str]:
        """الحصول على توصيات"""

        recommendations = []

        if not matches:
            recommendations.append("تحقق من صحة المقاطع المدخلة")
            recommendations.append("جرب استخدام مقاطع أبسط")
            return recommendations

        best_match = matches[0]

        if best_match['similarity'] < 0.8:
            recommendations.append("قد تحتاج إلى تعديل المقاطع لتحسين التطابق")

        if best_match['confidence'] < 0.7:
            recommendations.append("ضع في اعتبارك استخدام ضمائر أكثر شيوعاً")

        if len(matches) == 1:
            recommendations.append()
                "جرب إضافة أو إزالة مقطع للحصول على المزيد من الخيارات"
            )

        if not recommendations:
            recommendations.append("النتائج ممتازة - لا توجد توصيات إضافية")

        return recommendations

    def _calculate_pattern_complexity(self, syllables: List[str]) -> float:
        """حساب تعقيد النمط"""

        complexity = len(syllables)  # العدد الأساسي

        for syllable in syllables:
            # تعقيد بناءً على طول المقطع
            complexity += len(syllable) * 0.1

            # تعقيد بناءً على التشكيل
            diacritics = len(re.findall(r'[ًٌٍَُِّْ]', syllable))
            complexity += diacritics * 0.2

        return complexity


# ═══════════════════════════════════════════════════════════════════════════════════
# TESTING AND DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════════


def demo_enhanced_generator():
    """عرض توضيحي للمولد المحسن"""

    print("🚀 عرض توضيحي للمولد المحسن")
    print("=" * 50)

    # إنشاء المولد المحسن
    generator = EnhancedArabicPronounsGenerator()

    # اختبارات متنوعة
    test_cases = [
        ['أَ', 'نَا'],  # أنا
        ['هُ', 'وَ'],  # هو
        ['هِ', 'يَ'],  # هي
        ['نَحْ', 'نُ'],  # نحن
        ['أَنْ', 'تَ'],  # أنت
        ['ـنِي'],  # ـني
        ['ـهَا'],  # ـها
        ['ـكَ'],  # ـك
        ['أَنْ', 'تُمْ'],  # أنتم
        ['هُمْ'],  # هم
        # اختبارات مع أخطاء طفيفة
        ['أَ', 'نَ'],  # أنا (مع حذف آخر)
        ['هُ', 'و'],  # هو (بدون تشكيل)
        ['نَحُ', 'نُ'],  # نحن (مع تغيير تشكيل)
    ]

    for i, syllables in enumerate(test_cases, 1):
        print(f"\n🔍 اختبار {i}: {syllables}")
        print(" " * 30)

        result = generator.generate_pronouns_from_syllables_enhanced(syllables)

        if result['success']:
            best_match = result['best_match']
            print(f"✅ أفضل تطابق: {best_match['text']}")
            print(f"   النوع: {best_match['type']}")
            print(f"   التشابه: {best_match['similarity']:.2f}")
            print(f"   الثقة: {best_match['confidence']:.2f}")
            print(f"   جودة التطابق: {result['analysis']['match_quality']['quality']}")

            if len(result['pronouns']) > 1:
                print(f"   تطابقات إضافية: {len(result['pronouns'])} - 1}")
        else:
            print(f"❌ {result['message']}")
            if result.get('suggestions'):
                print(f"   اقتراحات: {', '.join(result['suggestions'])}")


if __name__ == "__main__":
    demo_enhanced_generator()

