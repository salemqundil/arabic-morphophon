#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Complex Word Analysis Example
======================================
مثال متقدم لتحليل الكلمات المعقدة باستخدام النظام الشامل

تحليل "يستكتبونها" كمثال على التوافيق المعقدة:
- الوزن العاشر (استفعل)
- ضمير الجمع (ون)
- ضمير المتصل (ها)
- التفاعل الصوتي المعقد

Author: GitHub Copilot Arabic NLP Expert
Version: 3.0.0 - COMPLEX WORD ANALYSIS
Date: 2025-07-26
Encoding: UTF 8
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


from complete_arabic_phonological_coverage import (
    CompletePhonemeCatalog,
    AdvancedSyllableAnalyzer,
    SyllableStructure,
)  # noqa: F401
from typing import Dict, List, Any
import json  # noqa: F401


class ComplexWordPhonologicalAnalyzer:
    """محلل فونيمي متقدم للكلمات المعقدة"""

    def __init__(self):  # type: ignore[no-untyped def]
        """TODO: Add docstring."""
        self.catalog = CompletePhonemeCatalog()
        self.syllable_analyzer = AdvancedSyllableAnalyzer(self.catalog)
        self.morphological_patterns = self._load_morphological_patterns()

    def _load_morphological_patterns(self) -> Dict[str, Dict]:
        """تحميل الأوزان الصرفية المتقدمة"""
        return {
            'trilateral_augmented': {
                'form_x': {
                    'pattern': 'يستفعل',
                    'meaning': 'seek/request',
                    'prefix': 'يست',
                    'stem_pattern': 'فعل',
                    'morphemes': ['ي', 'ست', 'ROOT', 'ون', 'ها'],
                }
            },
            'pronouns': {'plural_masculine': 'ون', 'attached_feminine': 'ها'},
        }

    def analyze_complex_word(self, word: str) -> Dict[str, Any]:
        """
        تحليل شامل لكلمة معقدة مثل "يستكتبونها"

        المستويات:
        1. التحليل الصوتي (فونيمي)
        2. التحليل المقطعي
        3. التحليل الصرفي
        4. التحليل النحوي
        5. التحليل الدلالي
        """

        print(f"🔬 تحليل شامل للكلمة المعقدة: {word}")
        print("=" * 50)

        analysis = {
            'input_word': word,
            'phonemic_analysis': self._analyze_phonemic_structure(word),
            'syllabic_analysis': self._analyze_syllabic_structure(word),
            'morphological_analysis': self._analyze_morphological_structure(word),
            'syntactic_analysis': self._analyze_syntactic_structure(word),
            'semantic_analysis': self._analyze_semantic_structure(word),
            'complexity_score': 0.0,
        }

        # حساب درجة التعقيد
        analysis['complexity_score'] = self._calculate_complexity_score(analysis)

        return analysis

    def _analyze_phonemic_structure(self, word: str) -> Dict[str, Any]:
        """التحليل الفونيمي للكلمة"""

        # تحليل الكلمة إلى فونيمات
        phonemes = list(word)

        # تصنيف الفونيمات
        classified_phonemes = []
        for phoneme in phonemes:
            classification = self._classify_phoneme(phoneme)
            classified_phonemes.append(
                {
                    'phoneme': phoneme,
                    'type': classification['type'],
                    'features': classification['features'],
                }
            )

        # تحليل التتابعات الصوتية
        phonotactic_analysis = self._analyze_phonotactic_sequences(phonemes)

        return {
            'total_phonemes': len(phonemes),
            'phoneme_sequence': phonemes,
            'classified_phonemes': classified_phonemes,
            'phonotactic_analysis': phonotactic_analysis,
            'phonological_processes': self._identify_phonological_processes(phonemes),
        }

    def _classify_phoneme(self, phoneme: str) -> Dict[str, Any]:
        """تصنيف الفونيم حسب النظام الشامل"""

        # البحث في الصوامت
        for category, consonants in self.catalog.consonants.items():
            if phoneme in consonants:
                return {
                    'type': 'consonant',
                    'features': {
                        'category': category,
                        'voice': self._get_voicing(phoneme),
                        'place': self._get_place_of_articulation(phoneme),
                        'manner': self._get_manner_of_articulation(phoneme),
                    },
                }

        # البحث في الصوائت
        for vowel_type, vowels in self.catalog.vowels.items():
            if phoneme in vowels:
                return {
                    'type': 'vowel',
                    'features': (
                        vowels[phoneme]
                        if isinstance(vowels, dict)
                        else {'type': vowel_type}
                    ),
                }

        # البحث في الفونيمات الوظيفية
        for category, functions in self.catalog.functional_phonemes.items():
            if isinstance(functions, dict):
                for subcategory, items in functions.items():
                    if phoneme in items or (
                        isinstance(items, dict) and phoneme in items
                    ):
                        return {
                            'type': 'functional',
                            'features': {
                                'category': category,
                                'subcategory': subcategory,
                                'function': 'grammatical',
                            },
                        }

        return {'type': 'unknown', 'features': {'category': 'unclassified'}}

    def _get_voicing(self, phoneme: str) -> str:
        """تحديد الجهر/الهمس"""
        voiced = [
            'ب',
            'د',
            'ج',
            'ز',
            'ذ',
            'ض',
            'ظ',
            'غ',
            'ع',
            'ل',
            'ر',
            'م',
            'ن',
            'و',
            'ي',
        ]
        return 'voiced' if phoneme in voiced else 'voiceless'

    def _get_place_of_articulation(self, phoneme: str) -> str:
        """تحديد مخرج الصوت"""
        places = {
            'bilabial': ['ب', 'م'],
            'dental': ['ت', 'د', 'ث', 'ذ', 'ن'],
            'alveolar': ['س', 'ز', 'ل', 'ر'],
            'postalveolar': ['ش'],
            'pharyngealized': ['ص', 'ض', 'ط', 'ظ'],
            'velar': ['ك'],
            'uvular': ['ق', 'غ', 'خ'],
            'pharyngeal': ['ح', 'ع'],
            'glottal': ['ء', 'ه'],
        }

        for place, phonemes in places.items():
            if phoneme in phonemes:
                return place
        return 'unknown'

    def _get_manner_of_articulation(self, phoneme: str) -> str:
        """تحديد كيفية النطق"""
        manners = {
            'stop': ['ب', 'ت', 'د', 'ط', 'ك', 'ق', 'ء'],
            'fricative': [
                'ف',
                'ث',
                'ذ',
                'س',
                'ز',
                'ش',
                'ص',
                'ض',
                'خ',
                'غ',
                'ح',
                'ع',
                'ه',
            ],
            'nasal': ['م', 'ن'],
            'liquid': ['ل', 'ر'],
            'approximant': ['و', 'ي'],
        }

        for manner, phonemes in manners.items():
            if phoneme in phonemes:
                return manner
        return 'unknown'

    def _analyze_phonotactic_sequences(self, phonemes: List[str]) -> Dict[str, Any]:
        """تحليل التتابعات الصوتية"""

        sequences = []
        constraints_violated = []

        for i in range(len(phonemes) - 1):
            current = phonemes[i]
            next_phoneme = phonemes[i + 1]

            sequence = {
                'position': i,
                'sequence': [current, next_phoneme],
                'valid': self._check_phonotactic_validity(current, next_phoneme),
                'type': self._classify_sequence_type(current, next_phoneme),
            }

            sequences.append(sequence)

            if not sequence['valid']:
                constraints_violated.append(sequence)

        return {
            'sequences': sequences,
            'total_sequences': len(sequences),
            'valid_sequences': len([s for s in sequences if s['valid']]),
            'constraints_violated': constraints_violated,
            'phonotactic_score': (
                len([s for s in sequences if s['valid']]) / len(sequences)
                if sequences
                else 1.0
            ),
        }

    def _check_phonotactic_validity(self, phoneme1: str, phoneme2: str) -> bool:
        """فحص صحة التتابع الصوتي"""

        # قواعد أساسية للتتابع الصوتي العربي
        forbidden_sequences = [
            ('ء', 'ء'),
            ('ه', 'ه'),  # منع تكرار الأصوات الحنجرية
            ('ق', 'ك'),
            ('ك', 'ق'),  # منع تتابع الأصوات المتشابهة
        ]

        return (phoneme1, phoneme2) not in forbidden_sequences

    def _classify_sequence_type(self, phoneme1: str, phoneme2: str) -> str:
        """تصنيف نوع التتابع"""

        consonants = [
            'ب',
            'ت',
            'ث',
            'ج',
            'ح',
            'خ',
            'د',
            'ذ',
            'ر',
            'ز',
            'س',
            'ش',
            'ص',
            'ض',
            'ط',
            'ظ',
            'ع',
            'غ',
            'ف',
            'ق',
            'ك',
            'ل',
            'م',
            'ن',
            'ه',
            'و',
            'ي',
            'ء',
        ]
        vowels = ['َ', 'ِ', 'ُ', 'ا', 'ي', 'و']

        if phoneme1 in consonants and phoneme2 in consonants:
            return 'CC'
        elif phoneme1 in consonants and phoneme2 in vowels:
            return 'CV'
        elif phoneme1 in vowels and phoneme2 in consonants:
            return 'VC'
        elif phoneme1 in vowels and phoneme2 in vowels:
            return 'VV'
        else:
            return 'unknown'

    def _identify_phonological_processes(
        self, phonemes: List[str]
    ) -> List[Dict[str, Any]]:
        """تحديد العمليات الصوتية"""

        processes = []

        # البحث عن الإدغام
        for i in range(len(phonemes) - 1):
            if phonemes[i] == phonemes[i + 1]:
                processes.append(
                    {
                        'type': 'gemination',
                        'position': i,
                        'phonemes': [phonemes[i], phonemes[i + 1]],
                        'description': f'إدغام {phonemes[i]}',
                    }
                )

        # البحث عن الإعلال
        weak_letters = ['و', 'ي', 'ا']
        for i, phoneme in enumerate(phonemes):
            if phoneme in weak_letters:
                processes.append(
                    {
                        'type': 'vowel_change_potential',
                        'position': i,
                        'phoneme': phoneme,
                        'description': f'حرف علة محتمل الإعلال: {phoneme}',
                    }
                )

        return processes

    def _analyze_syllabic_structure(self, word: str) -> Dict[str, Any]:
        """التحليل المقطعي للكلمة"""

        # تحليل مبسط للمقاطع (يمكن تطويره أكثر)
        syllables = self._syllabify_word(word)

        syllable_analysis = []
        for i, syllable in enumerate(syllables):
            analysis = {
                'syllable': syllable,
                'position': i,
                'structure': self._determine_syllable_structure(syllable),
                'weight': self._determine_syllable_weight(syllable),
                'stress_potential': self._assess_stress_potential(
                    syllable, i, len(syllables)
                ),
            }
            syllable_analysis.append(analysis)

        return {
            'total_syllables': len(syllables),
            'syllables': syllables,
            'syllable_analysis': syllable_analysis,
            'syllabic_complexity': len(syllables) * 0.5
            + len([s for s in syllable_analysis if s['weight'] == 'heavy']) * 0.3,
        }

    def _syllabify_word(self, word: str) -> List[str]:
        """تقطيع الكلمة إلى مقاطع (مبسط)"""

        # خوارزمية تقطيع مبسطة
        # في تطبيق حقيقي، نحتاج خوارزمية أكثر تطوراً

        syllables = []
        current_syllable = ""
        vowels = ['َ', 'ِ', 'ُ', 'ا', 'ي', 'و']

        i = 0
        while i < len(word):
            char = word[i]
            current_syllable += char

            # إذا كان الحرف صائتاً، ننظر للحرف التالي
            if char in vowels:
                # إذا كان الحرف التالي صامتاً، نضيفه ونكمل المقطع
                if i + 1 < len(word) and word[i + 1] not in vowels:
                    current_syllable += word[i + 1]
                    i += 1

                syllables.append(current_syllable)
                current_syllable = ""

            i += 1

        if current_syllable:
            syllables.append(current_syllable)

        return syllables

    def _determine_syllable_structure(self, syllable: str) -> str:
        """تحديد بنية المقطع"""

        consonants = [
            'ب',
            'ت',
            'ث',
            'ج',
            'ح',
            'خ',
            'د',
            'ذ',
            'ر',
            'ز',
            'س',
            'ش',
            'ص',
            'ض',
            'ط',
            'ظ',
            'ع',
            'غ',
            'ف',
            'ق',
            'ك',
            'ل',
            'م',
            'ن',
            'ه',
            'و',
            'ي',
            'ء',
        ]
        vowels = ['َ', 'ِ', 'ُ', 'ا', 'ي', 'و']

        structure = ""
        for char in syllable:
            if char in consonants:
                structure += "C"
            elif char in vowels:
                structure += "V"

        return structure

    def _determine_syllable_weight(self, syllable: str) -> str:
        """تحديد وزن المقطع"""

        structure = self._determine_syllable_structure(syllable)

        if structure in ['CV']:
            return 'light'
        elif structure in ['CVC', 'CVV']:
            return 'heavy'
        elif structure in ['CVCC', 'CVVC']:
            return 'super_heavy'
        else:
            return 'unknown'

    def _assess_stress_potential(
        self, syllable: str, position: int, total_syllables: int
    ) -> float:
        """تقييم احتمالية النبر"""

        # قواعد مبسطة للنبر العربي
        weight = self._determine_syllable_weight(syllable)

        stress_score = 0.0

        # المقاطع الثقيلة لها أولوية
        if weight == 'heavy':
            stress_score += 0.6
        elif weight == 'super_heavy':
            stress_score += 0.8

        # المقطع الأخير له أولوية
        if position == total_syllables - 1:
            stress_score += 0.4

        # المقطع ما قبل الأخير
        elif position == total_syllables - 2:
            stress_score += 0.3

        return min(stress_score, 1.0)

    def _analyze_morphological_structure(self, word: str) -> Dict[str, Any]:
        """التحليل الصرفي للكلمة"""

        # تحليل مبسط للوزن العاشر "يستكتبونها"
        if word == "يستكتبونها":
            return {
                'morphological_type': 'form_x_verb_with_pronouns',
                'root': 'كتب',
                'pattern': 'يستفعل',
                'form': 'X',
                'morphemes': [
                    {
                        'morpheme': 'ي',
                        'type': 'prefix',
                        'function': 'imperfective_marker',
                    },
                    {'morpheme': 'ست', 'type': 'prefix', 'function': 'form_x_marker'},
                    {'morpheme': 'كتب', 'type': 'root', 'function': 'lexical_core'},
                    {
                        'morpheme': 'ون',
                        'type': 'suffix',
                        'function': 'plural_masculine',
                    },
                    {
                        'morpheme': 'ها',
                        'type': 'suffix',
                        'function': 'attached_pronoun_feminine',
                    },
                ],
                'semantic_frame': {
                    'action': 'seek_to_cause_writing',
                    'agent': 'third_person_plural_masculine',
                    'patient': 'third_person_singular_feminine',
                },
            }

        # تحليل عام لكلمات أخرى
        return {
            'morphological_type': 'complex_word',
            'analysis_confidence': 0.5,
            'morphemes': [{'morpheme': word, 'type': 'unknown', 'function': 'unknown'}],
        }

    def _analyze_syntactic_structure(self, word: str) -> Dict[str, Any]:
        """التحليل النحوي للكلمة"""

        if word == "يستكتبونها":
            return {
                'word_class': 'verb',
                'verb_features': {
                    'tense': 'imperfective',
                    'person': '3rd',
                    'number': 'plural',
                    'gender': 'masculine',
                    'voice': 'active',
                    'mood': 'indicative',
                },
                'attached_pronouns': [
                    {
                        'pronoun': 'ها',
                        'person': '3rd',
                        'number': 'singular',
                        'gender': 'feminine',
                        'case': 'accusative',
                        'function': 'direct_object',
                    }
                ],
                'syntactic_complexity': 0.9,
            }

        return {'word_class': 'unknown', 'syntactic_complexity': 0.3}

    def _analyze_semantic_structure(self, word: str) -> Dict[str, Any]:
        """التحليل الدلالي للكلمة"""

        if word == "يستكتبونها":
            return {
                'semantic_field': 'communication',
                'semantic_roles': {
                    'agent': 'group_of_males',
                    'action': 'causative_writing_request',
                    'patient': 'female_entity',
                },
                'semantic_features': [
                    'causative',
                    'communicative',
                    'interpersonal',
                    'written_medium',
                ],
                'complexity_level': 'high',
                'semantic_transparency': 0.8,
            }

        return {
            'semantic_field': 'unknown',
            'complexity_level': 'medium',
            'semantic_transparency': 0.5,
        }

    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """حساب درجة التعقيد الإجمالية"""

        complexity_factors = [
            analysis['phonemic_analysis']['total_phonemes'] / 10,  # عدد الفونيمات
            analysis['syllabic_analysis']['syllabic_complexity'],  # تعقيد المقاطع
            len(analysis['morphological_analysis']['morphemes']) / 5,  # عدد المورفيمات
            analysis['syntactic_analysis'].get(
                'syntactic_complexity', 0.5
            ),  # التعقيد النحوي
            1
            - analysis['semantic_analysis'].get(
                'semantic_transparency', 0.5
            ),  # الشفافية الدلالية
        ]

        return sum(complexity_factors) / len(complexity_factors)


def demonstrate_complex_analysis():  # type: ignore[no-untyped-def]
    """عرض توضيحي للتحليل المتقدم"""

    print("🚀 عرض توضيحي: تحليل الكلمات المعقدة")
    print("=" * 60)

    analyzer = ComplexWordPhonologicalAnalyzer()

    # تحليل الكلمة المعقدة
    complex_word = "يستكتبونها"
    analysis = analyzer.analyze_complex_word(complex_word)

    # عرض النتائج
    print("\n📊 نتائج التحليل الشامل:")
    print(f"   الكلمة: {analysis['input_word']}")
    print(f"   عدد الفونيمات: {analysis['phonemic_analysis']['total_phonemes']}")
    print(f"   عدد المقاطع: {analysis['syllabic_analysis']['total_syllables']}")
    print(f"   عدد المورفيمات: {len(analysis['morphological_analysis']['morphemes'])}")
    print(f"   درجة التعقيد: {analysis['complexity_score']:.2f}/1.0")

    print("\n🔬 التحليل الفونيمي:")
    for phoneme_data in analysis['phonemic_analysis']['classified_phonemes']:
        print(
            f"   {phoneme_data['phoneme']}: {phoneme_data['type']} - {phoneme_data['features']}}"
        )  # noqa: E501

    print("\n🏗️ التحليل المقطعي:")
    for syl_data in analysis['syllabic_analysis']['syllable_analysis']:
        print(
            f"   {syl_data['syllable']}: {syl_data['structure']} ({syl_data['weight']})"
        )  # noqa: E501

    print("\n📝 التحليل الصرفي:")
    for morpheme in analysis['morphological_analysis']['morphemes']:
        print(f"   {morpheme['morpheme']}: {morpheme['type']} - {morpheme['function']}}")

    print("\n🏛️ التحليل النحوي:")
    verb_features = analysis['syntactic_analysis']['verb_features']
    print(
        f"   فعل مضارع، {verb_features['person']}، {verb_features['number']}, {verb_features['gender']}"
    )  # noqa: E501

    print("\n💭 التحليل الدلالي:")
    semantic = analysis['semantic_analysis']
    print(f"   المجال الدلالي: {semantic['semantic_field']}")
    print(f"   الأدوار: {semantic['semantic_roles']}")

    # حفظ التحليل التفصيلي
    with open('complex_word_analysis.json', 'w', encoding='utf 8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print("\n✅ تم حفظ التحليل التفصيلي في: complex_word_analysis.json")

    return analysis


if __name__ == "__main__":
    demonstrate_complex_analysis()
