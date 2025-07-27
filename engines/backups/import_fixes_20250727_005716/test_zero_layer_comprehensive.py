#!/usr/bin/env python3
"""
Comprehensive Test Suite for Zero Layer Phonology Engine
=======================================================
Tests all 29 Arabic phonemes and 8 vowels with complete coverage
Verifies classification accuracy and feature extraction

Author: Arabic NLP Test Team
Version: 1.0.0
Date: 2025-07 23
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


import sys  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Dict, List, Set, Tuple
from collections import defaultdict  # noqa: F401

# Add the engines directory to the path
sys.path.append(str(Path(__file__).parent))

from zero_layer_phonology import (
    ZeroLayerPhonologyEngine,
    PhonemeClassification,
    HarakaClassification,
)  # noqa: F401
from unified_phonemes import (
    get_unified_phonemes,
    extract_phonemes,
    get_phonetic_features,
    is_emphatic,
)  # noqa: F401


class ComprehensiveZeroLayerTester:
    """Complete test suite for Zero Layer Phonology Engine"""

    def __init__(self):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    self.engine = ZeroLayerPhonologyEngine()

        # 29 Arabic Phonemes (الحروف العربية الـ 29)
    self.all_arabic_phonemes = [
    'ا',
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

        # 8 Arabic Vowels/Harakat (الحركات الثمانية)
    self.all_arabic_vowels = [
    'َ',  # فتحة
    'ِ',  # كسرة
    'ُ',  # ضمة
    'ْ',  # سكون
    'ّ',  # شدة
    'ً',  # تنوين فتح
    'ٌ',  # تنوين ضم
    'ٍ',  # تنوين كسر
    ]

        # Test results storage
    self.test_results = {
    'phoneme_coverage': {},
    'vowel_coverage': {},
    'classification_accuracy': {},
    'feature_extraction': {},
    'combination_tests': {},
    'errors': [],
    }

    def test_all_phonemes_individual(self):  # type: ignore[no-untyped-def]
    """Test each of the 29 Arabic phonemes individually"""
    print("🔤 TESTING ALL 29 ARABIC PHONEMES INDIVIDUALLY")
    print("=" * 60)

        for phoneme in self.all_arabic_phonemes:
    print(f"\n📍 Testing phoneme: {phoneme}")

            try:
                # Test phoneme without harakat
    analysis = self.engine.analyze(phoneme)

                if analysis.units:
    unit = analysis.units[0]
                    classification = unit.phoneme_class

    print(f"   Phoneme: {unit.phoneme}")
    print(f"   Classification: {classification.value}")
    print(f"   Features: {unit.features}")

                    # Store results
    self.test_results['phoneme_coverage'][phoneme] = {
    'detected': True,
    'classification': classification.value,
    'features': unit.features,
    }

                    # Verify classification is not UNKNOWN
                    if classification == PhonemeClassification.UNKNOWN:
    print(f"   ⚠️ WARNING: {phoneme} classified as UNKNOWN")
    self.test_results['errors'].append(
    f"Phoneme {phoneme} classified as UNKNOWN"
    )
                    else:
    print(f"   ✅ SUCCESS: {phoneme} properly classified")

                else:
    print(f"   ❌ ERROR: No units extracted for {phoneme}")
    self.test_results['phoneme_coverage'][phoneme] = {'detected': False}
    self.test_results['errors'].append(
    f"Phoneme {phoneme} not detected"
    )

            except Exception as e:
    print(f"   💥 EXCEPTION: {e}")
    self.test_results['errors'].append(
    f"Exception for phoneme {phoneme}: {e}"
    )

    def test_all_vowels_individual(self):  # type: ignore[no-untyped def]
    """Test each of the 8 Arabic vowels individually"""
    print("\n\n🔊 TESTING ALL 8 ARABIC VOWELS INDIVIDUALLY")
    print("=" * 60)

        # Test with base consonant ب (ba)
    base_consonant = 'ب'

        for vowel in self.all_arabic_vowels:
    print(f"\n📍 Testing vowel: {vowel}")

            try:
                # Create word with consonant + vowel
    test_word = base_consonant + vowel
    analysis = self.engine.analyze(test_word)

                if analysis.units:
    unit = analysis.units[0]
    haraka_class = unit.haraka_class

    print(f"   Test Word: {test_word}")
    print(f"   Haraka: {unit.haraka}")
    print(f"   Haraka Classification: {haraka_class.value}")
    print(f"   Features: {unit.features}")

                    # Store results
    self.test_results['vowel_coverage'][vowel] = {
    'detected': True,
    'classification': haraka_class.value,
    'haraka_extracted': unit.haraka,
    }

                    # Verify haraka was properly extracted
                    if (
    vowel in unit.haraka
    or haraka_class != HarakaClassification.NONE
    ):
    print(
    f"   ✅ SUCCESS: {vowel} properly detected and classified"
    )
                    else:
    print(f"   ⚠️ WARNING: {vowel} not properly extracted")
    self.test_results['errors'].append(
    f"Vowel {vowel} not properly extracted"
    )

                else:
    print(f"   ❌ ERROR: No units extracted for {test_word}")
    self.test_results['vowel_coverage'][vowel] = {'detected': False}
    self.test_results['errors'].append(f"Vowel {vowel} not detected")

            except Exception as e:
    print(f"   💥 EXCEPTION: {e}")
    self.test_results['errors'].append(f"Exception for vowel {vowel}: {e}")

    def test_phoneme_vowel_combinations(self):  # type: ignore[no-untyped def]
    """Test combinations of phonemes with all vowels"""
    print("\n\n🔗 TESTING PHONEME VOWEL COMBINATIONS")
    print("=" * 60)

        # Test subset of phonemes with all vowels (to avoid too much output)
    test_phonemes = ['ب', 'ت', 'ك', 'م', 'ن', 'ر', 'س', 'ل']

    combination_results = defaultdict(dict)

        for phoneme in test_phonemes:
    print(f"\n📍 Testing {phoneme} with all vowels:")

            for vowel in self.all_arabic_vowels:
                try:
    test_word = phoneme + vowel
    analysis = self.engine.analyze(test_word)

                    if analysis.units:
    unit = analysis.units[0]

    result = {
    'phoneme_class': unit.phoneme_class.value,
    'haraka_class': unit.haraka_class.value,
    'haraka_extracted': unit.haraka,
    'features_count': len(unit.features),
    }

    combination_results[phoneme][vowel] = result
    print(
    f"   {test_word}: {unit.phoneme_class.value} + {unit.haraka_class.value} ✅"
    )  # noqa: E501

                    else:
    print(f"   {test_word}: Failed to extract ❌")
    combination_results[phoneme][vowel] = {'status': 'failed'}

                except Exception as e:
    print(f"   {test_word}: Exception {e} 💥")
    combination_results[phoneme][vowel] = {
    'status': 'exception',
    'error': str(e),
    }

    self.test_results['combination_tests'] = dict(combination_results)

    def test_complex_words(self):  # type: ignore[no-untyped-def]
    """Test complex Arabic words with multiple phonemes and vowels"""
    print("\n\n🏗️ TESTING COMPLEX ARABIC WORDS")
    print("=" * 60)

    complex_words = [
    'كِتَابٌ',  # كتاب - book
    'مُدَرِّسَةٌ',  # مدرسة - school
    'طَالِبٌ',  # طالب - student
    'مَكْتَبَةٌ',  # مكتبة - library
    'جَامِعَةٌ',  # جامعة - university
    'مُهَنْدِسٌ',  # مهندس - engineer
    'مُسْتَشْفَى',  # مستشفى - hospital
    'مَطْبَخٌ',  # مطبخ - kitchen
    'تِلْفَازٌ',  # تلفاز - television
    'حَاسُوبٌ',  # حاسوب - computer
    ]

    complex_results = {}

        for word in complex_words:
    print(f"\n📖 Analyzing complex word: {word}")

            try:
    analysis = self.engine.analyze(word)

                # Extract comprehensive statistics
    stats = {
    'total_units': len(analysis.units),
    'root_phonemes': len(analysis.root_phonemes),
    'affixal_phonemes': len(analysis.affixal_phonemes),
    'functional_phonemes': len(analysis.functional_phonemes),
    'harakat_count': analysis.statistics['harakat_count'],
    'confidence': analysis.confidence,
    'phoneme_breakdown': [],
    }

                for unit in analysis.units:
    stats['phoneme_breakdown'].append(
    {
    'phoneme': unit.phoneme,
    'haraka': unit.haraka,
    'phoneme_class': unit.phoneme_class.value,
    'haraka_class': unit.haraka_class.value,
    }
    )

    complex_results[word] = stats

    print(f"   Units: {stats['total_units']}")
    print(
    f"   Root: {stats['root_phonemes']}, Affixal: {stats['affixal_phonemes']}, Functional: {stats['functional_phonemes']}"
    )
    print(f"   Harakat: {stats['harakat_count']}")
    print(f"   Confidence: {stats['confidence']:.2%}")
    print("   Breakdown:", end="")
                for unit_data in stats['phoneme_breakdown']:
    print(f" {unit_data['phoneme']}{unit_data['haraka']}", end="")
    print(" ✅")

            except Exception as e:
    print(f"   💥 EXCEPTION: {e}")
    complex_results[word] = {'status': 'exception', 'error': str(e)}
    self.test_results['errors'].append(
    f"Exception for complex word {word}: {e}"
    )

    self.test_results['complex_words'] = complex_results

    def test_phoneme_features(self):  # type: ignore[no-untyped-def]
    """Test phonological feature extraction for all phonemes"""
    print("\n\n🔬 TESTING PHONOLOGICAL FEATURE EXTRACTION")
    print("=" * 60)

    feature_coverage = defaultdict(list)

        for phoneme in self.all_arabic_phonemes:
            try:
    analysis = self.engine.analyze(phoneme)

                if analysis.units:
    unit = analysis.units[0]
    features = unit.features

                    # Track which phonemes have which features
                    for feature, value in features.items():
                        if value:  # Only track positive features
    feature_coverage[feature].append(phoneme)

                    if features:
    print(f"   {phoneme}: {list(features.keys())} ✅")
                    else:
    print(f"   {phoneme}: No features extracted ⚠️")

            except Exception as e:
    print(f"   {phoneme}: Exception {e} 💥")

        # Print feature summary
    print("\n📊 FEATURE COVERAGE SUMMARY:")
        for feature, phonemes in feature_coverage.items():
    print(f"   {feature}: {len(phonemes)} phonemes - {phonemes}")

    self.test_results['feature_extraction'] = dict(feature_coverage)

    def generate_coverage_report(self):  # type: ignore[no-untyped-def]
    """Generate comprehensive coverage report"""
    print("\n\n📊 COMPREHENSIVE COVERAGE REPORT")
    print("=" * 80)

        # Phoneme coverage
    detected_phonemes = sum(
    1
            for p, data in self.test_results['phoneme_coverage'].items()
            if data.get('detected', False)
    )
    print(
    f"📤 PHONEME COVERAGE: {detected_phonemes}/{len(self.all_arabic_phonemes)} ({detected_phonemes/len(self.all_arabic_phonemes)*100:.1f}%)"
    )

        # Vowel coverage
    detected_vowels = sum(
    1
            for v, data in self.test_results['vowel_coverage'].items()
            if data.get('detected', False)
    )
    print(
    f"🔊 VOWEL COVERAGE: {detected_vowels}/{len(self.all_arabic_vowels)} ({detected_vowels/len(self.all_arabic_vowels)*100:.1f}%)"
    )

        # Classification accuracy
    root_count = sum(
    1
            for p, data in self.test_results['phoneme_coverage'].items()
            if data.get('classification') == 'root'
    )
    affixal_count = sum(
    1
            for p, data in self.test_results['phoneme_coverage'].items()
            if data.get('classification') == 'affixal'
    )
    functional_count = sum(
    1
            for p, data in self.test_results['phoneme_coverage'].items()
            if data.get('classification') == 'functional'
    )
    unknown_count = sum(
    1
            for p, data in self.test_results['phoneme_coverage'].items()
            if data.get('classification') == 'unknown'
    )

    print("🏷️ CLASSIFICATION BREAKDOWN:")
    print(f"   Root: {root_count} phonemes")
    print(f"   Affixal: {affixal_count} phonemes")
    print(f"   Functional: {functional_count} phonemes")
    print(f"   Unknown: {unknown_count} phonemes")

        # Feature extraction
    feature_count = len(self.test_results['feature_extraction'])
    print(f"🔬 FEATURES EXTRACTED: {feature_count} different phonological features")

        # Error summary
    error_count = len(self.test_results['errors'])
    print(f"❌ ERRORS FOUND: {error_count}")

        if self.test_results['errors']:
    print("\n🚨 ERROR DETAILS:")
            for i, error in enumerate(self.test_results['errors'], 1):
    print(f"   {i}. {error}")

        # Overall assessment
    overall_score = (detected_phonemes + detected_vowels) / (
    len(self.all_arabic_phonemes) + len(self.all_arabic_vowels)
    )
    print(f"\n🎯 OVERALL COVERAGE SCORE: {overall_score*100:.1f%}")

        if overall_score >= 0.95:
    print("🎉 EXCELLENT: Near complete coverage achieved!")
        elif overall_score >= 0.85:
    print("✅ GOOD: High coverage with minor gaps")
        elif overall_score >= 0.70:
    print("⚠️ ACCEPTABLE: Moderate coverage, improvements needed")
        else:
    print("❌ POOR: Significant gaps in coverage, major improvements required")

    def run_all_tests(self):  # type: ignore[no-untyped def]
    """Run all comprehensive tests"""
    print("🚀 COMPREHENSIVE ZERO LAYER PHONOLOGY ENGINE TEST")
    print("=" * 80)
    print("Testing complete coverage of 29 Arabic phonemes and 8 vowels")
    print("=" * 80)

        # Run all test suites
    self.test_all_phonemes_individual()
    self.test_all_vowels_individual()
    self.test_phoneme_vowel_combinations()
    self.test_complex_words()
    self.test_phoneme_features()

        # Generate final report
    self.generate_coverage_report()

    return self.test_results


def main():  # type: ignore[no-untyped def]
    """Main test execution"""
    tester = ComprehensiveZeroLayerTester()
    results = tester.run_all_tests()

    print("\n📝 Test Results Summary:")
    print(f"   Phonemes tested: {len(results['phoneme_coverage'])}")
    print(f"   Vowels tested: {len(results['vowel_coverage'])}")
    print(f"   Complex words tested: {len(results.get('complex_words', {}))}")
    print(f"   Features extracted: {len(results['feature_extraction'])}")
    print(f"   Total errors: {len(results['errors'])}")


if __name__ == "__main__":
    main()
