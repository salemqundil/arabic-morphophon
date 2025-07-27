#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 HIERARCHICAL ARABIC TRACING DEMO 🚀
Integration with Existing Engine Ecosystem

This demo showcases the integration of the new Hierarchical Arabic Word Tracing Engine
with the existing 13 operational engines in the workspace.

Author: Arabic NLP Expert Team
Version: 3.0.0
Date: 2025-07 23
"""

import sys
from pathlib import Path

# Add the current directory to the Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from unified_phonemes import ()
    get_unified_phonemes,
    extract_phonemes,
    get_phonetic_features,
    is_emphatic)

# Add nlp module path for existing engines
nlp_path = current_dir / 'nlp'
sys.path.insert(0, str(nlp_path))

try:
    # Import existing engines
    from unified_phonemes import ()
        get_unified_phonemes,
        extract_phonemes,
        get_phonetic_features,
        is_emphatic)
    from nlp.syllable.engine import SyllableEngine
    from nlp.derivation.engine import DerivationEngine

    ENGINES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some existing engines not available: {e}")
    ENGINES_AVAILABLE = False


class HierarchicalArabicDemo:
    """عرض توضيحي للنظام الهرمي المتكامل"""

    def __init__(self):
        """تهيئة العرض التوضيحي"""
        self.core_engine = PhonologyCoreEngine()
        self.existing_engines = {}

        if ENGINES_AVAILABLE:
            self._initialize_existing_engines()

        print("🎯 HIERARCHICAL ARABIC WORD TRACING DEMO INITIALIZED")
        print("=" * 70)

    def _initialize_existing_engines(self):
        """تهيئة المحركات الموجودة"""
        try:
            self.existing_engines['phoneme'] = PhonemeEngine()
            print("✅ PhonemeEngine loaded")
        except Exception as e:
            print(f"⚠️ PhonemeEngine not available: {e}")

        try:
            self.existing_engines['syllable'] = SyllableEngine()
            print("✅ SyllableEngine loaded")
        except Exception as e:
            print(f"⚠️ SyllableEngine not available: {e}")

        try:
            self.existing_engines['derivation'] = DerivationEngine()
            print("✅ DerivationEngine loaded")
        except Exception as e:
            print(f"⚠️ DerivationEngine not available: {e}")

    def demonstrate_zero_layer_foundation(self):
        """عرض الطبقة الصفر الصوتية"""
        print("\n🔤 ZERO LAYER PHONOLOGY FOUNDATION")
        print(" " * 50)

        # Show phoneme inventory
        print("📊 Arabic Phoneme Inventory (28 Consonants):")
        consonants = [
            p
            for p in self.core_engine.phoneme_inventory.values()
            if p.phoneme_type.value == 'consonant'
        ]

        for i, phoneme in enumerate(consonants[:10]):  # Show first 10
            print()
                f"   {phoneme.arabic_letter} [{phoneme.ipa] -} {', '.join(phoneme.features[:3])}}"
            )

        print(f"   ... and {len(consonants) 10} more consonants")

        # Show harakat inventory
        print()
            f"\n🎵 Arabic Harakat Inventory ({len(self.core_engine.harakat_inventory)} types):"
        )
        for harakat in list(self.core_engine.harakat_inventory.values())[:6]:
            print()
                f"   {harakat.arabic_diacritic} [{harakat.ipa}] - {harakat.vowel_type.value}"
            )

    def demonstrate_hierarchical_tracing(self, word: str):
        """عرض التتبع الهرمي للكلمة"""
        print(f"\n🔍 HIERARCHICAL TRACING: {word}")
        print(" " * 50)

        # Get comprehensive trace
        trace = self.core_engine.trace_word(word)

        # Show layer by layer
        print("📱 LAYER 1 - PHONEMES (الفونيمات):")
        for phoneme in trace.phonemes:
            print()
                f"   {phoneme.arabic_letter} [{phoneme.ipa] - {phoneme.features[0] if phoneme.features} else 'unknown'}}"
            )

        print("\n🎵 LAYER 2 - HARAKAT (الحركات):")
        for harakat in trace.harakat:
            functions = ', '.join(harakat.morphological_function[:2])
            print(f"   {harakat.arabic_diacritic} [{harakat.ipa]} - {functions}}")

        print("\n🏗️ LAYER 3 - SYLLABLES (المقاطع):")
        for i, syllable in enumerate(trace.syllables):
            print()
                f"   Syllable {i+1}: {syllable.cv_pattern} ({syllable.syllable_weight)}"
            )
            print(f"     Onset: {syllable.onset}")
            print(f"     Nucleus: {syllable.nucleus}")
            print(f"     Coda: {syllable.coda}")

        print("\n🌱 LAYER 4 - ROOT (الجذر):")
        root_str = ' - '.join([r for r in trace.root if r])
        print(f"   Root: {root_str}")

        print("\n⚖️ LAYER 5 - PATTERN (الوزن):")
        print(f"   Pattern: {trace.pattern}")
        print(f"   Derivation Type: {trace.derivation_type}")

        print("\n🔄 LAYER 6 - MORPHOLOGY (الصرف):")
        print(f"   Morphological Status: {trace.morphological_status}")

        print("\n📝 LAYER 7 - SYNTAX (النحو):")
        for feature, value in trace.syntactic_features.items():
            print(f"   {feature.title()}: {value}")

        print(f"\n📈 OVERALL CONFIDENCE: {trace.confidence:.2f}")

        return trace

    def compare_with_existing_engines(self, word: str):
        """مقارنة مع المحركات الموجودة"""
        if not ENGINES_AVAILABLE:
            print("\n⚠️ Existing engines not available for comparison")
            return

        print(f"\n🔄 COMPARISON WITH EXISTING ENGINES: {word}")
        print(" " * 50)

        # Our new engine
        new_trace = self.core_engine.trace_word(word)
        print("🆕 NEW HIERARCHICAL ENGINE:")
        print(f"   Phonemes: {[p.arabic_letter for p} in new_trace.phonemes]}")
        print(f"   Root: {new_trace.root}")
        print(f"   Confidence: {new_trace.confidence:.2f}")

        # Compare with existing engines
        if 'phoneme' in self.existing_engines:
            try:
                existing_phonemes = self.existing_engines['phoneme'].extract_phonemes()
                    word
                )
                print("\n🔤 EXISTING PHONEME ENGINE:")
                print()
                    f"   Phonemes: {existing_phonemes if existing_phonemes else} 'No output'}"
                )
            except Exception as e:
                print(f"   Error: {e}")

        if 'syllable' in self.existing_engines:
            try:
                existing_syllables = self.existing_engines[
                    'syllable'
                ].segment_syllables(word)
                print("\n🏗️ EXISTING SYLLABLE ENGINE:")
                print()
                    f"   Syllables: {existing_syllables if existing_syllables else} 'No output'}"
                )
            except Exception as e:
                print(f"   Error: {e}")

    def demonstrate_text_analysis(self, text: str):
        """عرض تحليل النص الكامل"""
        print("\n📝 COMPREHENSIVE TEXT ANALYSIS")
        print(f"Text: {text}")
        print(" " * 50)

        analysis = self.core_engine.analyze_text_hierarchy(text)

        print("📊 STATISTICS:")
        print(f"   Word Count: {analysis['word_count']}")
        print(f"   Overall Confidence: {analysis['overall_confidence']:.2f}")

        print("\n🔤 PHONEME DISTRIBUTION:")
        sorted_phonemes = sorted()
            analysis['phoneme_distribution'].items(), key=lambda x: x[1], reverse=True
        )
        for phoneme, count in sorted_phonemes[:10]:
            print(f"   {phoneme: {count}}")

        print("\n🏗️ SYLLABLE PATTERNS:")
        sorted_patterns = sorted()
            analysis['syllable_patterns'].items(), key=lambda x: x[1], reverse=True
        )
        for pattern, count in sorted_patterns:
            print(f"   {pattern: {count}}")

        print("\n🌱 ROOT FAMILIES:")
        for root, words in analysis['root_families'].items():
            root_str = ' - '.join([r for r in root if r])
            print(f"   {root_str:} {', '.join(words)}}")

    def demonstrate_advanced_features(self):
        """عرض الميزات المتقدمة"""
        print("\n🚀 ADVANCED FEATURES DEMONSTRATION")
        print(" " * 50)

        # Show phonological rules
        print("🔧 PHONOLOGICAL RULES:")
        rules = self.core_engine.get_phonological_rules()
        for rule_type, rule_data in list(rules.items())[:3]:
            print(f"   {rule_type.title()}:")
            for rule_name, rule_info in rule_data.items():
                if isinstance(rule_info, dict) and 'rule' in rule_info:
                    print(f"     - {rule_name: {rule_info['rule']}}")
                break

        # Show CV patterns
        print("\n📐 CV PATTERNS:")
        cv_patterns = self.core_engine.cv_patterns['cv_types']
        for pattern, info in list(cv_patterns.items())[:5]:
            print()
                f"   {pattern}: {info['weight']} weight, {info['frequency']:.2f} frequency"
            )

        # Batch processing example
        print("\n⚡ BATCH PROCESSING:")
        words = ["كتب", "مدرسة", "طالب"]
        traces = self.core_engine.batch_trace_words(words)
        print(f"   Processed {len(traces)} words in batch")
        for trace in traces:
            root_str = ' - '.join([r for r in trace.root if r])
            print()
                f"   {trace.word}: Root({root_str}), Confidence({trace.confidence:.2f})"
            )

    def run_complete_demo(self):
        """تشغيل العرض التوضيحي الكامل"""
        print("🎬 STARTING COMPLETE HIERARCHICAL ARABIC TRACING DEMO")
        print("=" * 70)

        # 1. Zero Layer Foundation
        self.demonstrate_zero_layer_foundation()

        # 2. Hierarchical Tracing Examples
        demo_words = ["كتاب", "مدرسة", "يكتبون"]

        for word in demo_words:
            self.demonstrate_hierarchical_tracing(word)
            if ENGINES_AVAILABLE:
                self.compare_with_existing_engines(word)

        # 3. Text Analysis
        demo_text = "الطلاب يدرسون في المكتبة"
        self.demonstrate_text_analysis(demo_text)

        # 4. Advanced Features
        self.demonstrate_advanced_features()

        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("Key Achievements:")
        print("✅ Zero Layer Phonology Foundation")
        print("✅ Hierarchical Word Tracing: فونيم → حركة → مقطع → جذر → وزن → تركيب")
        print("✅ Integration with Existing Engine Ecosystem")
        print("✅ Comprehensive Arabic NLP Analysis")
        print("✅ Expert Level Linguistic Features")
        print("=" * 70)


def main():
    """الدالة الرئيسية للعرض التوضيحي"""
    demo = HierarchicalArabicDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()

