"""
Comprehensive Phoneme Engine - Complete Arabic Processing Pipeline
================================================================

This engine demonstrates the full processing flow of Arabic words through all engines,
showing how harakat affects every step of the analysis pipeline.

Author: AI Arabic Linguistics Expert
Date: 2025-07 23
Version: 1.0.0
"""

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


import json  # noqa: F401
import re  # noqa: F401
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict  # noqa: F401
from enum import Enum  # noqa: F401

# Import our harakat engine
from harakat_engine import ArabicHarakatEngine  # noqa: F401


@dataclass
class PhonemeAnalysis:
    """Complete phoneme analysis result"""

    input_word: str
    has_harakat: bool
    phonetic_transcription: str
    phoneme_breakdown: List[Dict]
    syllable_analysis: List[Dict]
    morphological_analysis: Dict
    derivational_analysis: Dict
    stress_pattern: str
    prosodic_analysis: Dict
    alternative_pronunciations: List[str]
    engine_cascade_results: Dict


class ComprehensivePhonemeEngine:
    """
    Comprehensive Phoneme Engine with Full Pipeline Processing

    This engine provides:
    1. Phoneme analysis with/without harakat
    2. All possible harakat combinations
    3. Cascading effects through all engines
    4. Alternative pronunciations
    5. Complete linguistic analysis
    """

    def __init__(self, data_path: str = "data engine"):  # type: ignore[no-untyped-def]
        """TODO: Add docstring."""
        self.harakat_engine = ArabicHarakatEngine(data_path)
        self.data_path = data_path

        # Load phoneme and engine data
        self.phoneme_inventory = self._load_phoneme_inventory()
        self.syllable_templates = self._load_syllable_templates()
        self.morphology_rules = self._load_morphology_rules()
        self.derivation_patterns = self._load_derivation_patterns()

        # Harakat combinations for prediction
        self.harakat_combinations = self._generate_harakat_combinations()

    def _load_phoneme_inventory(self) -> Dict:
        """Load phoneme inventory from JSON"""
        try:
            with open()
                f"{self.data_path}/phonology/arabic_phonemes.json",
                'r',
                encoding='utf 8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _load_syllable_templates(self) -> Dict:
        """Load syllable templates"""
        try:
            with open()
                f"{self.data_path}/syllable/templates.json", 'r', encoding='utf 8'
            ) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _load_morphology_rules(self) -> Dict:
        """Load morphological rules"""
        try:
            with open()
                f"{self.data_path}/morphology/morphological_rules_corrected.json",
                'r',
                encoding='utf 8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _load_derivation_patterns(self) -> Dict:
        """Load derivation patterns"""
        try:
            with open()
                f"{self.data_path}/derivation/tri_roots.json", 'r', encoding='utf 8'
            ) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _generate_harakat_combinations(self) -> List[str]:
        """Generate common harakat patterns"""
        return [
            "َ",  # fatha
            "ُ",  # damma
            "ِ",  # kasra
            "ْ",  # sukun
            "ّ",  # shadda
            "ً",  # tanwin fath
            "ٌ",  # tanwin dam
            "ٍ",  # tanwin kasr
        ]

    def analyze_phoneme_comprehensive(self, word: str) -> PhonemeAnalysis:
        """
        Comprehensive phoneme analysis showing full pipeline

        Args:
            word: Arabic word (with or without harakat)

        Returns:
            Complete PhonemeAnalysis with all processing results
        """

        # Check if word has harakat
        has_harakat = bool(self.harakat_engine.detect_harakat(word))

        # 1. PHONEME ENGINE - Core Analysis
        phonetic = self.harakat_engine.text_to_phonetic(word)
        phoneme_breakdown = self._analyze_individual_phonemes(word, phonetic)

        # 2. SYLLABLE ENGINE - Driven by phonemes and harakat
        syllables = self.harakat_engine.syllabify_with_harakat(word)
        syllables_with_stress = self.harakat_engine.assign_stress(syllables)

        # 3. MORPHOLOGICAL ENGINE - Pattern recognition
        morphological = self.harakat_engine.analyze_morphological_harakat(word)
        morphological.update(self._advanced_morphological_analysis(word))

        # 4. DERIVATIONAL ENGINE - Root and pattern extraction
        derivational = self._comprehensive_derivational_analysis(word)

        # 5. PROSODIC ENGINE - Weight and meter analysis
        prosodic = self._prosodic_analysis(syllables_with_stress)

        # 6. Generate alternative pronunciations
        alternatives = self._generate_alternative_pronunciations(word)

        # 7. Cascade through all engines
        cascade_results = self._run_engine_cascade(word)

        # Create stress pattern string
        stress_pattern = self._create_stress_pattern(syllables_with_stress)

        return PhonemeAnalysis()
            input_word=word,
            has_harakat=has_harakat,
            phonetic_transcription=phonetic,
            phoneme_breakdown=phoneme_breakdown,
            syllable_analysis=syllables_with_stress,
            morphological_analysis=morphological,
            derivational_analysis=derivational,
            stress_pattern=stress_pattern,
            prosodic_analysis=prosodic,
            alternative_pronunciations=alternatives,
            engine_cascade_results=cascade_results)

    def _analyze_individual_phonemes(self, word: str, phonetic: str) -> List[Dict]:
        """Analyze each phoneme individually"""
        phonemes = []
        harakat_positions = self.harakat_engine.detect_harakat(word)

        # Map each character to its phonetic representation
        char_phoneme_map = self._create_char_phoneme_mapping(word, phonetic)

        for i, char in enumerate(word):
            if char not in "ًٌٍَُِّْٰٓ":  # Skip diacritics in character analysis
                phoneme_info = {
                    "position": i,
                    "orthographic": char,
                    "phonetic": char_phoneme_map.get(i, char),
                    "phoneme_class": self._classify_phoneme(char),
                    "features": self._get_phoneme_features(char),
                    "harakat": None,
                }

                # Check if this character has harakat
                for pos, harakat_char, harakat_info in harakat_positions:
                    if pos == i + 1:  # Harakat follows consonant
                        phoneme_info["harakat"] = {
                            "mark": harakat_char,
                            "type": harakat_info.type.value,
                            "phonetic": harakat_info.phonetic_value,
                            "grammatical_function": harakat_info.grammatical_function,
                        }
                        break

                phonemes.append(phoneme_info)

        return phonemes

    def _create_char_phoneme_mapping(self, word: str, phonetic: str) -> Dict[int, str]:
        """Create mapping between character positions and phonetic representation"""
        # Simplified mapping - real implementation would be more sophisticated
        mapping = {}
        consonant_map = {
            'ب': 'b',
            'ت': 't',
            'ث': 'θ',
            'ج': 'dʒ',
            'ح': 'ħ',
            'خ': 'x',
            'د': 'd',
            'ذ': 'ð',
            'ر': 'r',
            'ز': 'z',
            'س': 's',
            'ش': 'ʃ',
            'ص': 'sˤ',
            'ض': 'dˤ',
            'ط': 'tˤ',
            'ظ': 'ðˤ',
            'ع': 'ʕ',
            'غ': 'ɣ',
            'ف': 'f',
            'ق': 'q',
            'ك': 'k',
            'ل': 'l',
            'م': 'm',
            'ن': 'n',
            'ه': 'h',
            'و': 'w',
            'ي': 'j',
            'ء': 'ʔ',
            'ا': 'aː',
        }

        consonant_pos = 0
        for i, char in enumerate(word):
            if char in consonant_map:
                mapping[i] = consonant_map[char]
                consonant_pos += 1

        return mapping

    def _classify_phoneme(self, char: str) -> str:
        """Classify phoneme by type"""
        consonant_classes = {
            'ب': 'stop',
            'ت': 'stop',
            'ث': 'fricative',
            'ج': 'affricate',
            'ح': 'fricative',
            'خ': 'fricative',
            'د': 'stop',
            'ذ': 'fricative',
            'ر': 'liquid',
            'ز': 'fricative',
            'س': 'fricative',
            'ش': 'fricative',
            'ص': 'fricative',
            'ض': 'stop',
            'ط': 'stop',
            'ظ': 'fricative',
            'ع': 'fricative',
            'غ': 'fricative',
            'ف': 'fricative',
            'ق': 'stop',
            'ك': 'stop',
            'ل': 'liquid',
            'م': 'nasal',
            'ن': 'nasal',
            'ه': 'fricative',
            'و': 'glide',
            'ي': 'glide',
            'ء': 'stop',
            'ا': 'vowel',
        }
        return consonant_classes.get(char, 'unknown')

    def _get_phoneme_features(self, char: str) -> List[str]:
        """Get phonological features of phoneme"""
        feature_map = {
            'ب': ['voiced', 'bilabial', 'stop'],
            'ت': ['voiceless', 'alveolar', 'stop'],
            'ث': ['voiceless', 'dental', 'fricative'],
            'ج': ['voiced', 'postalveolar', 'affricate'],
            'ح': ['voiceless', 'pharyngeal', 'fricative'],
            'خ': ['voiceless', 'uvular', 'fricative'],
            'د': ['voiced', 'alveolar', 'stop'],
            'ذ': ['voiced', 'dental', 'fricative'],
            'ر': ['voiced', 'alveolar', 'trill'],
            'ز': ['voiced', 'alveolar', 'fricative'],
            'س': ['voiceless', 'alveolar', 'fricative'],
            'ش': ['voiceless', 'postalveolar', 'fricative'],
            'ص': ['voiceless', 'alveolar', 'fricative', 'pharyngealized'],
            'ض': ['voiced', 'alveolar', 'stop', 'pharyngealized'],
            'ط': ['voiceless', 'alveolar', 'stop', 'pharyngealized'],
            'ظ': ['voiced', 'dental', 'fricative', 'pharyngealized'],
            'ع': ['voiced', 'pharyngeal', 'fricative'],
            'غ': ['voiced', 'uvular', 'fricative'],
            'ف': ['voiceless', 'labiodental', 'fricative'],
            'ق': ['voiceless', 'uvular', 'stop'],
            'ك': ['voiceless', 'velar', 'stop'],
            'ل': ['voiced', 'alveolar', 'lateral'],
            'م': ['voiced', 'bilabial', 'nasal'],
            'ن': ['voiced', 'alveolar', 'nasal'],
            'ه': ['voiceless', 'glottal', 'fricative'],
            'و': ['voiced', 'labiovelar', 'approximant'],
            'ي': ['voiced', 'palatal', 'approximant'],
            'ء': ['voiceless', 'glottal', 'stop'],
            'ا': ['low', 'central', 'vowel', 'long'],
        }
        return feature_map.get(char, ['unknown'])

    def _advanced_morphological_analysis(self, word: str) -> Dict:
        """Advanced morphological analysis"""
        analysis = {
            "word_type": self._determine_word_type(word),
            "grammatical_category": self._determine_grammatical_category(word),
            "inflectional_features": self._extract_inflectional_features(word),
            "derivational_features": self._extract_derivational_features(word),
        }
        return analysis

    def _comprehensive_derivational_analysis(self, word: str) -> Dict:
        """Comprehensive derivational analysis"""
        self.harakat_engine.strip_harakat(word)

        analysis = {
            "root_extraction": self._extract_root_comprehensive(word),
            "pattern_identification": self._identify_pattern_comprehensive(word),
            "morphological_form": self._identify_morphological_form(word),
            "semantic_analysis": self._analyze_semantics(word),
        }
        return analysis

    def _prosodic_analysis(self, syllables: List[Dict]) -> Dict:
        """Comprehensive prosodic analysis"""
        weights = [s["weight"] for s in syllables]
        mora_total = sum(s["mora_count"] for s in syllables)

        return {
            "syllable_count": len(syllables),
            "mora_count": mora_total,
            "weight_pattern": weights,
            "rhythmic_pattern": self._create_rhythmic_pattern(syllables),
            "metrical_feet": self._identify_metrical_feet(syllables),
            "prosodic_word_structure": self._analyze_prosodic_word(syllables),
        }

    def _generate_alternative_pronunciations(self, word: str) -> List[str]:
        """Generate alternative pronunciations with different harakat"""
        alternatives = []

        if self.harakat_engine.detect_harakat(word):
            # Word has harakat - generate without
            unvocalized = self.harakat_engine.strip_harakat(word)
            alternatives.append()
                f"Unvocalized: {unvocalized} → /{self.harakat_engine.text_to_phonetic(unvocalized)}/"
            )
        else:
            # Word has no harakat - generate common patterns
            alternatives.extend(self._predict_harakat_variants(word))

        return alternatives

    def _predict_harakat_variants(self, unvocalized_word: str) -> List[str]:
        """Predict possible harakat variants for unvocalized word"""
        variants = []

        if len(unvocalized_word) == 3:  # Triliteral root patterns
            # Form I perfect: CaCaCa
            variant1 = ()
                f"{unvocalized_word[0]}َ{unvocalized_word[1]}َ{unvocalized_word[2]}َ"
            )
            variants.append()
                f"Form I perfect: {variant1} → /{self.harakat_engine.text_to_phonetic(variant1)}/"
            )

            # Form I imperfect: yaCCuCu
            variant2 = ()
                f"يَ{unvocalized_word[0]}ْ{unvocalized_word[1]}ُ{unvocalized_word[2]}ُ"
            )
            variants.append()
                f"Form I imperfect: {variant2} → /{self.harakat_engine.text_to_phonetic(variant2)}/"
            )

            # Active participle: CaaCiC
            variant3 = ()
                f"{unvocalized_word[0]}َا{unvocalized_word[1]ِ{unvocalized_word[2]}}"
            )
            variants.append()
                f"Active participle: {variant3} → /{self.harakat_engine.text_to_phonetic(variant3)}/"
            )

            # Passive participle: maCCuuC
            variant4 = ()
                f"مَ{unvocalized_word[0]}ْ{unvocalized_word[1]ُو{unvocalized_word[2]}}"
            )
            variants.append()
                f"Passive participle: {variant4} → /{self.harakat_engine.text_to_phonetic(variant4)}/"
            )

        return variants

    def _run_engine_cascade(self, word: str) -> Dict:
        """Run word through all engines showing cascade effects"""
        cascade = {}

        # Original word analysis
        cascade["original"] = {
            "input": word,
            "has_harakat": bool(self.harakat_engine.detect_harakat(word)),
            "phonetic": self.harakat_engine.text_to_phonetic(word),
        }

        # Strip harakat and analyze
        unvocalized = self.harakat_engine.strip_harakat(word)
        cascade["unvocalized"] = {
            "input": unvocalized,
            "phonetic": self.harakat_engine.text_to_phonetic(unvocalized),
            "syllables": len(self.harakat_engine.syllabify_with_harakat(unvocalized)),
            "predicted_variants": self._predict_harakat_variants(unvocalized)[
                :2
            ],  # Limit to 2
        }

        # Morphological cascade
        cascade["morphological_cascade"] = {
            "with_harakat": self.harakat_engine.analyze_morphological_harakat(word),
            "without_harakat": self.harakat_engine.analyze_morphological_harakat()
                unvocalized
            ),
        }

        # Syllable cascade
        syllables_with = self.harakat_engine.syllabify_with_harakat(word)
        syllables_without = self.harakat_engine.syllabify_with_harakat(unvocalized)

        cascade["syllable_cascade"] = {
            "with_harakat": {
                "count": len(syllables_with),
                "weights": [s["weight"] for s in syllables_with],
                "stress": self._create_stress_pattern()
                    self.harakat_engine.assign_stress(syllables_with)
                ),
            },
            "without_harakat": {
                "count": len(syllables_without),
                "weights": [s["weight"] for s in syllables_without],
                "stress": self._create_stress_pattern()
                    self.harakat_engine.assign_stress(syllables_without)
                ),
            },
        }

        return cascade

    def _create_stress_pattern(self, syllables: List[Dict]) -> str:
        """Create stress pattern string"""
        pattern = ""
        for i, syll in enumerate(syllables):
            if syll["stress"]:
                pattern += "ˈ"
            pattern += syll["orthographic"]
            if i < len(syllables) - 1:
                pattern += "."
        return pattern

    def _create_rhythmic_pattern(self, syllables: List[Dict]) -> str:
        """Create rhythmic pattern using musical notation"""
        pattern = ""
        for syll in syllables:
            if syll["weight"] == "light":
                pattern += "♪"  # eighth note
            elif syll["weight"] == "heavy":
                pattern += "♩"  # quarter note
            else:  # superheavy
                pattern += "𝅗𝅥"  # half note
        return pattern

    def _identify_metrical_feet(self, syllables: List[Dict]) -> List[str]:
        """Identify metrical feet in the word"""
        feet = []
        weights = [s["weight"] for s in syllables]

        i = 0
        while i < len(weights):
            if i + 1 < len(weights):
                # Check for common Arabic feet
                if weights[i] == "light" and weights[i + 1] == "heavy":
                    feet.append("iamb")
                    i += 2
                elif weights[i] == "heavy" and weights[i + 1] == "light":
                    feet.append("trochee")
                    i += 2
                else:
                    feet.append("single")
                    i += 1
            else:
                feet.append("single")
                i += 1

        return feet

    def _analyze_prosodic_word(self, syllables: List[Dict]) -> Dict:
        """Analyze prosodic word structure"""
        return {
            "minimal_word": len(syllables) >= 2,  # Arabic minimal word is bimoraic
            "stress_type": "weight_sensitive",
            "foot_structure": self._identify_metrical_feet(syllables),
            "prominence": [s["stress"] for s in syllables],
        }

    # Helper methods for morphological analysis
    def _determine_word_type(self, word: str) -> str:
        """Determine basic word type"""
        # Simplified - real implementation would be more complex
        if any(char in word for char in "يتنأس"):  # Common verbal prefixes
            return "verb"
        elif word.endswith("ة"):  # Feminine marker
            return "noun"
        else:
            return "noun"  # Default

    def _determine_grammatical_category(self, word: str) -> str:
        """Determine grammatical category"""
        # Simplified analysis
        harakat_list = self.harakat_engine.detect_harakat(word)
        if harakat_list and harakat_list[ 1][2].type.value in [
            "tanwin_fath",
            "tanwin_dam",
            "tanwin_kasr",
        ]:
            return "indefinite_noun"
        return "definite_noun"

    def _extract_inflectional_features(self, word: str) -> Dict:
        """Extract inflectional features"""
        features = {}
        harakat_list = self.harakat_engine.detect_harakat(word)

        if harakat_list:
            final_harakat = harakat_list[ 1][2]
            if final_harakat.type.value == "fatha":
                features["case"] = "accusative"
            elif final_harakat.type.value == "damma":
                features["case"] = "nominative"
            elif final_harakat.type.value == "kasra":
                features["case"] = "genitive"

        return features

    def _extract_derivational_features(self, word: str) -> Dict:
        """Extract derivational features"""
        features = {}
        unvocalized = self.harakat_engine.strip_harakat(word)

        if len(unvocalized) >= 3:
            features["root_type"] = "triliteral"
            features["pattern_type"] = "basic"

        return features

    def _extract_root_comprehensive(self, word: str) -> Dict:
        """Comprehensive root extraction"""
        unvocalized = self.harakat_engine.strip_harakat(word)

        if len(unvocalized) >= 3:
            return {
                "root": unvocalized[:3],
                "root_type": "triliteral",
                "extraction_confidence": 0.8,
            }

        return {
            "root": unvocalized,
            "root_type": "unknown",
            "extraction_confidence": 0.3,
        }

    def _identify_pattern_comprehensive(self, word: str) -> Dict:
        """Comprehensive pattern identification"""
        # Simplified pattern matching
        if len(self.harakat_engine.strip_harakat(word)) == 3:
            harakat_sequence = [
                h[2].type.value for h in self.harakat_engine.detect_harakat(word)
            ]
            if harakat_sequence == ["fatha", "fatha"]:
                return {"pattern": "CaCaC", "form": "Form_I_perfect"}

        return {"pattern": "unknown", "form": "unknown"}

    def _identify_morphological_form(self, word: str) -> str:
        """Identify morphological form"""
        # Simplified form identification
        return "Form_I"  # Default

    def _analyze_semantics(self, word: str) -> Dict:
        """Basic semantic analysis"""
        return {"semantic_field": "general", "lexical_category": "concrete"}

    def demonstrate_complete_pipeline(self, word: str) -> None:
        """
        Demonstrate complete processing pipeline

        Args:
            word: Arabic word to process
        """
        print(f"\\n{'='*80}")
        print(f"COMPREHENSIVE PHONEME ENGINE ANALYSIS: {word}")
        print(f"{'='*80\\n}")

        # Run comprehensive analysis
        analysis = self.analyze_phoneme_comprehensive(word)

        # Display results
        print("🔤 PHONEME ANALYSIS")
        print(" " * 50)
        print(f"Input Word: {analysis.input_word}")
        print(f"Has Harakat: {analysis.has_harakat}")
        print(f"IPA Transcription: /{analysis.phonetic_transcription}/")

        print("\\n📝 PHONEME BREAKDOWN")
        print(" " * 50)
        for phoneme in analysis.phoneme_breakdown:
            harakat_info = ()
                f" + {phoneme['harakat']['mark']} ({phoneme['harakat']['type']})"
                if phoneme['harakat']
                else ""
            )
            print()
                f"  {phoneme['orthographic']} → /{phoneme['phonetic']}/ [{phoneme['phoneme_class']]{harakat_info}}"
            )  # noqa: E501

        print("\\n🔊 SYLLABLE ANALYSIS")
        print(" " * 50)
        for i, syll in enumerate(analysis.syllable_analysis):
            stress_mark = "ˈ" if syll["stress"] else ""
            print()
                f"  {i+1}. {stress_mark}{syll['orthographic']} → /{syll['phonetic']}/ ({syll['weight']}, {syll['mora_count'] mora)}"
            )
        print(f"Stress Pattern: {analysis.stress_pattern}")

        print("\\n📚 MORPHOLOGICAL ANALYSIS")
        print(" " * 50)
        for key, value in analysis.morphological_analysis.items():
            if value and key != 'vowel_patterns':  # Skip complex nested data
                print(f"  {key: {value}}")

        print("\\n🌱 DERIVATIONAL ANALYSIS")
        print(" " * 50)
        for key, value in analysis.derivational_analysis.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey: {subvalue}}")
            else:
                print(f"  {key}: {value}")

        print("\\n🎵 PROSODIC ANALYSIS")
        print(" " * 50)
        prosodic = analysis.prosodic_analysis
        print(f"  Syllable Count: {prosodic['syllable_count']}")
        print(f"  Mora Count: {prosodic['mora_count']}")
        print(f"  Weight Pattern: {'} → '.join(prosodic['weight_pattern'])}")
        print(f"  Rhythmic Pattern: {prosodic['rhythmic_pattern']}")
        print(f"  Metrical Feet: {'} + '.join(prosodic['metrical_feet'])}")

        print("\\n🔄 ALTERNATIVE PRONUNCIATIONS")
        print(" " * 50)
        for alt in analysis.alternative_pronunciations:
            print(f"  {alt}")

        print("\\n⚡ ENGINE CASCADE EFFECTS")
        print(" " * 50)
        cascade = analysis.engine_cascade_results

        print("Original vs Unvocalized:")
        print()
            f"  With harakat: {cascade['original']['input']} → /{cascade['original']['phonetic']}/"
        )  # noqa: E501
        print()
            f"  Without harakat: {cascade['unvocalized']['input']} → /{cascade['unvocalized']['phonetic']}/"
        )  # noqa: E501

        print("\\nSyllable Cascade:")
        with_syll = cascade['syllable_cascade']['with_harakat']
        without_syll = cascade['syllable_cascade']['without_harakat']
        print()
            f"  With harakat: {with_syll['count'] syllables,} weights: {with_syll['weights']}}"
        )  # noqa: E501
        print()
            f"  Without harakat: {without_syll['count'] syllables,} weights: {without_syll['weights']}}"
        )  # noqa: E501
        print(f"  Stress change: {with_syll['stress']} → {without_syll['stress']}")

        if cascade['unvocalized']['predicted_variants']:
            print("\\nPredicted Variants:")
            for variant in cascade['unvocalized']['predicted_variants']:
                print(f"  {variant}")


def main():  # type: ignore[no-untyped def]
    """Main demonstration function"""
    engine = ComprehensivePhonemeEngine("data engine")

    # Test words for comprehensive analysis
    test_words = [
        "كتب",  # Unvocalized
        "كَتَبَ",  # With harakat - he wrote
        "كِتَاب",  # Book
        "مُدَرِّس",  # Teacher
        "مدرسة",  # School (partially vocalized)
    ]

    print("🚀 COMPREHENSIVE PHONEME ENGINE DEMONSTRATION")
    print("=" * 80)
    print("This demonstrates how phoneme processing affects ALL Arabic NLP engines")
    print("=" * 80)

    for word in test_words:
        engine.demonstrate_complete_pipeline(word)
        print("\\n" + "🔄" * 40 + "\\n")


if __name__ == "__main__":
    main()

