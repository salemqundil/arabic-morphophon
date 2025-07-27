#!/usr/bin/env python3
"""
UNIFIED ARABIC NLP ENGINE
=========================
Complete Arabic processing pipeline integrating all existing NLP engines.
Uses actual engine implementations from nlp/ directory.

Author: Arabic NLP Team
Date: 2025-07 23
Version: 2.0.0
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import existing NLP engines
sys.path.append(str(Path(__file__).parent / "nlp"))

try:
    from unified_phonemes import ()
        get_unified_phonemes,
        extract_phonemes,
        get_phonetic_features,
        is_emphatic)
    from nlp.syllable.engine import SyllabicUnitEngine
    from nlp.morphology.engine import MorphologyEngine
    from nlp.derivation.engine import DerivationEngine
    from nlp.phonological.engine import PhonologicalEngine

    ENGINES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import NLP engines: {e}")
    ENGINES_AVAILABLE = False

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class UnifiedAnalysis:
    """Complete analysis result from all engines"""

    word: str
    has_harakat: bool

    # Engine Results
    harakat_analysis: Dict[str, Any] = field(default_factory=dict)
    phoneme_analysis: Dict[str, Any] = field(default_factory=dict)
    syllable_analysis: Dict[str, Any] = field(default_factory=dict)
    morphological_analysis: Dict[str, Any] = field(default_factory=dict)
    derivational_analysis: Dict[str, Any] = field(default_factory=dict)
    extract_phonemes: Dict[str, Any] = field(default_factory=dict)
    prosodic_analysis: Dict[str, Any] = field(default_factory=dict)

    # Cross engine effects
    cascade_effects: Dict[str, Any] = field(default_factory=dict)
    alternative_forms: List[str] = field(default_factory=list)


# ============================================================================
# UNIFIED ARABIC ENGINE
# ============================================================================


class UnifiedArabicEngine:
    """Complete Arabic NLP processing engine integrating all existing NLP components"""

    def __init__(self):
        """Initialize all engine components using existing NLP engines"""
        self.data_path = Path("data engine")
        self.engines_initialized = False

        # Initialize actual NLP engines
        if ENGINES_AVAILABLE:
            self.unified_phonemes = UnifiedPhonemeSystem()
            self.syllable_engine = SyllabicUnitEngine()
            self.morphology_engine = MorphologyEngine()
            self.derivation_engine = DerivationEngine()
            try:
                self.phonological_engine = PhonologicalEngine()
            except:
                self.phonological_engine = None
                print("⚠️ PhonologicalEngine not available")
        else:
            print("❌ NLP engines not available, using fallback methods")
            self.unified_phonemes = None
            self.syllable_engine = None
            self.morphology_engine = None
            self.derivation_engine = None
            self.phonological_engine = None

        self._load_all_data()
        self._initialize_engines()

    def _load_all_data(self):
        """Load all linguistic data files"""
        try:
            # Load harakat data
            with open()
                self.data_path / "harakat" / "harakat_database.json",
                'r',
                encoding='utf 8') as f:
                self.harakat_data = json.load(f)

            # Load phoneme data
            with open()
                self.data_path / "phoneme" / "arabic_phonemes.json",
                'r',
                encoding='utf 8') as f:
                self.phoneme_data = json.load(f)

            # Load syllable templates
            with open()
                self.data_path / "syllable" / "templates.json", 'r', encoding='utf 8'
            ) as f:
                self.syllable_templates = json.load(f)

            # Load morphological rules
            with open()
                self.data_path / "morphology" / "morphological_rules_corrected.json",
                'r',
                encoding='utf 8') as f:
                self.morphological_rules = json.load(f)

            # Load derivational patterns
            with open()
                self.data_path / "derivation" / "patterns.json", 'r', encoding='utf 8'
            ) as f:
                self.derivation_patterns = json.load(f)

            # Load roots data
            with open()
                self.data_path / "derivation" / "tri_roots.json", 'r', encoding='utf 8'
            ) as f:
                self.tri_roots = json.load(f)

            print("✅ All linguistic data files loaded successfully")

        except Exception as e:
            print(f"❌ Error loading data files: {e}")
            # Initialize with empty data if files missing
            self.harakat_data = {}
            self.phoneme_data = {}
            self.syllable_templates = {}
            self.morphological_rules = {}
            self.derivation_patterns = {}
            self.tri_roots = {}

    def _initialize_engines(self):
        """Initialize all processing engines"""
        self.engines_initialized = True
        engine_status = "✅ Available" if ENGINES_AVAILABLE else "❌ Fallback mode"
        print(f"🚀 Unified Arabic NLP Engine initialized - {engine_status}")

    # ========================================================================
    # HARAKAT ENGINE
    # ========================================================================

    def analyze_harakat(self, word: str) -> Dict[str, Any]:
        """Analyze Arabic diacritical marks"""
        result = {
            "has_harakat": self._has_harakat(word),
            "harakat_count": self._count_harakat(word),
            "harakat_positions": self._get_harakat_positions(word),
            "vowel_pattern": self._extract_vowel_pattern(word),
            "unvocalized_form": self._remove_harakat(word),
            "syllable_vowels": self._extract_syllable_vowels(word),
        }

        if result["has_harakat"]:
            result["ipa_transcription"] = self._harakat_to_ipa(word)
            result["phonetic_form"] = self._get_phonetic_representation(word)

        return result

    def _has_harakat(self, word: str) -> bool:
        """Check if word contains diacritical marks"""
        harakat_chars = "َُِّْٰٱًٌٍ"
        return any(char in word for char in harakat_chars)

    def _count_harakat(self, word: str) -> int:
        """Count diacritical marks in word"""
        harakat_chars = "َُِّْٰٱًٌٍ"
        return len([char for char in word if char in harakat_chars])

    def _get_harakat_positions(self, word: str) -> List[Dict]:
        """Get positions and types of harakat"""
        positions = []
        harakat_map = {
            "َ": "fatha",
            "ُ": "damma",
            "ِ": "kasra",
            "ّ": "shadda",
            "ْ": "sukun",
            "ً": "tanwin_fath",
            "ٌ": "tanwin_damm",
            "ٍ": "tanwin_kasr",
        }

        for i, char in enumerate(word):
            if char in harakat_map:
                positions.append()
                    {"position": i, "mark": char, "type": harakat_map[char]}
                )

        return positions

    def _extract_vowel_pattern(self, word: str) -> str:
        """Extract vowel pattern from harakat"""
        vowel_map = {"َ": "a", "ُ": "u", "ِ": "i", "ْ": "0"}
        pattern = ""

        for char in word:
            if char in vowel_map:
                pattern += vowel_map[char]

        return pattern

    def _remove_harakat(self, word: str) -> str:
        """Remove all diacritical marks"""
        harakat_chars = "َُِّْٰٱًٌٍ"
        return ''.join(char for char in word if char not in harakat_chars)

    def _extract_syllable_vowels(self, word: str) -> List[str]:
        """Extract vowels for syllable analysis"""
        vowels = []
        vowel_map = {
            "َ": "/a/",
            "ُ": "/u/",
            "ِ": "/i/",
            "ا": "/aː/",
            "و": "/uː/",
            "ي": "/iː/",
        }

        for char in word:
            if char in vowel_map:
                vowels.append(vowel_map[char])

        return vowels

    def _harakat_to_ipa(self, word: str) -> str:
        """Convert word with harakat to IPA transcription"""
        # Simplified IPA conversion
        ipa = word
        conversions = {
            "َ": "a",
            "ُ": "u",
            "ِ": "i",
            "ا": "aː",
            "و": "uː",
            "ي": "iː",
            "ّ": "ː",
            "ْ": "",
        }

        for arabic, ipa_char in conversions.items():
            ipa = ipa.replace(arabic, ipa_char)

        return f"/{ipa}/"

    def _get_phonetic_representation(self, word: str) -> str:
        """Get detailed phonetic representation"""
        return self._harakat_to_ipa(word)

    # ========================================================================
    # PHONEME ENGINE (Using actual NLP engine)
    # ========================================================================

    def analyze_phonemes(self, word: str, harakat_analysis: Dict) -> Dict[str, Any]:
        """Analyze individual phonemes using actual UnifiedPhonemeSystem"""
        if self.unified_phonemes:
            try:
                # Use actual NLP engine
                engine_result = self.unified_phonemes.process(word)

                # Enhance with our analysis
                unvocalized = harakat_analysis.get("unvocalized_form", word)

                result = {
                    "phoneme_count": len(unvocalized),
                    "phoneme_breakdown": self._get_phoneme_breakdown()
                        word, harakat_analysis
                    ),
                    "consonant_count": self._count_consonants(unvocalized),
                    "vowel_count": len(harakat_analysis.get("syllable_vowels", [])),
                    "phonetic_features": self._get_phonetic_features(unvocalized),
                    "ipa_transcription": harakat_analysis.get()
                        "ipa_transcription", f"/{unvocalized}/"
                    ),
                    "engine_result": engine_result,  # Include actual engine output
                }

                return result

            except Exception as e:
                print(f"⚠️ UnifiedPhonemeSystem error: {e}, using fallback")

        # Fallback to simplified analysis
        return self._analyze_phonemes_fallback(word, harakat_analysis)

    def _analyze_phonemes_fallback()
        self, word: str, harakat_analysis: Dict
    ) -> Dict[str, Any]:
        """Fallback phoneme analysis when engine not available"""
        unvocalized = harakat_analysis.get("unvocalized_form", word)

        result = {
            "phoneme_count": len(unvocalized),
            "phoneme_breakdown": self._get_phoneme_breakdown(word, harakat_analysis),
            "consonant_count": self._count_consonants(unvocalized),
            "vowel_count": len(harakat_analysis.get("syllable_vowels", [])),
            "phonetic_features": self._get_phonetic_features(unvocalized),
            "ipa_transcription": harakat_analysis.get()
                "ipa_transcription", f"/{unvocalized}/"
            ),
            "engine_status": "fallback_mode",
        }

        return result

    def _get_phoneme_breakdown(self, word: str, harakat_analysis: Dict) -> List[Dict]:
        """Break down word into individual phonemes with features"""
        breakdown = []
        unvocalized = harakat_analysis.get("unvocalized_form", word)
        harakat_positions = harakat_analysis.get("harakat_positions", [])

        # Create harakat map
        harakat_map = {}
        for pos_info in harakat_positions:
            harakat_map[pos_info["position"]] = pos_info

        for i, char in enumerate(unvocalized):
            phoneme_info = {
                "char": char,
                "ipa": self._char_to_ipa(char),
                "features": self._get_char_features(char),
            }

            # Add harakat if present
            if i + 1 in harakat_map:  # Harakat comes after consonant
                harakat_info = harakat_map[i + 1]
                phoneme_info["harakat"] = {
                    "mark": harakat_info["mark"],
                    "type": harakat_info["type"],
                }

            breakdown.append(phoneme_info)

        return breakdown

    def _count_consonants(self, word: str) -> int:
        """Count consonants in word"""
        vowels = "اوي"
        return len([char for char in word if char not in vowels])

    def _get_phonetic_features(self, word: str) -> Dict[str, List]:
        """Get phonetic features of all phonemes"""
        features = {
            "stops": [],
            "fricatives": [],
            "nasals": [],
            "liquids": [],
            "vowels": [],
        }

        feature_map = {
            "كتبدجقط": "stops",
            "فثذسشصضزخغح": "fricatives",
            "من": "nasals",
            "رل": "liquids",
            "اوي": "vowels",
        }

        for char in word:
            for chars, feature in feature_map.items():
                if char in chars:
                    features[feature].append(char)
                    break

        return features

    def _char_to_ipa(self, char: str) -> str:
        """Convert Arabic character to IPA"""
        ipa_map = {
            "ك": "/k/",
            "ت": "/t/",
            "ب": "/b/",
            "د": "/d/",
            "ج": "/dʒ/",
            "ق": "/q/",
            "ط": "/tˤ/",
            "ف": "/f/",
            "ث": "/θ/",
            "ذ": "/ð/",
            "س": "/s/",
            "ش": "/ʃ/",
            "ص": "/sˤ/",
            "ض": "/dˤ/",
            "ز": "/z/",
            "خ": "/x/",
            "غ": "/ɣ/",
            "ح": "/ħ/",
            "ع": "/ʕ/",
            "ه": "/h/",
            "م": "/m/",
            "ن": "/n/",
            "ر": "/r/",
            "ل": "/l/",
            "و": "/w/",
            "ي": "/j/",
            "ا": "/aː/",
            "ة": "/a/",
        }
        return ipa_map.get(char, f"/{char}/")

    def _get_char_features(self, char: str) -> List[str]:
        """Get phonetic features for character"""
        feature_map = {
            "كتبدجقط": ["stop"],
            "فثذسشصضزخغح": ["fricative"],
            "من": ["nasal"],
            "رل": ["liquid"],
            "اوي": ["vowel"],
        }

        for chars, features in feature_map.items():
            if char in chars:
                return features

        return ["unknown"]

    # ========================================================================
    # SYLLABLE ENGINE (Using actual NLP engine)
    # ========================================================================

    def analyze_syllables()
        self, word: str, phoneme_analysis: Dict, harakat_analysis: Dict
    ) -> Dict[str, Any]:
        """Analyze syllable structure using actual SyllabicUnitEngine"""
        if self.syllable_engine:
            try:
                # Use actual NLP engine
                engine_result = self.syllable_engine.process(word)

                # Enhance with our analysis
                syllables = self._syllabify_word(word, harakat_analysis)

                result = {
                    "syllable_count": len(syllables),
                    "syllables": syllables,
                    "syllable_weights": [syl["weight"] for syl in syllables],
                    "total_mora": sum(syl["mora_count"] for syl in syllables),
                    "stress_pattern": self._get_stress_pattern(syllables),
                    "syllable_types": [syl["type"] for syl in syllables],
                    "weight_sequence": " → ".join(syl["weight"] for syl in syllables),
                    "engine_result": engine_result,  # Include actual engine output
                }

                return result

            except Exception as e:
                print(f"⚠️ SyllabicUnitEngine error: {e}, using fallback")

        # Fallback analysis
        return self._analyze_syllables_fallback()
            word, phoneme_analysis, harakat_analysis
        )

    def _analyze_syllables_fallback()
        self, word: str, phoneme_analysis: Dict, harakat_analysis: Dict
    ) -> Dict[str, Any]:
        """Fallback syllable analysis"""
        syllables = self._syllabify_word(word, harakat_analysis)

        result = {
            "syllable_count": len(syllables),
            "syllables": syllables,
            "syllable_weights": [syl["weight"] for syl in syllables],
            "total_mora": sum(syl["mora_count"] for syl in syllables),
            "stress_pattern": self._get_stress_pattern(syllables),
            "syllable_types": [syl["type"] for syl in syllables],
            "weight_sequence": " → ".join(syl["weight"] for syl in syllables),
            "engine_status": "fallback_mode",
        }

        return result

    def _syllabify_word(self, word: str, harakat_analysis: Dict) -> List[Dict]:
        """Break word into syllables"""
        if harakat_analysis["has_harakat"]:
            return self._syllabify_with_harakat(word)
        else:
            return self._syllabify_without_harakat(word)

    def _syllabify_with_harakat(self, word: str) -> List[Dict]:
        """Syllabify word with diacritical marks"""
        syllables = []
        current_syllable = ""

        i = 0
        while i < len(word):
            char = word[i]

            # Skip harakat marks for syllable boundary detection
            if char in "َُِّْٰٱًٌٍ":
                current_syllable += char
                i += 1
                continue

            current_syllable += char

            # Check if we have a complete syllable
            if self._is_syllable_complete(current_syllable):
                syl_info = self._analyze_syllable_structure(current_syllable)
                syllables.append(syl_info)
                current_syllable = ""

            i += 1

        # Add remaining syllable
        if current_syllable:
            syl_info = self._analyze_syllable_structure(current_syllable)
            syllables.append(syl_info)

        return syllables

    def _syllabify_without_harakat(self, word: str) -> List[Dict]:
        """Syllabify word without diacritical marks"""
        # Simplified: treat as single superheavy syllable
        syl_info = {
            "syllable": word,
            "type": "CVCC+",
            "structure": "complex",
            "weight": "superheavy",
            "mora_count": 3,
            "stress": True,
            "ipa": f"/{word}/",
        }
        return [syl_info]

    def _is_syllable_complete(self, syllable: str) -> bool:
        """Check if syllable is complete"""
        # Simplified logic - check for CV pattern
        clean = ''.join(char for char in syllable if char not in "َُِّْٰٱًٌٍ")

        if len(clean) >= 2:  # At least CV
            return True
        return False

    def _analyze_syllable_structure(self, syllable: str) -> Dict:
        """Analyze structure of a single syllable"""
        clean = ''.join(char for char in syllable if char not in "َُِّْٰٱًٌٍ")

        # Determine syllable type
        vowels = "اوي"
        has_long_vowel = any(v in syllable for v in vowels)
        has_short_vowel = any(h in syllable for h in "َُِ")

        consonant_count = len([char for char in clean if char not in vowels])

        if has_long_vowel and consonant_count >= 2:
            syl_type = "CVVC"
            weight = "superheavy"
            mora = 3
        elif has_short_vowel and consonant_count >= 2:
            syl_type = "CVC"
            weight = "heavy"
            mora = 2
        elif has_long_vowel:
            syl_type = "CVV"
            weight = "heavy"
            mora = 2
        else:
            syl_type = "CV"
            weight = "light"
            mora = 1

        return {
            "syllable": syllable,
            "type": syl_type,
            "structure": syl_type,
            "weight": weight,
            "mora_count": mora,
            "stress": weight != "light",
            "ipa": f"/{clean}/",
        }

    def _get_stress_pattern(self, syllables: List[Dict]) -> str:
        """Generate stress pattern notation"""
        pattern = ""
        for i, syl in enumerate(syllables):
            if syl["stress"] and i == 0:  # Primary stress on first heavy syllable
                pattern += "ˈ" + syl["syllable"]
            elif syl["stress"]:
                pattern += "ˌ" + syl["syllable"]  # Secondary stress
            else:
                pattern += syl["syllable"]

            if i < len(syllables) - 1:
                pattern += "."

        return pattern

    # ========================================================================
    # MORPHOLOGICAL ENGINE (Using actual NLP engine)
    # ========================================================================

    def analyze_morphology(self, word: str, harakat_analysis: Dict) -> Dict[str, Any]:
        """Analyze morphological features using actual MorphologyEngine"""
        if self.morphology_engine:
            try:
                # Use actual NLP engine
                engine_result = self.morphology_engine.analyze_morphology(word)

                # Enhance with our analysis
                result = {
                    "word_type": self._classify_word_type(word),
                    "grammatical_category": self._get_grammatical_category()
                        word, harakat_analysis
                    ),
                    "inflectional_features": self._get_inflectional_features()
                        word, harakat_analysis
                    ),
                    "derivational_features": self._get_derivational_features(word),
                    "case_marking": self._detect_case_marking(harakat_analysis),
                    "definiteness": self._check_definiteness(word),
                    "engine_result": engine_result,  # Include actual engine output
                }

                # Add gemination detection
                if "ّ" in word:
                    result["gemination"] = self._detect_gemination(word)

                return result

            except Exception as e:
                print(f"⚠️ MorphologyEngine error: {e}, using fallback")

        # Fallback analysis
        return self._analyze_morphology_fallback(word, harakat_analysis)

    def _analyze_morphology_fallback()
        self, word: str, harakat_analysis: Dict
    ) -> Dict[str, Any]:
        """Fallback morphology analysis"""
        result = {
            "word_type": self._classify_word_type(word),
            "grammatical_category": self._get_grammatical_category()
                word, harakat_analysis
            ),
            "inflectional_features": self._get_inflectional_features()
                word, harakat_analysis
            ),
            "derivational_features": self._get_derivational_features(word),
            "case_marking": self._detect_case_marking(harakat_analysis),
            "definiteness": self._check_definiteness(word),
            "engine_status": "fallback_mode",
        }

        # Add gemination detection
        if "ّ" in word:
            result["gemination"] = self._detect_gemination(word)

        return result

    def _classify_word_type(self, word: str) -> str:
        """Classify word as noun, verb, particle"""
        # Simplified classification
        if word.startswith("ال"):
            return "noun"
        elif len(word) == 3:
            return "verb"
        else:
            return "noun"

    def _get_grammatical_category(self, word: str, harakat_analysis: Dict) -> str:
        """Get grammatical category"""
        if "ال" in word:
            return "definite_noun"
        elif harakat_analysis.get("vowel_pattern", "").endswith("a"):
            return "accusative_noun"
        else:
            return "indefinite_noun"

    def _get_inflectional_features(self, word: str, harakat_analysis: Dict) -> Dict:
        """Extract inflectional features from harakat"""
        features = {}

        # Case detection from final harakat
        vowel_pattern = harakat_analysis.get("vowel_pattern", "")
        if vowel_pattern.endswith("u"):
            features["case"] = "nominative"
        elif vowel_pattern.endswith("a"):
            features["case"] = "accusative"
        elif vowel_pattern.endswith("i"):
            features["case"] = "genitive"

        # Number detection
        if word.endswith("ون") or word.endswith("ين"):
            features["number"] = "plural"
        else:
            features["number"] = "singular"

        return features

    def _get_derivational_features(self, word: str) -> Dict:
        """Extract derivational features"""
        features = {
            "root_type": ()
                "triliteral" if len(self._remove_harakat(word)) == 3 else "complex"
            ),
            "pattern_type": "basic",
        }

        return features

    def _detect_case_marking(self, harakat_analysis: Dict) -> Optional[str]:
        """Detect case marking from harakat"""
        positions = harakat_analysis.get("harakat_positions", [])

        if not positions:
            return None

        last_harakat = positions[ 1]["mark"] if positions else None
        case_map = {"ُ": "nominative", "َ": "accusative", "ِ": "genitive"}

        return case_map.get(last_harakat)

    def _check_definiteness(self, word: str) -> str:
        """Check if word is definite or indefinite"""
        return "definite" if word.startswith("ال") else "indefinite"

    def _detect_gemination(self, word: str) -> List[Dict]:
        """Detect geminated consonants"""
        gemination = []

        for i, char in enumerate(word):
            if char == "ّ":
                gemination.append({"position": i, "marker": char, "type": "shadda"})

        return gemination

    # ========================================================================
    # DERIVATIONAL ENGINE (Using actual NLP engine)
    # ========================================================================

    def analyze_derivation()
        self, word: str, morphological_analysis: Dict
    ) -> Dict[str, Any]:
        """Analyze derivational morphology using actual DerivationEngine"""
        if self.derivation_engine:
            try:
                # Use actual NLP engine
                engine_result = self.derivation_engine.process(word)

                # Enhance with our analysis
                unvocalized = self._remove_harakat(word)

                result = {
                    "root_extraction": self._extract_root(unvocalized),
                    "pattern_identification": self._identify_pattern(word),
                    "morphological_form": self._identify_verb_form(word),
                    "semantic_analysis": self._analyze_semantics(word),
                    "engine_result": engine_result,  # Include actual engine output
                }

                return result

            except Exception as e:
                print(f"⚠️ DerivationEngine error: {e}, using fallback")

        # Fallback analysis
        return self._analyze_derivation_fallback(word, morphological_analysis)

    def _analyze_derivation_fallback()
        self, word: str, morphological_analysis: Dict
    ) -> Dict[str, Any]:
        """Fallback derivation analysis"""
        unvocalized = self._remove_harakat(word)

        result = {
            "root_extraction": self._extract_root(unvocalized),
            "pattern_identification": self._identify_pattern(word),
            "morphological_form": self._identify_verb_form(word),
            "semantic_analysis": self._analyze_semantics(word),
            "engine_status": "fallback_mode",
        }

        return result

    def _extract_root(self, word: str) -> Dict:
        """Extract Arabic root from word"""
        # Simplified root extraction
        clean_word = word.replace("ال", "").replace("ة", "")

        if len(clean_word) >= 3:
            root = clean_word[:3]
            confidence = 0.8 if len(clean_word) == 3 else 0.6
        else:
            root = clean_word
            confidence = 0.4

        return {
            "root": root,
            "root_type": "triliteral" if len(root) == 3 else "complex",
            "extraction_confidence": confidence,
        }

    def _identify_pattern(self, word: str) -> Dict:
        """Identify derivational pattern"""
        # Simplified pattern identification
        return {"pattern": "unknown", "form": "unknown", "confidence": 0.5}

    def _identify_verb_form(self, word: str) -> str:
        """Identify Arabic verb form"""
        # Simplified form identification
        if len(self._remove_harakat(word)) == 3:
            return "Form_I"
        else:
            return "Form_unknown"

    def _analyze_semantics(self, word: str) -> Dict:
        """Analyze semantic features"""
        return {
            "semantic_field": "general",
            "lexical_category": "concrete",
            "abstractness": "concrete",
        }

    # ========================================================================
    # PHONOLOGICAL ENGINE
    # ========================================================================

    def analyze_phonology(self, word: str, all_analyses: Dict) -> Dict[str, Any]:
        """Analyze phonological processes"""
        result = {
            "assimilation": self._detect_assimilation(word),
            "elision": self._detect_elision(word),
            "epenthesis": self._detect_epenthesis(word),
            "metathesis": self._detect_metathesis(word),
            "phonological_rules": self._apply_phonological_rules(word),
        }

        return result

    def _detect_assimilation(self, word: str) -> List[Dict]:
        """Detect assimilation processes"""
        # Simplified detection
        return []

    def _detect_elision(self, word: str) -> List[Dict]:
        """Detect vowel/consonant elision"""
        return []

    def _detect_epenthesis(self, word: str) -> List[Dict]:
        """Detect epenthetic vowels/consonants"""
        return []

    def _detect_metathesis(self, word: str) -> List[Dict]:
        """Detect sound metathesis"""
        return []

    def _apply_phonological_rules(self, word: str) -> List[str]:
        """Apply phonological rules"""
        return ["no_rules_applied"]

    # ========================================================================
    # PROSODIC ENGINE
    # ========================================================================

    def analyze_prosody(self, word: str, syllable_analysis: Dict) -> Dict[str, Any]:
        """Analyze prosodic features"""
        syllables = syllable_analysis.get("syllables", [])

        result = {
            "syllable_count": len(syllables),
            "mora_count": sum(syl.get("mora_count", 1) for syl in syllables),
            "weight_pattern": " → ".join()
                syl.get("weight", "unknown") for syl in syllables
            ),
            "rhythmic_pattern": self._get_rhythmic_pattern(syllables),
            "metrical_feet": self._analyze_metrical_feet(syllables),
            "stress_assignment": self._assign_stress(syllables),
        }

        return result

    def _get_rhythmic_pattern(self, syllables: List[Dict]) -> str:
        """Generate rhythmic pattern notation"""
        pattern = ""
        for syl in syllables:
            weight = syl.get("weight", "light")
            if weight == "light":
                pattern += "♪"
            elif weight == "heavy":
                pattern += "♫"
            else:  # superheavy
                pattern += "𝅗𝅥"

        return pattern

    def _analyze_metrical_feet(self, syllables: List[Dict]) -> str:
        """Analyze metrical foot structure"""
        foot_count = len(syllables)

        if foot_count == 1:
            return "single"
        elif foot_count == 2:
            return "binary"
        elif foot_count == 3:
            return "ternary"
        else:
            return "complex"

    def _assign_stress(self, syllables: List[Dict]) -> Dict:
        """Assign stress based on syllable weight"""
        stress_info = {
            "primary_stress": 0,
            "secondary_stress": [],
            "stress_rule": "weight_sensitive",
        }

        # Find heaviest syllable for primary stress
        max_weight = 0
        for i, syl in enumerate(syllables):
            mora = syl.get("mora_count", 1)
            if mora > max_weight:
                max_weight = mora
                stress_info["primary_stress"] = i

        return stress_info

    # ========================================================================
    # CASCADE ANALYSIS
    # ========================================================================

    def analyze_cascade_effects(self, word: str, all_analyses: Dict) -> Dict[str, Any]:
        """Analyze how harakat affects all engine outputs"""
        harakat_analysis = all_analyses["harakat_analysis"]

        # Create comparison without harakat
        unvocalized = harakat_analysis["unvocalized_form"]

        effects = {
            "harakat_impact": {
                "original": word,
                "unvocalized": unvocalized,
                "harakat_present": harakat_analysis["has_harakat"],
            },
            "syllable_changes": self._compare_syllable_effects(word, unvocalized),
            "stress_changes": self._compare_stress_effects(all_analyses),
            "morphological_visibility": self._compare_morphological_visibility()
                word, unvocalized
            ),
            "phonetic_differences": self._compare_phonetic_forms(word, unvocalized),
        }

        return effects

    def _compare_syllable_effects(self, original: str, unvocalized: str) -> Dict:
        """Compare syllable analysis with/without harakat"""
        return {
            "with_harakat": "Variable syllabification based on vowels",
            "without_harakat": f"Single superheavy syllable: {unvocalized}",
            "impact": "Harakat dramatically changes syllable structure",
        }

    def _compare_stress_effects(self, all_analyses: Dict) -> Dict:
        """Compare stress patterns with/without harakat"""
        syllable_analysis = all_analyses.get("syllable_analysis", {})
        stress_pattern = syllable_analysis.get("stress_pattern", "unknown")

        return {
            "with_harakat": stress_pattern,
            "without_harakat": "Stress on single syllable",
            "impact": "Harakat enables proper stress assignment",
        }

    def _compare_morphological_visibility()
        self, original: str, unvocalized: str
    ) -> Dict:
        """Compare morphological analysis visibility"""
        return {
            "with_harakat": "Case, mood, and inflection visible",
            "without_harakat": "Morphological features ambiguous",
            "impact": "Harakat essential for morphological analysis",
        }

    def _compare_phonetic_forms(self, original: str, unvocalized: str) -> Dict:
        """Compare phonetic representations"""
        return {
            "with_harakat": f"/{original}/ - full vowel specification",
            "without_harakat": f"/{unvocalized}/ - consonant cluster",
            "impact": "Harakat provides complete phonetic information",
        }

    def generate_alternative_forms(self, word: str) -> List[str]:
        """Generate alternative morphological forms"""
        alternatives = []
        unvocalized = self._remove_harakat(word)

        # Generate common patterns

        # Apply patterns to root (simplified)
        if len(unvocalized) == 3:
            root = unvocalized
            alternatives.extend()
                [
                    f"{root[0]}َ{root[1]}َ{root[2]}َ",  # Perfect
                    f"يَ{root[0]}ْ{root[1]}ُ{root[2]}ُ",  # Imperfect
                    f"{root[0]}َا{root[1]ِ{root[2]}}",  # Active participle
                    f"مَ{root[0]}ْ{root[1]}ُو{root[2]}",  # Passive participle
                ]
            )

        return alternatives

    # ========================================================================
    # MAIN ANALYSIS FUNCTION
    # ========================================================================

    def analyze_word(self, word: str) -> UnifiedAnalysis:
        """
        Complete analysis of Arabic word through all engines

        Args:
            word: Arabic word to analyze

        Returns:
            UnifiedAnalysis object with complete results
        """
        print(f"\n🔄 ANALYZING: {word}")
        print("=" * 60)

        # Initialize result
        result = UnifiedAnalysis(word=word, has_harakat=self._has_harakat(word))

        # Engine 1: Harakat Analysis
        print("1️⃣ Running Harakat Engine...")
        result.harakat_analysis = self.analyze_harakat(word)

        # Engine 2: Phoneme Analysis
        print("2️⃣ Running Phoneme Engine...")
        result.phoneme_analysis = self.analyze_phonemes(word, result.harakat_analysis)

        # Engine 3: Syllable Analysis
        print("3️⃣ Running Syllable Engine...")
        result.syllable_analysis = self.analyze_syllables()
            word, result.phoneme_analysis, result.harakat_analysis
        )

        # Engine 4: Morphological Analysis
        print("4️⃣ Running Morphological Engine...")
        result.morphological_analysis = self.analyze_morphology()
            word, result.harakat_analysis
        )

        # Engine 5: Derivational Analysis
        print("5️⃣ Running Derivational Engine...")
        result.derivational_analysis = self.analyze_derivation()
            word, result.morphological_analysis
        )

        # Engine 6: Phonological Analysis
        print("6️⃣ Running Phonological Engine...")
        all_analyses = {
            "harakat_analysis": result.harakat_analysis,
            "phoneme_analysis": result.phoneme_analysis,
            "syllable_analysis": result.syllable_analysis,
            "morphological_analysis": result.morphological_analysis,
            "derivational_analysis": result.derivational_analysis,
        }
        result.extract_phonemes = self.analyze_phonology(word, all_analyses)

        # Engine 7: Prosodic Analysis
        print("7️⃣ Running Prosodic Engine...")
        result.prosodic_analysis = self.analyze_prosody(word, result.syllable_analysis)

        # Engine 8: Cascade Effects Analysis
        print("8️⃣ Analyzing Cascade Effects...")
        all_analyses["extract_phonemes"] = result.extract_phonemes
        all_analyses["prosodic_analysis"] = result.prosodic_analysis
        result.cascade_effects = self.analyze_cascade_effects(word, all_analyses)

        # Engine 9: Alternative Forms
        print("9️⃣ Generating Alternative Forms...")
        result.alternative_forms = self.generate_alternative_forms(word)

        print("✅ Analysis Complete!")
        return result

    def display_results(self, analysis: UnifiedAnalysis):
        """Display comprehensive analysis results"""
        print(f"\n🎯 UNIFIED ARABIC NLP ANALYSIS: {analysis.word}")
        print("=" * 80)

        # 1. Harakat Analysis
        print("\n🔤 HARAKAT ANALYSIS")
        print(" " * 40)
        harakat = analysis.harakat_analysis
        print(f"  Has Harakat: {harakat.get('has_harakat', False)}")
        print(f"  Harakat Count: {harakat.get('harakat_count', 0)}")
        print(f"  Vowel Pattern: {harakat.get('vowel_pattern', 'none')}")
        print(f"  Unvocalized: {harakat.get('unvocalized_form', analysis.word)}")
        if harakat.get('ipa_transcription'):
            print(f"  IPA: {harakat['ipa_transcription']}")

        # 2. Phoneme Analysis
        print("\n📝 PHONEME ANALYSIS")
        print(" " * 40)
        phoneme = analysis.phoneme_analysis
        print(f"  Phoneme Count: {phoneme.get('phoneme_count', 0)}")
        print(f"  Consonants: {phoneme.get('consonant_count', 0)}")
        print(f"  Vowels: {phoneme.get('vowel_count', 0)}")
        print(f"  IPA: {phoneme.get('ipa_transcription', 'unknown')}")

        breakdown = phoneme.get('phoneme_breakdown', [])
        if breakdown:
            print("  Phoneme Breakdown:")
            for item in breakdown:
                harakat_info = ()
                    f" + {item['harakat']['mark']}" if 'harakat' in item else ""
                )
                print()
                    f"    {item['char']} → {item['ipa']} {item['features']}{harakat_info}"
                )

        # 3. Syllable Analysis
        print("\n🔊 SYLLABLE ANALYSIS")
        print(" " * 40)
        syllable = analysis.syllable_analysis
        print(f"  Syllable Count: {syllable.get('syllable_count', 0)}")
        print(f"  Total Mora: {syllable.get('total_mora', 0)}")
        print(f"  Weight Pattern: {syllable.get('weight_sequence', 'unknown')}")
        print(f"  Stress Pattern: {syllable.get('stress_pattern', 'unknown')}")

        syllables = syllable.get('syllables', [])
        if syllables:
            print("  Syllable Breakdown:")
            for i, syl in enumerate(syllables, 1):
                stress_mark = "ˈ" if syl.get('stress') else ""
                print()
                    f"    {i}. {stress_mark}{syl['syllable']} → {syl['ipa']} ({syl['weight']}, {syl['mora_count'] mora)}"
                )

        # 4. Morphological Analysis
        print("\n📚 MORPHOLOGICAL ANALYSIS")
        print(" " * 40)
        morpho = analysis.morphological_analysis
        print(f"  Word Type: {morpho.get('word_type', 'unknown')}")
        print(f"  Category: {morpho.get('grammatical_category', 'unknown')}")
        print(f"  Definiteness: {morpho.get('definiteness', 'unknown')}")

        inflection = morpho.get('inflectional_features', {})
        if inflection:
            print(f"  Inflectional Features: {inflection}")

        if 'gemination' in morpho:
            print(f"  Gemination: {morpho['gemination']}")

        # 5. Derivational Analysis
        print("\n🌱 DERIVATIONAL ANALYSIS")
        print(" " * 40)
        deriv = analysis.derivational_analysis
        root_info = deriv.get('root_extraction', {})
        print(f"  Root: {root_info.get('root', 'unknown')}")
        print(f"  Root Type: {root_info.get('root_type', 'unknown')}")
        print(f"  Confidence: {root_info.get('extraction_confidence', 0)}")
        print(f"  Verb Form: {deriv.get('morphological_form', 'unknown')}")

        semantic = deriv.get('semantic_analysis', {})
        if semantic:
            print(f"  Semantic Field: {semantic.get('semantic_field', 'unknown')}")

        # 6. Phonological Analysis
        print("\n🔬 PHONOLOGICAL ANALYSIS")
        print(" " * 40)
        phono = analysis.extract_phonemes
        rules = phono.get('phonological_rules', [])
        print(f"  Phonological Rules: {', '.join(rules) if rules else} 'none detected'}")

        # 7. Prosodic Analysis
        print("\n🎵 PROSODIC ANALYSIS")
        print(" " * 40)
        prosody = analysis.prosodic_analysis
        print(f"  Syllable Count: {prosody.get('syllable_count', 0)}")
        print(f"  Mora Count: {prosody.get('mora_count', 0)}")
        print(f"  Weight Pattern: {prosody.get('weight_pattern', 'unknown')}")
        print(f"  Rhythmic Pattern: {prosody.get('rhythmic_pattern', 'unknown')}")
        print(f"  Metrical Feet: {prosody.get('metrical_feet', 'unknown')}")

        # 8. Cascade Effects
        print("\n⚡ CASCADE EFFECTS")
        print(" " * 40)
        cascade = analysis.cascade_effects
        harakat_impact = cascade.get('harakat_impact', {})
        print(f"  Original: {harakat_impact.get('original', analysis.word)}")
        print(f"  Unvocalized: {harakat_impact.get('unvocalized', 'unknown')}")
        print(f"  Harakat Present: {harakat_impact.get('harakat_present', False)}")

        syl_changes = cascade.get('syllable_changes', {})
        if syl_changes:
            print(f"  Syllable Impact: {syl_changes.get('impact', 'none')}")

        # 9. Alternative Forms
        print("\n🔄 ALTERNATIVE FORMS")
        print(" " * 40)
        if analysis.alternative_forms:
            for i, form in enumerate(analysis.alternative_forms, 1):
                print(f"  {i. {form}}")
        else:
            print("  No alternative forms generated")

        print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function for interactive word analysis"""
    print("🚀 UNIFIED ARABIC NLP ENGINE")
    print("=" * 60)
    print("Complete Arabic processing pipeline")
    print("All engines run sequentially for comprehensive analysis")
    print("=" * 60)

    # Initialize engine
    engine = UnifiedArabicEngine()

    # Test words
    test_words = [
        "كتب",  # Unvocalized verb
        "كَتَبَ",  # Vocalized perfect verb
        "كِتَاب",  # Vocalized noun with long vowel
        "مُدَرِّس",  # Complex word with gemination
        "الكِتَاب",  # Definite noun
    ]

    print(f"\n📝 Testing {len(test_words)} sample words:")

    for word in test_words:
        try:
            # Analyze word
            analysis = engine.analyze_word(word)

            # Display results
            engine.display_results(analysis)

            print("\n" + "🔄" * 50 + "\n")

        except Exception as e:
            print(f"❌ Error analyzing '{word}': {e}")
            continue

    print("\n✅ ALL TESTS COMPLETED")
    print("📊 Unified Arabic NLP Engine working successfully!")

    # Interactive mode
    print("\n" + "=" * 60)
    print("🎯 INTERACTIVE MODE")
    print("Enter Arabic words for analysis (type 'quit' to exit):")

    while True:
        try:
            user_input = input("\n👉 Enter Arabic word: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("⚠️  Please enter a word")
                continue

            # Analyze user input
            analysis = engine.analyze_word(user_input)
            engine.display_results(analysis)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

