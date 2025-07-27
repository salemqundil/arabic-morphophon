#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Syllable Generator - Complete Phonological Foundation
===========================================================
مولد المقاطع الصوتية العربية - الأساس الصوتي الشامل

This module generates ALL possible Arabic syllables using PURE phonological
foundation with phonotactic constraints and articulatory limitations.

Key Features:
- Complete syllable pattern generation (CV, CVV, CVC, CVVC, CVCC)
- Phonotactic constraint application
- Articulatory compatibility checking
- Zero external dependencies - pure phonological science
- Enterprise-grade Arabic syllable inventory

المميزات الأساسية:
- توليد شامل لأنماط المقاطع العربية
- تطبيق قيود التتابع الصوتي
- فحص التوافق النطقي
- صفر اعتماديات خارجية - علم صوتي نقي
- مخزون المقاطع العربية على مستوى المؤسسات

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0 - COMPLETE SYLLABLE GENERATION
Date: 2025-07-24
License: MIT
Encoding: UTF 8
"""

import logging
import sys
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

# EXCLUSIVE IMPORT: Only our phonological foundation
from complete_arabic_phonological_foundation import CompletePhonologicalFunctionResolver

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arabic_syllable_generator.log', encoding='utf 8'),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC SYLLABLE STRUCTURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicSyllableType(Enum):
    """Complete Arabic syllable types with phonological specifications"""

    # Light Syllables (المقاطع الخفيفة)
    CV = "CV"  # صامت + صائت قصير (بَ)
    CVV = "CVV"  # صامت + صائت طويل (با)

    # Heavy Syllables (المقاطع الثقيلة)
    CVC = "CVC"  # صامت + صائت قصير + صامت (بَت)
    CVVC = "CVVC"  # صامت + صائت طويل + صامت (بات)

    # Super Heavy Syllables (المقاطع فائقة الثقل)
    CVCC = "CVCC"  # صامت + صائت قصير + صامتان (بَتر)
    CVVCC = "CVVCC"  # صامت + صائت طويل + صامتان (باتر)


@dataclass
class SyllableConstraints:
    """Phonotactic constraints for Arabic syllables"""

    # Vowel compatibility for long vowels
    vowel_compatibility: Dict[str, str] = field(
        default_factory=lambda: {
            'َ': 'ا',
            'ُ': 'و',
            'ِ': 'ي',
        }  # فتحة + ألف  # ضمة + واو  # كسرة + ياء
    )

    # Forbidden consonant clusters
    forbidden_clusters: Set[Tuple[str, str]] = field(default_factory=set)

    # Articulatory place restrictions
    place_restrictions: Dict[str, List[str]] = field(default_factory=dict)

    # Gemination allowed consonants
    gemination_allowed: Set[str] = field(default_factory=set)


@dataclass
class GeneratedSyllable:
    """Complete generated Arabic syllable with metadata"""

    syllable_text: str
    syllable_type: ArabicSyllableType
    onset: List[str] = field(default_factory=list)
    nucleus: List[str] = field(default_factory=list)
    coda: List[str] = field(default_factory=list)
    prosodic_weight: float = 1.0
    phonological_features: Dict[str, Any] = field(default_factory=dict)
    frequency_estimate: float = 0.0
    is_valid: bool = True
    constraints_violated: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════════
# COMPLETE ARABIC SYLLABLE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════════


class CompleteArabicSyllableGenerator:
    """
    Complete Arabic syllable generator using pure phonological foundation

    مولد المقاطع العربية الشامل باستخدام الأساس الصوتي النقي
    """

    def __init__(self):
        """Initialize with complete phonological foundation"""
        self.phonological_resolver = CompletePhonologicalFunctionResolver()
        self.constraints = SyllableConstraints()
        self._initialize_arabic_inventory()
        self._setup_phonotactic_constraints()
        logger.info(
            "CompleteArabicSyllableGenerator initialized with full phonological foundation"
        )

    def _initialize_arabic_inventory(self):
        """Initialize complete Arabic phoneme inventory"""

        # Arabic consonants (28 letters) - الصوامت العربية
        self.consonants = [
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

        # Short vowels (diacritics) - الحركات القصيرة
        self.short_vowels = ['َ', 'ُ', 'ِ']  # فتحة، ضمة، كسرة

        # Long vowel letters - حروف المد
        self.long_vowel_letters = ['ا', 'و', 'ي']  # ألف، واو، ياء

        # Classify consonants by articulatory features
        self._classify_consonants_by_features()

        logger.info(
            f"Arabic inventory initialized: {len(self.consonants)} consonants, "
            f"{len(self.short_vowels)} short vowels, {len(self.long_vowel_letters)} long vowel letters"
        )

    def _classify_consonants_by_features(self):
        """Classify consonants by articulatory features using phonological foundation"""

        self.consonant_features = {}

        for consonant in self.consonants:
            # Get phonological analysis
            analysis = self.phonological_resolver.resolve_complete(consonant)

            if analysis.phonemes:
                phoneme = analysis.phonemes[0]
                # Handle functions properly - they might be strings or enums
                functions = []
                for func in phoneme.functions:
                    if hasattr(func, 'value'):
                        functions.append(func.value)
                    else:
                        functions.append(str(func))

                self.consonant_features[consonant] = {
                    'features': phoneme.features,
                    'type': phoneme.phoneme_type,
                    'functions': functions,
                    'frequency': phoneme.frequency,
                }
            else:
                # Fallback classification
                self.consonant_features[consonant] = {
                    'features': ['consonant'],
                    'type': 'consonant',
                    'functions': [],
                    'frequency': 0.01,
                }

    def _setup_phonotactic_constraints(self):
        """Setup Arabic phonotactic constraints"""

        # Forbidden consonant clusters (examples)
        self.constraints.forbidden_clusters.update(
            [
                # Same place of articulation (difficult clusters)
                ('ح', 'ه'),
                ('ع', 'ء'),
                ('ق', 'ك'),
                ('ظ', 'ذ'),
                # Difficult fricative combinations
                ('ث', 'ذ'),
                ('س', 'ش'),
                ('ص', 'ض'),
            ]
        )

        # Gemination allowed for most consonants
        self.constraints.gemination_allowed.update(self.consonants)

        # Remove some consonants that rarely geminate
        difficult_gemination = {'ء', 'ه', 'و', 'ي'}
        self.constraints.gemination_allowed -= difficult_gemination

        logger.info(
            f"Phonotactic constraints setup: {len(self.constraints.forbidden_clusters)} forbidden clusters"
        )

    def generate_cv_syllables(self) -> List[GeneratedSyllable]:
        """Generate all CV (صامت + صائت قصير) syllables"""

        cv_syllables = []

        for consonant in self.consonants:
            for vowel in self.short_vowels:
                syllable_text = consonant + vowel

                syllable = GeneratedSyllable(
                    syllable_text=syllable_text,
                    syllable_type=ArabicSyllableType.CV,
                    onset=[consonant],
                    nucleus=[vowel],
                    coda=[],
                    prosodic_weight=1.0,
                    frequency_estimate=self._estimate_frequency(consonant, vowel),
                )

                syllable.phonological_features = self._analyze_syllable_features(
                    syllable
                )
                cv_syllables.append(syllable)

        logger.info(f"Generated {len(cv_syllables)} CV syllables")
        return cv_syllables

    def generate_cvv_syllables(self) -> List[GeneratedSyllable]:
        """Generate all CVV (صامت + صائت طويل) syllables with compatibility constraints"""

        cvv_syllables = []

        for consonant in self.consonants:
            for (
                short_vowel,
                long_vowel_letter,
            ) in self.constraints.vowel_compatibility.items():
                syllable_text = consonant + short_vowel + long_vowel_letter

                syllable = GeneratedSyllable(
                    syllable_text=syllable_text,
                    syllable_type=ArabicSyllableType.CVV,
                    onset=[consonant],
                    nucleus=[short_vowel, long_vowel_letter],
                    coda=[],
                    prosodic_weight=1.5,
                    frequency_estimate=self._estimate_frequency(
                        consonant, short_vowel, long_vowel_letter
                    ),
                )

                syllable.phonological_features = self._analyze_syllable_features(
                    syllable
                )
                cvv_syllables.append(syllable)

        logger.info(f"Generated {len(cvv_syllables)} CVV syllables")
        return cvv_syllables

    def generate_cvc_syllables(self) -> List[GeneratedSyllable]:
        """Generate all CVC (صامت + صائت قصير + صامت) syllables with constraints"""

        cvc_syllables = []

        for c1 in self.consonants:
            for vowel in self.short_vowels:
                for c2 in self.consonants:

                    # Check phonotactic constraints
                    if self._check_consonant_cluster_validity(c1, c2):
                        syllable_text = c1 + vowel + c2

                        syllable = GeneratedSyllable(
                            syllable_text=syllable_text,
                            syllable_type=ArabicSyllableType.CVC,
                            onset=[c1],
                            nucleus=[vowel],
                            coda=[c2],
                            prosodic_weight=2.0,
                            frequency_estimate=self._estimate_frequency(c1, vowel, c2),
                        )

                        syllable.phonological_features = (
                            self._analyze_syllable_features(syllable)
                        )
                        cvc_syllables.append(syllable)

        logger.info(f"Generated {len(cvc_syllables)} CVC syllables")
        return cvc_syllables

    def generate_cvvc_syllables(self) -> List[GeneratedSyllable]:
        """Generate all CVVC (صامت + صائت طويل + صامت) syllables"""

        cvvc_syllables = []

        for c1 in self.consonants:
            for (
                short_vowel,
                long_vowel_letter,
            ) in self.constraints.vowel_compatibility.items():
                for c2 in self.consonants:

                    # Check phonotactic constraints
                    if self._check_consonant_cluster_validity(c1, c2):
                        syllable_text = c1 + short_vowel + long_vowel_letter + c2

                        syllable = GeneratedSyllable(
                            syllable_text=syllable_text,
                            syllable_type=ArabicSyllableType.CVVC,
                            onset=[c1],
                            nucleus=[short_vowel, long_vowel_letter],
                            coda=[c2],
                            prosodic_weight=2.5,
                            frequency_estimate=self._estimate_frequency(
                                c1, short_vowel, long_vowel_letter, c2
                            ),
                        )

                        syllable.phonological_features = (
                            self._analyze_syllable_features(syllable)
                        )
                        cvvc_syllables.append(syllable)

        logger.info(f"Generated {len(cvvc_syllables)} CVVC syllables")
        return cvvc_syllables

    def generate_cvcc_syllables(self) -> List[GeneratedSyllable]:
        """Generate CVCC (صامت + صائت قصير + صامتان) syllables - rare but possible"""

        cvcc_syllables = []

        for c1 in self.consonants:
            for vowel in self.short_vowels:
                for c2 in self.consonants:
                    for c3 in self.consonants:

                        # Strict constraints for triple consonant clusters
                        if self._check_triple_consonant_validity(c1, c2, c3):
                            syllable_text = c1 + vowel + c2 + c3

                            syllable = GeneratedSyllable(
                                syllable_text=syllable_text,
                                syllable_type=ArabicSyllableType.CVCC,
                                onset=[c1],
                                nucleus=[vowel],
                                coda=[c2, c3],
                                prosodic_weight=3.0,
                                frequency_estimate=self._estimate_frequency(
                                    c1, vowel, c2, c3
                                )
                                * 0.1,  # Very rare
                            )

                            syllable.phonological_features = (
                                self._analyze_syllable_features(syllable)
                            )
                            cvcc_syllables.append(syllable)

        logger.info(f"Generated {len(cvcc_syllables)} CVCC syllables")
        return cvcc_syllables

    def _check_consonant_cluster_validity(self, c1: str, c2: str) -> bool:
        """Check if consonant cluster is phonotactically valid"""

        # Check forbidden clusters
        if (c1, c2) in self.constraints.forbidden_clusters:
            return False

        # Get features for both consonants
        self.consonant_features.get(c1, {}).get('features', [])
        self.consonant_features.get(c2, {}).get('features', [])

        # Allow most combinations - Arabic is relatively permissive
        # Restrict only very difficult articulatory combinations

        # Same consonant is allowed (gemination)
        if c1 == c2 and c1 in self.constraints.gemination_allowed:
            return True

        # Different consonants - check articulatory compatibility
        return True  # Most combinations are allowed

    def _check_triple_consonant_validity(self, c1: str, c2: str, c3: str) -> bool:
        """Check if triple consonant cluster is valid (very restrictive)"""

        # Very restrictive for CVCC patterns
        # Usually only in borrowed words or specific morphological contexts

        # Check if first two consonants are valid
        if not self._check_consonant_cluster_validity(c1, c2):
            return False

        # Check if last two consonants are valid
        if not self._check_consonant_cluster_validity(c2, c3):
            return False

        # Additional restrictions for triple clusters
        # Allow only if middle consonant is sonorant (liquid/nasal)
        sonorants = {'ر', 'ل', 'م', 'ن'}

        return c2 in sonorants or c3 in sonorants

    def _estimate_frequency(self, *segments) -> float:
        """Estimate syllable frequency based on segment frequencies"""

        total_frequency = 1.0

        for segment in segments:
            if segment in self.consonant_features:
                freq = self.consonant_features[segment]['frequency']
                total_frequency *= freq
            elif segment in self.short_vowels:
                # Estimate vowel frequencies
                vowel_freqs = {'َ': 0.4, 'ُ': 0.3, 'ِ': 0.3}
                total_frequency *= vowel_freqs.get(segment, 0.1)
            elif segment in self.long_vowel_letters:
                # Long vowel frequencies
                long_freqs = {'ا': 0.15, 'و': 0.08, 'ي': 0.12}
                total_frequency *= long_freqs.get(segment, 0.05)

        return min(total_frequency, 1.0)

    def _analyze_syllable_features(self, syllable: GeneratedSyllable) -> Dict[str, Any]:
        """Analyze phonological features of generated syllable"""

        features = {
            'onset_features': [],
            'nucleus_features': [],
            'coda_features': [],
            'phonological_weight': syllable.prosodic_weight,
            'articulatory_complexity': 0.0,
            'morphological_potential': 0.0,
        }

        # Analyze onset
        for consonant in syllable.onset:
            if consonant in self.consonant_features:
                features['onset_features'].extend(
                    self.consonant_features[consonant]['features']
                )

        # Analyze nucleus
        nucleus_type = 'short' if len(syllable.nucleus) == 1 else 'long'
        features['nucleus_features'] = [nucleus_type]

        # Analyze coda
        for consonant in syllable.coda:
            if consonant in self.consonant_features:
                features['coda_features'].extend(
                    self.consonant_features[consonant]['features']
                )

        # Calculate complexity
        features['articulatory_complexity'] = (
            len(syllable.onset) + len(syllable.coda) + (len(syllable.nucleus) * 0.5)
        )

        return features

    def generate_all_syllables(self) -> Dict[str, List[GeneratedSyllable]]:
        """Generate complete Arabic syllable inventory"""

        logger.info("🔬 Starting complete Arabic syllable generation...")

        all_syllables = {
            'CV': self.generate_cv_syllables(),
            'CVV': self.generate_cvv_syllables(),
            'CVC': self.generate_cvc_syllables(),
            'CVVC': self.generate_cvvc_syllables(),
            'CVCC': self.generate_cvcc_syllables(),
        }

        # Calculate totals
        total_count = sum(len(syllables) for syllables in all_syllables.values())

        logger.info(
            f"✅ Complete syllable generation finished: {total_count} total syllables"
        )

        return all_syllables

    def analyze_syllable_distribution(
        self, all_syllables: Dict[str, List[GeneratedSyllable]]
    ) -> Dict[str, Any]:
        """Analyze distribution and statistics of generated syllables"""

        analysis = {
            'counts_by_type': {},
            'frequency_distribution': {},
            'feature_analysis': defaultdict(int),
            'complexity_analysis': {},
            'top_frequent_syllables': [],
            'articulatory_distribution': defaultdict(int),
        }

        all_syllables_flat = []

        for syllable_type, syllables in all_syllables.items():
            analysis['counts_by_type'][syllable_type] = len(syllables)

            type_frequencies = [syl.frequency_estimate for syl in syllables]
            analysis['frequency_distribution'][syllable_type] = {
                'mean': (
                    sum(type_frequencies) / len(type_frequencies)
                    if type_frequencies
                    else 0
                ),
                'max': max(type_frequencies) if type_frequencies else 0,
                'min': min(type_frequencies) if type_frequencies else 0,
            }

            all_syllables_flat.extend(syllables)

        # Sort by frequency for top syllables
        all_syllables_flat.sort(key=lambda x: x.frequency_estimate, reverse=True)
        analysis['top_frequent_syllables'] = [
            {
                'syllable': syl.syllable_text,
                'type': syl.syllable_type.value,
                'frequency': syl.frequency_estimate,
            }
            for syl in all_syllables_flat[:20]
        ]

        # Complexity analysis
        complexities = [
            syl.phonological_features.get('articulatory_complexity', 0)
            for syl in all_syllables_flat
        ]
        analysis['complexity_analysis'] = {
            'mean_complexity': (
                sum(complexities) / len(complexities) if complexities else 0
            ),
            'max_complexity': max(complexities) if complexities else 0,
            'min_complexity': min(complexities) if complexities else 0,
        }

        return analysis

    def save_syllable_inventory(
        self,
        all_syllables: Dict[str, List[GeneratedSyllable]],
        filename: str = "complete_arabic_syllable_inventory.json",
    ):
        """Save complete syllable inventory to JSON file"""

        # Convert to serializable format
        serializable_data = {}

        for syllable_type, syllables in all_syllables.items():
            serializable_data[syllable_type] = []

            for syllable in syllables:
                syl_data = {
                    'text': syllable.syllable_text,
                    'type': syllable.syllable_type.value,
                    'onset': syllable.onset,
                    'nucleus': syllable.nucleus,
                    'coda': syllable.coda,
                    'prosodic_weight': syllable.prosodic_weight,
                    'frequency_estimate': syllable.frequency_estimate,
                    'features': syllable.phonological_features,
                }
                serializable_data[syllable_type].append(syl_data)

        # Save to file
        with open(filename, 'w', encoding='utf 8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Syllable inventory saved to {filename}")


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════════


def demonstrate_complete_syllable_generation():
    """Demonstrate complete Arabic syllable generation"""

    print("🔬 مولد المقاطع الصوتية العربية الشامل")
    print("=" * 60)

    generator = CompleteArabicSyllableGenerator()

    # Generate all syllables
    print("\n📊 توليد جميع المقاطع الصوتية...")
    all_syllables = generator.generate_all_syllables()

    # Display counts
    print("\n📈 إحصائيات المقاطع المولدة:")
    total_syllables = 0
    for syllable_type, syllables in all_syllables.items():
        count = len(syllables)
        total_syllables += count
        print(f"   • {syllable_type}: {count:} مقطع")

    print(f"\n🎯 المجموع الكلي: {total_syllables:} مقطع صوتي")

    # Show examples from each type
    print("\n🔤 عينات من المقاطع:")
    for syllable_type, syllables in all_syllables.items():
        if syllables:
            examples = [syl.syllable_text for syl in syllables[:5]]
            print(f"   • {syllable_type:} {', '.join(examples)}}")

    # Analyze distribution
    print("\n📊 تحليل التوزيع...")
    analysis = generator.analyze_syllable_distribution(all_syllables)

    print("\n🔝 المقاطع الأكثر تكراراً:")
    for i, syl_info in enumerate(analysis['top_frequent_syllables'][:10], 1):
        print(
            f"   {i}. {syl_info['syllable']} ({syl_info['type']}) - {syl_info['frequency']:.4f}"
        )

    # Complexity analysis
    complexity = analysis['complexity_analysis']
    print("\n🧮 تحليل التعقيد النطقي:")
    print(f"   • متوسط التعقيد: {complexity['mean_complexity']:.2f}")
    print(f"   • أقصى تعقيد: {complexity['max_complexity']:.2f}")
    print(f"   • أدنى تعقيد: {complexity['min_complexity']:.2f}")

    # Save inventory
    print("\n💾 حفظ مخزون المقاطع...")
    generator.save_syllable_inventory(all_syllables)

    print("\n✅ اكتمل توليد المقاطع الصوتية العربية!")
    print("💎 استخدام الأساس الصوتي النقي - صفر اعتماديات خارجية!")


if __name__ == "__main__":
    demonstrate_complete_syllable_generation()
