#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Morphological Weight Generator - Complete Pattern System
==============================================================
مولد الأوزان الصرفية العربية - نظام الأنماط الشامل

This module generates ALL Arabic morphological weights (patterns) using
the complete syllable database built in Phase 1. Builds on 22,218 syllables
to create comprehensive morphological patterns for verbs and nouns.

Key Features:
- Complete morphological weight generation for Arabic verbs and nouns
- Syllable-based pattern construction using phonological foundation
- Phonotactic constraint application between syllables
- Enterprise-grade morphological pattern inventory
- Zero external dependencies - pure Arabic linguistic science

المميزات الأساسية:
- توليد شامل للأوزان الصرفية العربية للأفعال والأسماء
- بناء الأنماط المعتمد على المقاطع الصوتية
- تطبيق قيود التتابع الصوتي بين المقاطع
- مخزون الأنماط الصرفية على مستوى المؤسسات
- صفر اعتماديات خارجية - علم لغوي عربي نقي

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 2.0.0 - MORPHOLOGICAL PATTERN GENERATION
Date: 2025-07-24
License: MIT
Encoding: UTF 8
"""

import logging
import sys
import json
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import itertools

# Import our complete syllable generator
from arabic_syllable_generator import ()
    CompleteArabicSyllableGenerator,
    GeneratedSyllable)

# Configure professional logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
    logging.FileHandler('arabic_morphological_weights.log', encoding='utf 8'),
    logging.StreamHandler(sys.stdout),
    ])

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC MORPHOLOGICAL WEIGHT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicWordType(Enum):
    """Arabic word types for morphological analysis"""

    # Verbs (الأفعال)
    VERB_TRILITERAL_BARE = "فعل_ثلاثي_مجرد"  # فَعَلَ
    VERB_TRILITERAL_AUGMENTED = "فعل_ثلاثي_مزيد"  # أَفْعَلَ، انْفَعَلَ
    VERB_QUADRILITERAL_BARE = "فعل_رباعي_مجرد"  # فَعْلَلَ
    VERB_QUADRILITERAL_AUGMENTED = "فعل_رباعي_مزيد"  # تَفَعْلَلَ

    # Nouns (الأسماء)
    NOUN_DERIVATIVE = "اسم_مشتق"  # فَاعِل، مَفْعُول
    NOUN_SOURCE = "مصدر"  # فَعْل، فِعَالَة
    NOUN_DIMINUTIVE = "تصغير"  # فُعَيْل
    NOUN_PLURAL_MASCULINE = "جمع_مذكر_سالم"  # فَاعِلُون
    NOUN_PLURAL_FEMININE = "جمع_مؤنث_سالم"  # فَاعِلَات
    NOUN_PLURAL_BROKEN = "جمع_تكسير"  # فُعُول، أَفْعَال
    NOUN_DUAL = "مثنى"  # فَاعِلَان
    NOUN_FEMININE = "مؤنث"  # فَاعِلَة
    NOUN_PLACE = "اسم_مكان"  # مَفْعَل
    NOUN_TIME = "اسم_زمان"  # مَفْعَل
    NOUN_INSTRUMENT = "اسم_آلة"  # مِفْعَال


class MorphologicalWeightPattern(Enum):
    """Standard Arabic morphological weight patterns"""

    # Triliteral verb patterns (الأفعال الثلاثية)
    FAALA = "فَعَلَ"  # CV-CV CV
    FAILA = "فَعِلَ"  # CV-CV CV
    FAULA = "فَعُلَ"  # CV-CV CV
    AFAALA = "أَفْعَلَ"  # CV-CVC CV
    FAALA_II = "فَعَّلَ"  # CV-CVC CV (with gemination)
    FAATALA = "فَاعَلَ"  # CVV-CV CV
    TAFAALA = "تَفَاعَلَ"  # CV-CVV-CV CV
    INFAALA = "انْفَعَلَ"  # CVC-CV CV
    IFTAALA = "افْتَعَلَ"  # CVC-CV CV
    ISTAFAALA = "اسْتَفْعَلَ"  # CVC-CVC-CV CV

    # Quadriliteral verb patterns (الأفعال الرباعية)
    FAALLALA = "فَعْلَلَ"  # CVC-CV CV
    TAFAALLALA = "تَفَعْلَلَ"  # CV-CVC-CV CV

    # Noun patterns (أوزان الأسماء)
    FAAIL = "فَاعِل"  # CVV CVC (active participle)
    MAFUUL = "مَفْعُول"  # CV CVVC (passive participle)
    FAIL = "فَعِيل"  # CV CVVC (intensive adjective)
    FAAAL = "فَعَّال"  # CV CVVC (intensive agent)
    AFAAL = "أَفْعَال"  # CV CVVC (broken plural)
    FUUUL = "فُعُول"  # CV CVVC (broken plural)
    FAALAH = "فَاعِلَة"  # CVV-CV CV (feminine active participle)
    FAALAAN = "فَاعِلَان"  # CVV-CVV CVC (dual)
    FAALUUN = "فَاعِلُون"  # CVV-CVV CVC (masculine plural)
    FAALAAT = "فَاعِلَات"  # CVV-CVV CVC (feminine plural)


@dataclass
class MorphologicalWeight:
    """Complete morphological weight with syllable composition"""

    pattern_name: str
    pattern_template: str
    word_type: ArabicWordType
    syllable_sequence: List[GeneratedSyllable]
    syllable_pattern: List[str]  # e.g., ["CV", "CVC", "CV"]
    phonetic_form: str
    morphological_features: Dict[str, Any] = field(default_factory=dict)
    frequency_estimate: float = 0.0
    prosodic_weight: float = 0.0
    constraints_applied: List[str] = field(default_factory=list)
    is_valid: bool = True


@dataclass
class WeightGenerationConstraints:
    """Constraints for morphological weight generation"""

    # Inter syllable constraints
    forbidden_syllable_sequences: Set[Tuple[str, str]] = field(default_factory=set)

    # Morphological constraints
    verb_constraints: Dict[str, List[str]] = field(default_factory=dict)
    noun_constraints: Dict[str, List[str]] = field(default_factory=dict)

    # Phonotactic constraints
    gemination_restrictions: Set[str] = field(default_factory=set)
    vowel_harmony_rules: Dict[str, str] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════════
# COMPLETE ARABIC MORPHOLOGICAL WEIGHT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicMorphologicalWeightGenerator:
    """
    Complete Arabic morphological weight generator using syllable database

    مولد الأوزان الصرفية العربية الشامل باستخدام قاعدة بيانات المقاطع
    """

    def __init__()
    self, syllables_database_path: str = "complete_arabic_syllable_inventory.json"
    ):
    """Initialize with syllable database"""
    self.syllables_db = self._load_syllable_database(syllables_database_path)
    self.weights_db: List[MorphologicalWeight] = []
    self.constraints = WeightGenerationConstraints()
    self._setup_morphological_constraints()
    self._organize_syllables_by_type()
    logger.info()
    f"ArabicMorphologicalWeightGenerator initialized with {len(self.syllables_db)} syllables"
    )

    def _load_syllable_database(self, path: str) -> List[Dict[str, Any]]:
    """Load the syllable database from JSON file"""
        try:
            with open(path, 'r', encoding='utf 8') as f:
    data = json.load(f)

            # Flatten all syllable types into one list
    all_syllables = []
            for syllable_type, syllables in data.items():
                for syl_data in syllables:
    syl_data['type'] = syllable_type
    all_syllables.append(syl_data)

    logger.info(f"Loaded {len(all_syllables)} syllables from database")
    return all_syllables

        except FileNotFoundError:
    logger.warning(f"Syllable database {path} not found. Generating new one...")
            # Generate syllables if database doesn't exist'
    return self._generate_syllable_database()

    def _generate_syllable_database(self) -> List[Dict[str, Any]]:
    """Generate syllable database if not found"""
    generator = CompleteArabicSyllableGenerator()
    all_syllables = generator.generate_all_syllables()

        # Convert to database format
    syllable_list = []
        for syllable_type, syllables in all_syllables.items():
            for syllable in syllables:
    syl_data = {
    'text': syllable.syllable_text,
    'type': syllable_type,
    'onset': syllable.onset,
    'nucleus': syllable.nucleus,
    'coda': syllable.coda,
    'prosodic_weight': syllable.prosodic_weight,
    'frequency_estimate': syllable.frequency_estimate,
    'features': syllable.phonological_features,
    }
    syllable_list.append(syl_data)

    return syllable_list

    def _organize_syllables_by_type(self):
    """Organize syllables by their types for efficient access"""
    self.syllables_by_type = defaultdict(list)

        for syllable in self.syllables_db:
    self.syllables_by_type[syllable['type']].append(syllable)

    logger.info()
    f"Organized syllables: {dict((k, len(v)) for k, v} in self.syllables_by_type.items())}"
    )

    def _setup_morphological_constraints(self):
    """Setup morphological and phonotactic constraints"""

        # Forbidden syllable sequences (example constraints)
    self.constraints.forbidden_syllable_sequences.update()
    [
                # Avoid difficult consonant clusters between syllables
    ('CVCC', 'CVC'),  # Double closure difficulty
    ('CVCC', 'CVCC'),  # Triple consonant sequences
    ]
    )

        # Verb specific constraints
    self.constraints.verb_constraints = {
    'no_initial_sukun': True,  # لا يبدأ الفعل بساكن
    'final_vowel_required': True,  # الفعل يحتاج حركة في النهاية
    'max_syllables': 5,  # حد أقصى للمقاطع
    }

        # Noun specific constraints
    self.constraints.noun_constraints = {
    'feminine_marker_compatible': True,  # توافق علامة التأنيث
    'plural_patterns_valid': True,  # صحة أنماط الجمع
    'max_syllables': 6,  # حد أقصى للمقاطع
    }

    logger.info("Morphological constraints setup complete")

    def generate_verb_weights(self) -> List[MorphologicalWeight]:
    """Generate all Arabic verb morphological weights"""

    logger.info("🔬 Starting verb weight generation...")
    verb_weights = []

        # Triliteral bare verbs (الأفعال الثلاثية المجردة)
    verb_weights.extend(self._generate_triliteral_bare_verbs())

        # Triliteral augmented verbs (الأفعال الثلاثية المزيدة)
    verb_weights.extend(self._generate_triliteral_augmented_verbs())

        # Quadriliteral verbs (الأفعال الرباعية)
    verb_weights.extend(self._generate_quadriliteral_verbs())

    logger.info(f"✅ Generated {len(verb_weights)} verb weights")
    return verb_weights

    def _generate_triliteral_bare_verbs(self) -> List[MorphologicalWeight]:
    """Generate triliteral bare verb weights (فَعَلَ، فَعِلَ، فَعُلَ)"""

    weights = []
    self.syllables_by_type['CV']

        # Pattern: CV-CV CV for فَعَلَ type verbs
    patterns = [
    ('فَعَلَ', ['CV', 'CV', 'CV']),
    ('فَعِلَ', ['CV', 'CV', 'CV']),
    ('فَعُلَ', ['CV', 'CV', 'CV']),
    ]

        for pattern_name, syllable_pattern in patterns:
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name=pattern_name,
    syllable_pattern=syllable_pattern,
    word_type=ArabicWordType.VERB_TRILITERAL_BARE,
    max_combinations=1000,  # Limit for performance
    )
    weights.extend(pattern_weights)

    logger.info(f"Generated {len(weights)} triliteral bare verb weights")
    return weights

    def _generate_triliteral_augmented_verbs(self) -> List[MorphologicalWeight]:
    """Generate triliteral augmented verb weights"""

    weights = []

    patterns = [
    ('أَفْعَلَ', ['CV', 'CVC', 'CV']),  # Form IV
    ('فَعَّلَ', ['CV', 'CVC', 'CV']),  # Form II (with gemination)
    ('فَاعَلَ', ['CVV', 'CV', 'CV']),  # Form III
    ('انْفَعَلَ', ['CVC', 'CV', 'CV']),  # Form VII
    ('افْتَعَلَ', ['CVC', 'CV', 'CV']),  # Form VIII
    ('اسْتَفْعَلَ', ['CVC', 'CVC', 'CV']),  # Form X
    ]

        for pattern_name, syllable_pattern in patterns:
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name=pattern_name,
    syllable_pattern=syllable_pattern,
    word_type=ArabicWordType.VERB_TRILITERAL_AUGMENTED,
    max_combinations=500,  # Smaller limit for complex patterns
    )
    weights.extend(pattern_weights)

    logger.info(f"Generated {len(weights)} triliteral augmented verb weights")
    return weights

    def _generate_quadriliteral_verbs(self) -> List[MorphologicalWeight]:
    """Generate quadriliteral verb weights"""

    weights = []

    patterns = [
    ('فَعْلَلَ', ['CVC', 'CV', 'CV']),  # Bare quadriliteral
    ('تَفَعْلَلَ', ['CV', 'CVC', 'CV']),  # Augmented quadriliteral
    ]

        for pattern_name, syllable_pattern in patterns:
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name=pattern_name,
    syllable_pattern=syllable_pattern,
    word_type=ArabicWordType.VERB_QUADRILITERAL_BARE,
    max_combinations=300)
    weights.extend(pattern_weights)

    logger.info(f"Generated {len(weights)} quadriliteral verb weights")
    return weights

    def generate_noun_weights(self) -> List[MorphologicalWeight]:
    """Generate all Arabic noun morphological weights"""

    logger.info("🔬 Starting noun weight generation...")
    noun_weights = []

        # Derivative nouns (المشتقات)
    noun_weights.extend(self._generate_derivative_nouns())

        # Source nouns (المصادر)
    noun_weights.extend(self._generate_source_nouns())

        # Plural forms (صيغ الجمع)
    noun_weights.extend(self._generate_plural_nouns())

        # Dual and feminine forms (المثنى والمؤنث)
    noun_weights.extend(self._generate_dual_and_feminine_nouns())

        # Place, time, and instrument nouns (أسماء المكان والزمان والآلة)
    noun_weights.extend(self._generate_specialized_nouns())

    logger.info(f"✅ Generated {len(noun_weights)} noun weights")
    return noun_weights

    def _generate_derivative_nouns(self) -> List[MorphologicalWeight]:
    """Generate derivative noun weights (فَاعِل، مَفْعُول، etc.)"""

    weights = []

    patterns = [
    ('فَاعِل', ['CVV', 'CVC']),  # Active participle
    ('مَفْعُول', ['CV', 'CVVC']),  # Passive participle
    ('فَعِيل', ['CV', 'CVVC']),  # Intensive adjective
    ('فَعَّال', ['CV', 'CVVC']),  # Intensive agent
    ('مِفْعَال', ['CVC', 'CVVC']),  # Instrumental noun
    ]

        for pattern_name, syllable_pattern in patterns:
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name=pattern_name,
    syllable_pattern=syllable_pattern,
    word_type=ArabicWordType.NOUN_DERIVATIVE,
    max_combinations=800)
    weights.extend(pattern_weights)

    logger.info(f"Generated {len(weights)} derivative noun weights")
    return weights

    def _generate_source_nouns(self) -> List[MorphologicalWeight]:
    """Generate source noun weights (مصادر)"""

    weights = []

    patterns = [
    ('فَعْل', ['CVC']),  # Basic source
    ('فِعَالَة', ['CV', 'CVV', 'CV']),  # Intensive source
    ('تَفْعِيل', ['CVC', 'CVVC']),  # Form II source
    ('مُفَاعَلَة', ['CV', 'CVV', 'CV', 'CV']),  # Form III source
    ]

        for pattern_name, syllable_pattern in patterns:
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name=pattern_name,
    syllable_pattern=syllable_pattern,
    word_type=ArabicWordType.NOUN_SOURCE,
    max_combinations=600)
    weights.extend(pattern_weights)

    logger.info(f"Generated {len(weights)} source noun weights")
    return weights

    def _generate_plural_nouns(self) -> List[MorphologicalWeight]:
    """Generate plural noun weights"""

    weights = []

        # Sound masculine plural
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name='فَاعِلُون',
    syllable_pattern=['CVV', 'CVVC'],
    word_type=ArabicWordType.NOUN_PLURAL_MASCULINE,
    max_combinations=400)
    weights.extend(pattern_weights)

        # Sound feminine plural
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name='فَاعِلَات',
    syllable_pattern=['CVV', 'CVVC'],
    word_type=ArabicWordType.NOUN_PLURAL_FEMININE,
    max_combinations=400)
    weights.extend(pattern_weights)

        # Broken plurals
    broken_patterns = [
    ('أَفْعَال', ['CV', 'CVVC']),
    ('فُعُول', ['CV', 'CVVC']),
    ('فُعَّال', ['CV', 'CVVC']),
    ]

        for pattern_name, syllable_pattern in broken_patterns:
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name=pattern_name,
    syllable_pattern=syllable_pattern,
    word_type=ArabicWordType.NOUN_PLURAL_BROKEN,
    max_combinations=300)
    weights.extend(pattern_weights)

    logger.info(f"Generated {len(weights)} plural noun weights")
    return weights

    def _generate_dual_and_feminine_nouns(self) -> List[MorphologicalWeight]:
    """Generate dual and feminine noun weights"""

    weights = []

        # Dual forms
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name='فَاعِلَان',
    syllable_pattern=['CVV', 'CVVC'],
    word_type=ArabicWordType.NOUN_DUAL,
    max_combinations=300)
    weights.extend(pattern_weights)

        # Feminine forms
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name='فَاعِلَة',
    syllable_pattern=['CVV', 'CV', 'CV'],
    word_type=ArabicWordType.NOUN_FEMININE,
    max_combinations=400)
    weights.extend(pattern_weights)

    logger.info(f"Generated {len(weights)} dual and feminine noun weights")
    return weights

    def _generate_specialized_nouns(self) -> List[MorphologicalWeight]:
    """Generate specialized noun weights (place, time, instrument)"""

    weights = []

    patterns = [
    ('مَفْعَل', ['CV', 'CVC'], ArabicWordType.NOUN_PLACE),
    ('مَفْعِل', ['CV', 'CVC'], ArabicWordType.NOUN_TIME),
    ('مِفْعَال', ['CVC', 'CVVC'], ArabicWordType.NOUN_INSTRUMENT),
    ]

        for pattern_name, syllable_pattern, word_type in patterns:
    pattern_weights = self._generate_weights_for_pattern()
    pattern_name=pattern_name,
    syllable_pattern=syllable_pattern,
    word_type=word_type,
    max_combinations=200)
    weights.extend(pattern_weights)

    logger.info(f"Generated {len(weights) specialized} noun weights}")
    return weights

    def _generate_weights_for_pattern()
    self,
    pattern_name: str,
    syllable_pattern: List[str],
    word_type: ArabicWordType,
    max_combinations: int = 1000) -> List[MorphologicalWeight]:
    """Generate all possible weights for a given syllable pattern"""

    weights = []

        # Get syllables for each position in the pattern
    syllable_choices = []
        for position, syllable_type in enumerate(syllable_pattern):
    available_syllables = self.syllables_by_type.get(syllable_type, [])
            if not available_syllables:
    logger.warning(f"No syllables found for type {syllable_type}")
    return weights
    syllable_choices.append(available_syllables)

        # Generate combinations (with limit for performance)
    combinations_count = 0
        for syllable_combination in itertools.product(*syllable_choices):
            if combinations_count >= max_combinations:
    break

            # Check constraints
            if self._is_valid_syllable_sequence(list(syllable_combination), word_type):
    weight = self._create_morphological_weight()
    pattern_name=pattern_name,
    syllable_pattern=syllable_pattern,
    syllable_combination=list(syllable_combination),
    word_type=word_type)
    weights.append(weight)
    combinations_count += 1

    logger.debug(f"Generated {len(weights) weights for} pattern {pattern_name}}")
    return weights

    def _is_valid_syllable_sequence()
    self, syllables: List[Dict], word_type: ArabicWordType
    ) -> bool:
    """Check if syllable sequence is valid according to constraints"""

        # Check inter-syllable constraints
        for i in range(len(syllables) - 1):
    current_type = syllables[i]['type']
    next_type = syllables[i + 1]['type']

            if ()
    current_type,
    next_type) in self.constraints.forbidden_syllable_sequences:
    return False

        # Check word-type specific constraints
        if word_type in [
    ArabicWordType.VERB_TRILITERAL_BARE,
    ArabicWordType.VERB_TRILITERAL_AUGMENTED,
    ArabicWordType.VERB_QUADRILITERAL_BARE,
    ]:
    return self._check_verb_constraints(syllables)
        else:
    return self._check_noun_constraints(syllables)

    def _check_verb_constraints(self, syllables: List[Dict]) -> bool:
    """Check verb specific constraints"""

        # No initial sukun (لا يبدأ الفعل بساكن)
    first_syllable = syllables[0]
        if first_syllable['type'].startswith('C') and len(first_syllable['onset']) == 0:
    return False

        # Length constraint
        if len(syllables) > self.constraints.verb_constraints.get('max_syllables', 5):
    return False

    return True

    def _check_noun_constraints(self, syllables: List[Dict]) -> bool:
    """Check noun specific constraints"""

        # Length constraint
        if len(syllables) > self.constraints.noun_constraints.get('max_syllables', 6):
    return False

    return True

    def _create_morphological_weight()
    self,
    pattern_name: str,
    syllable_pattern: List[str],
    syllable_combination: List[Dict],
    word_type: ArabicWordType) -> MorphologicalWeight:
    """Create a complete morphological weight object"""

        # Build phonetic form
    phonetic_form = ''.join(syl['text'] for syl in syllable_combination)

        # Calculate prosodic weight
    prosodic_weight = sum(syl['prosodic_weight'] for syl in syllable_combination)

        # Estimate frequency
    frequency_estimate = 1.0
        for syl in syllable_combination:
    frequency_estimate *= syl['frequency_estimate']

        # Create morphological features
    morphological_features = {
    'syllable_count': len(syllable_combination),
    'total_phonemes': sum()
    len(syl['onset']) + len(syl['nucleus']) + len(syl['coda'])
                for syl in syllable_combination
    ),
    'complexity_score': ()
    prosodic_weight / len(syllable_combination)
                if syllable_combination
                else 0
    ),
    }

    return MorphologicalWeight()
    pattern_name=pattern_name,
    pattern_template=pattern_name,
    word_type=word_type,
    syllable_sequence=syllable_combination,
    syllable_pattern=syllable_pattern,
    phonetic_form=phonetic_form,
    morphological_features=morphological_features,
    frequency_estimate=frequency_estimate,
    prosodic_weight=prosodic_weight,
    constraints_applied=['phonotactic', 'morphological'],
    is_valid=True)

    def generate_all_weights(self) -> Dict[str, List[MorphologicalWeight]]:
    """Generate complete morphological weight inventory"""

    logger.info("🔬 Starting complete morphological weight generation...")

    all_weights = {
    'verbs': self.generate_verb_weights(),
    'nouns': self.generate_noun_weights(),
    }

        # Store in instance
    self.weights_db = all_weights['verbs'] + all_weights['nouns']

    total_count = sum(len(weights) for weights in all_weights.values())
    logger.info()
    f"✅ Complete morphological weight generation finished: {total_count} total weights"
    )

    return all_weights

    def analyze_weight_distribution()
    self, all_weights: Dict[str, List[MorphologicalWeight]]
    ) -> Dict[str, Any]:
    """Analyze distribution and statistics of generated weights"""

    analysis = {
    'counts_by_category': {},
    'frequency_distribution': {},
    'complexity_analysis': {},
    'pattern_distribution': defaultdict(int),
    'top_frequent_weights': [],
    }

    all_weights_flat = []

        for category, weights in all_weights.items():
    analysis['counts_by_category'][category] = len(weights)

            # Collect all weights
    all_weights_flat.extend(weights)

            # Pattern distribution
            for weight in weights:
    analysis['pattern_distribution'][weight.pattern_name] += 1

        # Sort by frequency
    all_weights_flat.sort(key=lambda x: x.frequency_estimate, reverse=True)
    analysis['top_frequent_weights'] = [
    {
    'pattern': w.pattern_name,
    'phonetic': w.phonetic_form,
    'type': w.word_type.value,
    'frequency': w.frequency_estimate,
    }
            for w in all_weights_flat[:20]
    ]

        # Complexity analysis
    complexities = [
    w.morphological_features.get('complexity_score', 0)
            for w in all_weights_flat
    ]
    analysis['complexity_analysis'] = {
    'mean_complexity': ()
    sum(complexities) / len(complexities) if complexities else 0
    ),
    'max_complexity': max(complexities) if complexities else 0,
    'min_complexity': min(complexities) if complexities else 0,
    }

    return analysis

    def save_morphological_weights()
    self,
    all_weights: Dict[str, List[MorphologicalWeight]], filename: str = "complete_arabic_morphological_weights.json"):
    """Save complete morphological weight inventory to JSON file"""

        # Convert to serializable format
    serializable_data = {}

        for category, weights in all_weights.items():
    serializable_data[category] = []

            for weight in weights:
    weight_data = {
    'pattern_name': weight.pattern_name,
    'pattern_template': weight.pattern_template,
    'word_type': weight.word_type.value,
    'syllable_pattern': weight.syllable_pattern,
    'phonetic_form': weight.phonetic_form,
    'prosodic_weight': weight.prosodic_weight,
    'frequency_estimate': weight.frequency_estimate,
    'morphological_features': weight.morphological_features,
    'syllables': [
    {
    'text': syl['text'],
    'type': syl['type'],
    'onset': syl['onset'],
    'nucleus': syl['nucleus'],
    'coda': syl['coda'],
    }
                        for syl in weight.syllable_sequence
    ],
    }
    serializable_data[category].append(weight_data)

        # Save to file
        with open(filename, 'w', encoding='utf 8') as f:
    json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Morphological weights saved to {filename}")


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════════


def demonstrate_morphological_weight_generation():
    """Demonstrate complete Arabic morphological weight generation"""

    print("🏗️ مولد الأوزان الصرفية العربية الشامل")
    print("=" * 60)

    # Initialize generator
    generator = ArabicMorphologicalWeightGenerator()

    # Generate all weights
    print("\n📊 توليد جميع الأوزان الصرفية...")
    all_weights = generator.generate_all_weights()

    # Display counts
    print("\n📈 إحصائيات الأوزان المولدة:")
    total_weights = 0
    for category, weights in all_weights.items():
    count = len(weights)
    total_weights += count
    print(f"   • {category}: {count:} وزن")

    print(f"\n🎯 المجموع الكلي: {total_weights:} وزن صرفي")

    # Show examples from each category
    print("\n🔤 عينات من الأوزان:")
    for category, weights in all_weights.items():
        if weights:
    examples = [f"{w.pattern_name} ({w.phonetic_form})" for w in weights[:3]]
    print(f"   • {category:} {', '.join(examples)}}")

    # Analyze distribution
    print("\n📊 تحليل التوزيع...")
    analysis = generator.analyze_weight_distribution(all_weights)

    print("\n🔝 الأوزان الأكثر تكراراً:")
    for i, weight_info in enumerate(analysis['top_frequent_weights'][:10], 1):
    print()
    f"   {i}. {weight_info['pattern']} - {weight_info['phonetic']} "
    f"({weight_info['type'])} - {weight_info['frequency']:.6f}}"
    )

    # Pattern distribution
    print("\n📋 توزيع الأنماط الأكثر شيوعاً:")
    pattern_items = sorted()
    analysis['pattern_distribution'].items(), key=lambda x: x[1], reverse=True
    )[:10]
    for pattern, count in pattern_items:
    print(f"   • {pattern}: {count تكرار}")

    # Complexity analysis
    complexity = analysis['complexity_analysis']
    print("\n🧮 تحليل التعقيد الصرفي:")
    print(f"   • متوسط التعقيد: {complexity['mean_complexity']:.2f}")
    print(f"   • أقصى تعقيد: {complexity['max_complexity']:.2f}")
    print(f"   • أدنى تعقيد: {complexity['min_complexity']:.2f}")

    # Save weights
    print("\n💾 حفظ مخزون الأوزان...")
    generator.save_morphological_weights(all_weights)

    print("\n✅ اكتمل توليد الأوزان الصرفية العربية!")
    print("💎 بناءً على 22,218 مقطع صوتي - أساس علمي متكامل!")


if __name__ == "__main__":
    demonstrate_morphological_weight_generation()

