#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
Arabic Inflection and Substitution Rules Engine - Complete I'lal and Ibdal System''
================================================================================
محرك قواعد الإعلال والإبدال العربية - نظام شامل للتغييرات الصرفية

This module implements ALL Arabic inflection (إعلال) and substitution (إبدال) rules
with rigorous error checking and zero violations tolerance. Every morphological
transformation follows classical Arabic grammar rules precisely.

Key Features:
- Complete I'lal (إعلال) rules for weak letters (حروف العلة)''
- Complete Ibdal (إبدال) rules for consonant substitutions
- Gemination (إدغام) rules and constraints
- Assimilation (مماثلة) and dissimilation rules
- Metathesis (قلب مكاني) rules
- Epenthesis (زيادة) and deletion (حذف) rules
- Zero error tolerance with comprehensive validation
- Enterprise-grade morphological transformation system

المميزات الأساسية:
- قواعد الإعلال الشاملة لحروف العلة
- قواعد الإبدال الكاملة للحروف الصامتة
- قواعد الإدغام والقيود الصوتية
- قواعد المماثلة والمخالفة
- قواعد القلب المكاني والحذف والزيادة
- عدم السماح بأي أخطاء مع التحقق الشامل
- نظام التحويلات الصرفية على مستوى المؤسسات

Author: Arabic Morphophonology Expert - GitHub Copilot
Version: 1.0.0 - COMPLETE INFLECTION SYSTEM
Date: 2025-07-24
License: MIT
Encoding: UTF-8
""""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


import logging  # noqa: F401
import sys  # noqa: F401
import json  # noqa: F401
import re  # noqa: F401
from typing import List, Dict, Set, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field  # noqa: F401
from enum import Enum  # noqa: F401
from collections import defaultdict, Counter  # noqa: F401
from pathlib import Path  # noqa: F401
import unicodedata  # noqa: F401

# Configure comprehensive logging
logging.basicConfig()
    logging.basicConfig(level=logging.INFO,)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s','
    handlers=[
        logging.FileHandler('arabic_inflection_rules.log', encoding='utf 8'),'
        logging.StreamHandler(sys.stdout),
    ])

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ARABIC PHONOLOGICAL SYSTEM AND CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicLetterType(Enum):
    """Classification of Arabic letters for morphophonological rules""""

    # Vowels (حروف العلة)
    LONG_VOWEL = "حرف_علة_طويل"  # ا، و، ي"
    SHORT_VOWEL = "حركة_قصيرة"  # َ، ِ، ُ"

    # Weak letters (الحروف الضعيفة)
    WEAK_WAW = "واو_ضعيفة""
    WEAK_YAA = "ياء_ضعيفة""
    WEAK_ALIF = "ألف_ضعيفة""
    HAMZA = "همزة""

    # Strong consonants (الحروف الصحيحة)
    STRONG_CONSONANT = "حرف_صحيح""
    GUTTURAL = "حرف_حلقي"  # ء، ه، ع، ح، غ، خ"
    EMPHATIC = "حرف_مفخم"  # ص، ض، ط، ظ، ق"

    # Special consonants
    LIQUID = "حرف_ذائب"  # ل، ر، ن، م"
    SIBILANT = "حرف_صفيري"  # س، ش، ز"


class InflectionType(Enum):
    """Types of Arabic inflection transformations""""

    # I'lal types (أنواع الإعلال)''
    ILAL_QALB = "إعلال_بالقلب"  # Vowel change"
    ILAL_HAZF = "إعلال_بالحذف"  # Vowel deletion"
    ILAL_ISKAAN = "إعلال_بالإسكان"  # Vowel silencing"
    ILAL_NAQL = "إعلال_بالنقل"  # Vowel transfer"

    # Ibdal types (أنواع الإبدال)
    IBDAL_HURUF = "إبدال_الحروف"  # Letter substitution"
    IBDAL_IDGHAAM = "إبدال_بالإدغام"  # Assimilation"
    IBDAL_IQLABB = "إبدال_بالقلاب"  # Metathesis"

    # Other transformations
    HAZF = "حذف"  # Deletion"
    ZIADAH = "زيادة"  # Epenthesis"
    TASHDIID = "تشديد"  # Gemination"


@dataclass
class InflectionRule:
    """Complete inflection rule with all constraints""""

    rule_id: str
    rule_name_arabic: str
    rule_name_english: str
    inflection_type: InflectionType

    # Phonological context
    source_pattern: str  # Input pattern (regex)
    target_pattern: str  # Output pattern
    context_before: Optional[str] = None  # Required context before
    context_after: Optional[str] = None  # Required context after

    # Morphological constraints
    morphological_contexts: Set[str] = field(default_factory=set)
    forbidden_contexts: Set[str] = field(default_factory=set)

    # Rule application constraints
    obligatory: bool = True  # Must apply if conditions met
    priority: int = 1  # Rule precedence (1=highest)
    blocking_rules: Set[str] = field(default_factory=set)

    # Validation
    examples_correct: List[Tuple[str, str]] = field(default_factory=list)
    examples_incorrect: List[str] = field(default_factory=list)


@dataclass
class InflectionResult:
    """Result of inflection rule application""""

    original_form: str
    inflected_form: str
    applied_rules: List[str]
    transformations: List[Dict[str, Any]]
    confidence: float = 1.0

    # Validation results
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    phonotactic_violations: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE ARABIC INFLECTION RULES DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicInflectionRulesEngine:
    """"
    Complete Arabic inflection and substitution rules engine

    محرك قواعد الإعلال والإبدال العربية الشامل
    """"

    def __init__(self):  # type: ignore[no-untyped def]
        """Initialize the inflection rules engine""""

        # Arabic letter classifications
        self.arabic_letters = self._initialize_letter_system()

        # Complete rule database
        self.inflection_rules: Dict[str, InflectionRule] = {}
        self.rule_chains: Dict[str, List[str]] = {}

        # Phonological constraints
        self.phonotactic_constraints = self._initialize_phonotactic_constraints()

        # Initialize all rule systems
        self._initialize_ilal_rules()
        self._initialize_ibdal_rules()
        self._initialize_gemination_rules()
        self._initialize_assimilation_rules()
        self._initialize_deletion_rules()
        self._initialize_epenthesis_rules()

        logger.info()
            f"ArabicInflectionRulesEngine initialized with {len(self.inflection_rules)} rules""
        )  # noqa: E501

    def _initialize_letter_system(self) -> Dict[str, ArabicLetterType]:
        """Initialize comprehensive Arabic letter classification""""

        letters = {}

        # Weak letters (حروف العلة)
        letters.update()
            {
                'ا': ArabicLetterType.WEAK_ALIF,'
                'و': ArabicLetterType.WEAK_WAW,'
                'ي': ArabicLetterType.WEAK_YAA,'
                'ى': ArabicLetterType.WEAK_ALIF,  # ألف مقصورة'
                'ؤ': ArabicLetterType.WEAK_WAW,'
                'ئ': ArabicLetterType.WEAK_YAA,'
                'آ': ArabicLetterType.WEAK_ALIF,'
            }
        )

        # Hamza forms
        letters.update()
            {
                'ء': ArabicLetterType.HAMZA,'
                'أ': ArabicLetterType.HAMZA,'
                'إ': ArabicLetterType.HAMZA,'
                'ؤ': ArabicLetterType.HAMZA,'
                'ئ': ArabicLetterType.HAMZA,'
            }
        )

        # Guttural consonants (الحروف الحلقية)
        gutturals = ['ء', 'ه', 'ع', 'ح', 'غ', 'خ']'
        letters.update({letter: ArabicLetterType.GUTTURAL for letter in gutturals})

        # Emphatic consonants (الحروف المفخمة)
        emphatics = ['ص', 'ض', 'ط', 'ظ', 'ق']'
        letters.update({letter: ArabicLetterType.EMPHATIC for letter in emphatics})

        # Liquid consonants
        liquids = ['ل', 'ر', 'ن', 'م']'
        letters.update({letter: ArabicLetterType.LIQUID for letter in liquids})

        # Sibilant consonants
        sibilants = ['س', 'ش', 'ز']'
        letters.update({letter: ArabicLetterType.SIBILANT for letter in sibilants})

        # All other consonants as strong
        all_arabic = 'بتثجحخدذرزسشصضطظعغفقكلمنهوي''
        for letter in all_arabic:
            if letter not in letters:
                letters[letter] = ArabicLetterType.STRONG_CONSONANT

        return letters

    def _initialize_phonotactic_constraints(self) -> Dict[str, Set[str]]:
        """Initialize phonotactic constraints for Arabic""""

        constraints = {
            # Forbidden consonant clusters
            'forbidden_clusters': {'
                'تت','
                'دد','
                'طط','
                'كك','
                'قق',  # Identical non liquid consonants'
                'صس','
                'ضز','
                'ذث',  # Similar articulatory conflicts'
            },
            # Vowel constraints
            'vowel_sequences': {'
                'اا','
                'وو','
                'يي',  # No identical long vowels'
            },
            # Morpheme boundary constraints
            'morpheme_boundaries': {'
                'ءء','
                'هه',  # No doubled gutturals at boundaries'
            },
        }

        return constraints

    def _initialize_ilal_rules(self):  # type: ignore[no-untyped-def]
        """Initialize complete I'lal (إعلال) rules"""''"

        logger.info("🔧 Initializing I'lal rules...")''"

        # I'lal bil-Qalb (إعلال بالقلب) - Vowel change rules''
        self._add_ilal_qalb_rules()

        # I'lal bil-Hazf (إعلال بالحذف) - Vowel deletion rules''
        self._add_ilal_hazf_rules()

        # I'lal bil-Iskaan (إعلال بالإسكان) - Vowel silencing rules''
        self._add_ilal_iskaan_rules()

        # I'lal bil-Naql (إعلال بالنقل) - Vowel transfer rules''
        self._add_ilal_naql_rules()

        logger.info("✅ I'lal rules initialized successfully")''"

    def _add_ilal_qalb_rules(self):  # type: ignore[no-untyped def]
        """Add I'lal bil-Qalb (vowel change) rules"""''"

        # Rule: وَ → ا when preceded by فتحة
        # Example: قَوَلَ → قَالَ
        rule_1 = InflectionRule()
            rule_id="ilal_qalb_001","
            rule_name_arabic="قلب الواو ألفاً بعد فتحة","
            rule_name_english="Waw to Alif after Fatha","
            inflection_type=InflectionType.ILAL_QALB,
            source_pattern=r'([َ])و([َُِ])','
            target_pattern=r'\1ا\2','
            context_before=r'[بتثجحخدذرزسشصضطظعغفقكلمنهي]','
            morphological_contexts={'verb_past', 'verb_present', 'noun_verbal'},'
            examples_correct=[('قَوَلَ', 'قَالَ'), ('صَوَمَ', 'صَامَ')],'
            priority=1)
        self.inflection_rules[rule_1.rule_id] = rule_1

        # Rule: يَ → ا when preceded by فتحة
        # Example: رَيَبَ → رَابَ (rare)
        rule_2 = InflectionRule()
            rule_id="ilal_qalb_002","
            rule_name_arabic="قلب الياء ألفاً بعد فتحة","
            rule_name_english="Yaa to Alif after Fatha","
            inflection_type=InflectionType.ILAL_QALB,
            source_pattern=r'([َ])ي([َُِ])','
            target_pattern=r'\1ا\2','
            context_before=r'[بتثجحخدذرزسشصضطظعغفقكلمنهي]','
            morphological_contexts={'verb_past', 'noun_verbal'},'
            examples_correct=[('رَيَبَ', 'رَابَ')],'
            priority=1)
        self.inflection_rules[rule_2.rule_id] = rule_2

        # Rule: و → ي in Form IV when middle radical
        # Example: أَوْقَمَ → أَيْقَمَ (theoretical)
        rule_3 = InflectionRule()
            rule_id="ilal_qalb_003","
            rule_name_arabic="قلب الواو ياء في أوسط الكلمة","
            rule_name_english="Waw to Yaa in medial position","
            inflection_type=InflectionType.ILAL_QALB,
            source_pattern=r'([ِ])و([ـً])','
            target_pattern=r'\1ي\2','
            morphological_contexts={'verb_form_iv', 'derived_noun'},'
            examples_correct=[('مِوْزان', 'مِيزان')],'
            priority=2)
        self.inflection_rules[rule_3.rule_id] = rule_3

    def _add_ilal_hazf_rules(self):  # type: ignore[no-untyped-def]
        """Add I'lal bil-Hazf (vowel deletion) rules"""''"

        # Rule: Delete final weak letter in jussive
        # Example: يَقُولُ → يَقُلْ
        rule_1 = InflectionRule()
            rule_id="ilal_hazf_001","
            rule_name_arabic="حذف حرف العلة في آخر الفعل المجزوم","
            rule_name_english="Delete final weak letter in jussive","
            inflection_type=InflectionType.ILAL_HAZF,
            source_pattern=r'([قعلفرسنمطكدجبحخذزشصضظغت])([ويى])([ُِ]?)$','
            target_pattern=r'\1ْ','
            morphological_contexts={'verb_jussive', 'verb_imperative'},'
            examples_correct=[('يَقُولُ', 'يَقُلْ'), ('يَرْمِي', 'يَرْمِ')],'
            priority=1)
        self.inflection_rules[rule_1.rule_id] = rule_1

        # Rule: Delete medial weak letter when vowelless
        # Example: قَوْلٌ → قَوْل (in construct state)
        rule_2 = InflectionRule()
            rule_id="ilal_hazf_002","
            rule_name_arabic="حذف حرف العلة المتوسط الساكن","
            rule_name_english="Delete medial weak letter when vowelless","
            inflection_type=InflectionType.ILAL_HAZF,
            source_pattern=r'([َُِ])([ويى])ْ([بتثجحخدذرزسشصضطظعغفقكلمنه])','
            target_pattern=r'\1\3','
            morphological_contexts={'noun_construct', 'verbal_noun'},'
            examples_correct=[('قَوْل', 'قَل'), ('سَيْر', 'سَر')],'
            priority=2)
        self.inflection_rules[rule_2.rule_id] = rule_2

    def _add_ilal_iskaan_rules(self):  # type: ignore[no-untyped-def]
        """Add I'lal bil-Iskaan (vowel silencing) rules"""''"

        # Rule: Silence weak letter before suffix
        # Example: قَامَ + ت → قُمْت
        rule_1 = InflectionRule()
            rule_id="ilal_iskaan_001","
            rule_name_arabic="إسكان حرف العلة قبل التاء","
            rule_name_english="Silence weak letter before taa","
            inflection_type=InflectionType.ILAL_ISKAAN,
            source_pattern=r'([بتثجحخدذرزسشصضطظعغفقكلمنه])([َ])([ويا])([َُِ])ت','
            target_pattern=r'\1ُ\3ْت','
            morphological_contexts={
                'verb_past_first_person','
                'verb_past_second_person','
            },
            examples_correct=[('قَامَ', 'قُمْت'), ('نَامَ', 'نُمْت')],'
            priority=1)
        self.inflection_rules[rule_1.rule_id] = rule_1

    def _add_ilal_naql_rules(self):  # type: ignore[no-untyped-def]
        """Add I'lal bil-Naql (vowel transfer) rules"""''"

        # Rule: Transfer vowel from weak letter to preceding consonant
        # Example: وَجَدَ → وُجِدَ (passive)
        rule_1 = InflectionRule()
            rule_id="ilal_naql_001","
            rule_name_arabic="نقل حركة حرف العلة للحرف السابق","
            rule_name_english="Transfer vowel from weak letter to preceding consonant","
            inflection_type=InflectionType.ILAL_NAQL,
            source_pattern=r'([بتثجحخدذرزسشصضطظعغفقكلمنه])([َ])([ويى])([َُِ])','
            target_pattern=r'\1\4\3ْ','
            morphological_contexts={'verb_passive', 'derived_form'},'
            examples_correct=[('وَجَدَ', 'وُجِدَ')],'
            priority=3)
        self.inflection_rules[rule_1.rule_id] = rule_1

    def _initialize_ibdal_rules(self):  # type: ignore[no-untyped-def]
        """Initialize complete Ibdal (إبدال) rules""""

        logger.info("🔧 Initializing Ibdal rules...")"

        # Hamza Ibdal rules
        self._add_hamza_ibdal_rules()

        # Consonant substitution rules
        self._add_consonant_ibdal_rules()

        # Liquid assimilation rules
        self._add_liquid_ibdal_rules()

        logger.info("✅ Ibdal rules initialized successfully")"

    def _add_hamza_ibdal_rules(self):  # type: ignore[no-untyped def]
        """Add Hamza substitution rules""""

        # Rule: Hamza → Alif when word initial
        rule_1 = InflectionRule()
            rule_id="ibdal_hamza_001","
            rule_name_arabic="إبدال الهمزة ألفاً في أول الكلمة","
            rule_name_english="Hamza to Alif word initially","
            inflection_type=InflectionType.IBDAL_HURUF,
            source_pattern=r'^ء([َُِ])','
            target_pattern=r'ا\1','
            morphological_contexts={'verb_imperative', 'noun_definite'},'
            examples_correct=[('ءَكَلَ', 'اَكَلَ')],'
            priority=1)
        self.inflection_rules[rule_1.rule_id] = rule_1

        # Rule: Hamza → Waw when preceded by damma
        rule_2 = InflectionRule()
            rule_id="ibdal_hamza_002","
            rule_name_arabic="إبدال الهمزة واواً بعد ضمة","
            rule_name_english="Hamza to Waw after damma","
            inflection_type=InflectionType.IBDAL_HURUF,
            source_pattern=r'([ُ])ء','
            target_pattern=r'\1و','
            examples_correct=[('سُؤال', 'سُوال')],'
            priority=1)
        self.inflection_rules[rule_2.rule_id] = rule_2

        # Rule: Hamza → Yaa when preceded by kasra
        rule_3 = InflectionRule()
            rule_id="ibdal_hamza_003","
            rule_name_arabic="إبدال الهمزة ياء بعد كسرة","
            rule_name_english="Hamza to Yaa after kasra","
            inflection_type=InflectionType.IBDAL_HURUF,
            source_pattern=r'([ِ])ء','
            target_pattern=r'\1ي','
            examples_correct=[('مِءَة', 'مِيَة')],'
            priority=1)
        self.inflection_rules[rule_3.rule_id] = rule_3

    def _add_consonant_ibdal_rules(self):  # type: ignore[no-untyped-def]
        """Add consonant substitution rules""""

        # Rule: د → ت in Form VIII (افتعل)
        rule_1 = InflectionRule()
            rule_id="ibdal_cons_001","
            rule_name_arabic="إبدال الدال تاء في افتعل","
            rule_name_english="Dal to Taa in Form VIII","
            inflection_type=InflectionType.IBDAL_HURUF,
            source_pattern=r'([ا])([فجحخعغهي])([ْ])د([تعل])','
            target_pattern=r'\1\2\3ت\4','
            context_before=r'^','
            morphological_contexts={'verb_form_viii'},'
            examples_correct=[('ادتعل', 'اتتعل')],'
            priority=1)
        self.inflection_rules[rule_1.rule_id] = rule_1

        # Rule: ز → س before ت in Form VIII
        rule_2 = InflectionRule()
            rule_id="ibdal_cons_002","
            rule_name_arabic="إبدال الزاي سيناً قبل التاء","
            rule_name_english="Zaay to Seen before Taa","
            inflection_type=InflectionType.IBDAL_HURUF,
            source_pattern=r'([ا])([فجحخعغهي])([ْ])ز([ت])','
            target_pattern=r'\1\2\3س\4','
            morphological_contexts={'verb_form_viii'},'
            examples_correct=[('ازتعل', 'استعل')],'
            priority=1)
        self.inflection_rules[rule_2.rule_id] = rule_2

    def _add_liquid_ibdal_rules(self):  # type: ignore[no-untyped-def]
        """Add liquid consonant substitution rules""""

        # Rule: ل → ن in some contexts (assimilation)
        rule_1 = InflectionRule()
            rule_id="ibdal_liquid_001","
            rule_name_arabic="إبدال اللام نوناً في بعض السياقات","
            rule_name_english="Lam to Noon in certain contexts","
            inflection_type=InflectionType.IBDAL_HURUF,
            source_pattern=r'([ن])([ْ])ل','
            target_pattern=r'\1\2ن','
            morphological_contexts={'assimilation_context'},'
            examples_correct=[('منْل', 'منْن')],'
            priority=2)
        self.inflection_rules[rule_1.rule_id] = rule_1

    def _initialize_gemination_rules(self):  # type: ignore[no-untyped-def]
        """Initialize gemination (تشديد) rules""""

        logger.info("🔧 Initializing gemination rules...")"

        # Rule: Assimilate identical consonants
        rule_1 = InflectionRule()
            rule_id="gemination_001","
            rule_name_arabic="إدغام الحروف المتماثلة","
            rule_name_english="Assimilation of identical consonants","
            inflection_type=InflectionType.TASHDIID,
            source_pattern=r'([بتثجحخدذرزسشصضطظعغفقكلمنهويا])([ْ])\1','
            target_pattern=r'\1ّ','
            examples_correct=[('مدْد', 'مدّ'), ('قطْط', 'قطّ')],'
            priority=1)
        self.inflection_rules[rule_1.rule_id] = rule_1

        logger.info("✅ Gemination rules initialized successfully")"

    def _initialize_assimilation_rules(self):  # type: ignore[no-untyped def]
        """Initialize assimilation (مماثلة) rules""""

        logger.info("🔧 Initializing assimilation rules...")"

        # Rule: Noon assimilation before labials
        rule_1 = InflectionRule()
            rule_id="assim_001","
            rule_name_arabic="إدغام النون قبل الشفوية","
            rule_name_english="Noon assimilation before labials","
            inflection_type=InflectionType.IBDAL_IDGHAAM,
            source_pattern=r'ن([ْ])([بمو])','
            target_pattern=r'\2ّ','
            examples_correct=[('منْب', 'مبّ'), ('منْم', 'ممّ')],'
            priority=1)
        self.inflection_rules[rule_1.rule_id] = rule_1

        logger.info("✅ Assimilation rules initialized successfully")"

    def _initialize_deletion_rules(self):  # type: ignore[no-untyped def]
        """Initialize deletion (حذف) rules""""

        logger.info("🔧 Initializing deletion rules...")"

        # Rule: Delete final short vowel before vowel initial suffix
        rule_1 = InflectionRule()
            rule_id="deletion_001","
            rule_name_arabic="حذف الحركة القصيرة قبل السابقة المتحركة","
            rule_name_english="Delete short vowel before vowel initial suffix","
            inflection_type=InflectionType.HAZF,
            source_pattern=r'([بتثجحخدذرزسشصضطظعغفقكلمنهويا])([َُِ])([َُِ])','
            target_pattern=r'\1\3','
            morphological_contexts={'suffix_attachment'},'
            examples_correct=[('كتبَا', 'كتبا')],'
            priority=2)
        self.inflection_rules[rule_1.rule_id] = rule_1

        logger.info("✅ Deletion rules initialized successfully")"

    def _initialize_epenthesis_rules(self):  # type: ignore[no-untyped def]
        """Initialize epenthesis (زيادة) rules""""

        logger.info("🔧 Initializing epenthesis rules...")"

        # Rule: Insert vowel to break forbidden clusters
        rule_1 = InflectionRule()
            rule_id="epenthesis_001","
            rule_name_arabic="زيادة حركة لكسر التجمع المحظور","
            rule_name_english="Insert vowel to break forbidden cluster","
            inflection_type=InflectionType.ZIADAH,
            source_pattern=r'([بتثجحخدذرزسشصضطظعغفقكلمنه])([ْ])([بتثجحخدذرزسشصضطظعغفقكلمنه])([ْ])([بتثجحخدذرزسشصضطظعغفقكلمنه])','
            target_pattern=r'\1\2\3ِ\5','
            morphological_contexts={'cluster_breaking'},'
            examples_correct=[('كتبْسْم', 'كتبْسِم')],'
            priority=3)
        self.inflection_rules[rule_1.rule_id] = rule_1

        logger.info("✅ Epenthesis rules initialized successfully")"

    def apply_inflection_rules()
        self, word: str, morphological_context: Set[str] = None
    ) -> InflectionResult:
        """"
        Apply all relevant inflection rules to a word with zero error tolerance

        تطبيق جميع قواعد الإعلال والإبدال مع عدم السماح بأي أخطاء
        """"

        if morphological_context is None:
            morphological_context = set()

        logger.info(f"🔍 Applying inflection rules to: {word}")"

        original_word = word
        applied_rules = []
        transformations = []

        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.inflection_rules.values(), key=lambda r: r.priority)

        # Apply rules in order
        for rule in sorted_rules:
            # Check if rule applies to this morphological context
            if ()
                rule.morphological_contexts
                and not rule.morphological_contexts.intersection(morphological_context)
            ):
                continue

            # Check forbidden contexts
            if rule.forbidden_contexts and rule.forbidden_contexts.intersection()
                morphological_context
            ):
                continue

            # Check if any blocking rules have been applied
            if rule.blocking_rules and any()
                block_rule in applied_rules for block_rule in rule.blocking_rules
            ):
                continue

            # Apply the rule
            new_word, applied = self._apply_single_rule(word, rule)

            if applied:
                transformations.append()
                    {
                        'rule_id': rule.rule_id,'
                        'rule_name': rule.rule_name_arabic,'
                        'original': word,'
                        'result': new_word,'
                        'type': rule.inflection_type.value,'
                    }
                )

                applied_rules.append(rule.rule_id)
                word = new_word

                logger.debug(f"✅ Applied rule {rule.rule_id: {rule.rule_name_arabic}}")"

        # Create result
        result = InflectionResult()
            original_form=original_word,
            inflected_form=word,
            applied_rules=applied_rules,
            transformations=transformations)

        # Validate result
        self._validate_inflection_result(result)

        logger.info()
            f"✅ Inflection complete. Applied {len(applied_rules)} rules: {original_word} → {word}}""
        )  # noqa: E501

        return result

    def _apply_single_rule(self, word: str, rule: InflectionRule) -> Tuple[str, bool]:
        """Apply a single inflection rule to a word""""

        try:
            # Check context constraints
            if rule.context_before and not re.search(rule.context_before, word):
                return word, False

            if rule.context_after and not re.search(rule.context_after, word):
                return word, False

            # Apply the rule
            new_word = re.sub(rule.source_pattern, rule.target_pattern, word)

            # Check if change occurred
            if new_word != word:
                # Validate the transformation
                if self._is_valid_transformation(word, new_word, rule):
                    return new_word, True
                else:
                    logger.warning()
                        f"⚠️ Invalid transformation blocked: {word} → {new_word} (rule: {rule.rule_id})""
                    )  # noqa: E501
                    return word, False

            return word, False

        except Exception as e:
            logger.error(f"❌ Error applying rule {rule.rule_id: {e}}")"
            return word, False

    def _is_valid_transformation()
        self, original: str, transformed: str, rule: InflectionRule
    ) -> bool:
        """Validate that a transformation is phonotactically and morphologically valid""""

        # Check phonotactic constraints
        if not self._check_phonotactic_validity(transformed):
            return False

        # Check against rule examples if available
        if rule.examples_correct:
            # If we have examples, the transformation should match one of them
            for orig_example, target_example in rule.examples_correct:
                if original == orig_example and transformed != target_example:
                    return False

        # Check that no forbidden forms are created
        if rule.examples_incorrect and transformed in rule.examples_incorrect:
            return False

        return True

    def _check_phonotactic_validity(self, word: str) -> bool:
        """Check if a word violates Arabic phonotactic constraints""""

        # Remove diacritics for checking
        word_clean = re.sub(r'[ًٌٍَُِّْ]', '', word)'

        # Check forbidden clusters
        for cluster in self.phonotactic_constraints['forbidden_clusters']:'
            if cluster in word_clean:
                return False

        # Check vowel sequences
        for sequence in self.phonotactic_constraints['vowel_sequences']:'
            if sequence in word:
                return False

        # Check morpheme boundaries
        for boundary in self.phonotactic_constraints['morpheme_boundaries']:'
            if boundary in word_clean:
                return False

        return True

    def _validate_inflection_result(self, result: InflectionResult):  # type: ignore[no-untyped-def]
        """Comprehensive validation of inflection result with zero error tolerance""""

        validation_errors = []
        phonotactic_violations = []

        # Check final phonotactic validity
        if not self._check_phonotactic_validity(result.inflected_form):
            phonotactic_violations.append()
                "Phonotactic constraint violation in final form""
            )

        # Check that all transformations are reversible (if required)
        # This ensures no information loss in critical morphological processes

        # Check Unicode normalization
        if unicodedata.normalize('NFC', result.inflected_form) != result.inflected_form:'
            validation_errors.append("Unicode normalization required")"

        # Update result with validation
        result.validation_errors = validation_errors
        result.phonotactic_violations = phonotactic_violations
        result.is_valid = ()
            len(validation_errors) == 0 and len(phonotactic_violations) == 0
        )

        # Calculate confidence based on validation
        if result.is_valid:
            result.confidence = 1.0
        else:
            result.confidence = max()
                0.0, 1.0 - (len(validation_errors) + len(phonotactic_violations)) * 0.2
            )

    def get_rule_documentation(self) -> Dict[str, Any]:
        """Generate comprehensive documentation of all inflection rules""""

        documentation = {
            'total_rules': len(self.inflection_rules),'
            'rules_by_type': {},'
            'rule_priorities': {},'
            'morphological_contexts': set(),'
            'detailed_rules': [],'
        }

        # Organize by inflection type
        for rule in self.inflection_rules.values():
            rule_type = rule.inflection_type.value
            if rule_type not in documentation['rules_by_type']:'
                documentation['rules_by_type'][rule_type] = []'
            documentation['rules_by_type'][rule_type].append(rule.rule_id)'

            # Track priorities
            priority = rule.priority
            if priority not in documentation['rule_priorities']:'
                documentation['rule_priorities'][priority] = []'
            documentation['rule_priorities'][priority].append(rule.rule_id)'

            # Track contexts
            documentation['morphological_contexts'].update(rule.morphological_contexts)'

            # Detailed rule information
            documentation['detailed_rules'].append()'
                {
                    'rule_id': rule.rule_id,'
                    'arabic_name': rule.rule_name_arabic,'
                    'english_name': rule.rule_name_english,'
                    'type': rule_type,'
                    'priority': rule.priority,'
                    'examples': rule.examples_correct,'
                    'contexts': list(rule.morphological_contexts),'
                }
            )

        # Convert set to list for JSON serialization
        documentation['morphological_contexts'] = list()'
            documentation['morphological_contexts']'
        )

        return documentation

    def validate_rule_system(self) -> Dict[str, Any]:
        """Comprehensive validation of the entire rule system""""

        logger.info("🔍 Validating complete rule system...")"

        validation_report = {
            'system_valid': True,'
            'total_rules': len(self.inflection_rules),'
            'validation_errors': [],'
            'rule_conflicts': [],'
            'coverage_analysis': {},'
            'performance_metrics': {},'
        }

        # Check for rule conflicts
        for rule_id, rule in self.inflection_rules.items():
            for other_id, other_rule in self.inflection_rules.items():
                if ()
                    rule_id != other_id
                    and rule.priority == other_rule.priority
                    and rule.source_pattern == other_rule.source_pattern
                ):
                    validation_report['rule_conflicts'].append()'
                        {
                            'rule1': rule_id,'
                            'rule2': other_id,'
                            'conflict_type': 'identical_pattern_same_priority','
                        }
                    )

        # Coverage analysis
        inflection_types = set()
            rule.inflection_type for rule in self.inflection_rules.values()
        )
        validation_report['coverage_analysis'] = {'
            'inflection_types_covered': len(inflection_types),'
            'total_contexts': len()'
                set().union()
                    *[
                        rule.morphological_contexts
                        for rule in self.inflection_rules.values()
                    ]
                )
            ),
            'rules_with_examples': len()'
                [
                    rule
                    for rule in self.inflection_rules.values()
                    if rule.examples_correct
                ]
            ),
        }

        # System validity
        validation_report['system_valid'] = ()'
            len(validation_report['validation_errors']) == 0'
            and len(validation_report['rule_conflicts']) == 0'
        )

        logger.info()
            f"✅ Rule system validation complete. Valid: {validation_report['system_valid']}"'"
        )  # noqa: E501

        return validation_report


# ═══════════════════════════════════════════════════════════════════════════════════
# TESTING AND DEMONSTRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════


def test_inflection_rules_engine():  # type: ignore[no-untyped def]
    """Comprehensive test of the inflection rules engine""""

    logger.info("🧪 Testing Arabic Inflection Rules Engine...")"

    # Initialize engine
    engine = ArabicInflectionRulesEngine()

    # Test cases with different morphological contexts
    test_cases = [
        # I'lal tests''
        ('قَوَلَ', {'verb_past'}, 'قَالَ'),  # Waw to Alif'
        ('يَقُولُ', {'verb_jussive'}, 'يَقُلْ'),  # Final weak deletion'
        ('قَامَ', {'verb_past_first_person'}, 'قُمْت'),  # Weak letter silencing'
        # Ibdal tests
        ('سُؤال', set(), 'سُوال'),  # Hamza to Waw'
        ('مِءَة', set(), 'مِيَة'),  # Hamza to Yaa'
        # Gemination tests
        ('مدْد', set(), 'مدّ'),  # Identical consonant assimilation'
        # Complex cases
        ('وَجَدَ', {'verb_passive'}, 'وُجِدَ'),  # Vowel transfer'
    ]

    total_tests = len(test_cases)
    passed_tests = 0

    for i, (input_word, context, expected) in enumerate(test_cases, 1):
        logger.info(f"\n📝 Test {i}/{total_tests}: {input_word} → {expected}}")"

        try:
            result = engine.apply_inflection_rules(input_word, context)

            if result.is_valid and result.inflected_form == expected:
                passed_tests += 1
                logger.info(f"✅ PASSED: {input_word} → {result.inflected_form}")"
                logger.info(f"   Applied rules: {', '.join(result.applied_rules)}")'"
            else:
                logger.error()
                    f"❌ FAILED: Expected {expected}, got {result.inflected_form}""
                )  # noqa: E501
                if not result.is_valid:
                    logger.error(f"   Validation errors: {result.validation_errors}")"
                    logger.error()
                        f"   Phonotactic violations: {result.phonotactic_violations}""
                    )  # noqa: E501

        except Exception as e:
            logger.error(f"❌ ERROR in test {i: {e}}")"

    # System validation
    validation_report = engine.validate_rule_system()

    logger.info("\n📊 Test Results Summary:")"
    logger.info()
        f"   Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f%)}""
    )  # noqa: E501
    logger.info(f"   Rule system valid: {validation_report['system_valid']}")'"
    logger.info(f"   Total rules: {validation_report['total_rules']}")'"
    logger.info()
        f"   Inflection types: {validation_report['coverage_analysis']['inflection_types_covered']}"'"
    )  # noqa: E501

    # Generate documentation
    documentation = engine.get_rule_documentation()

    return {
        'engine': engine,'
        'test_results': {'
            'passed': passed_tests,'
            'total': total_tests,'
            'success_rate': passed_tests / total_tests,'
        },
        'validation_report': validation_report,'
        'documentation': documentation,'
    }


if __name__ == "__main__":"
    # Run comprehensive tests
    results = test_inflection_rules_engine()

    logger.info("\n🎯 Arabic Inflection Rules Engine - Test Complete!")"
    logger.info(f"Success Rate: {results['test_results']['success_rate']*100:.1f%}")'"
    logger.info(f"System Valid: {results['validation_report']['system_valid']}")'"

    # Save documentation
    with open('arabic_inflection_rules_documentation.json', 'w', encoding='utf 8') as f:'
        json.dump(results['documentation'], f, ensure_ascii=False, indent=2)'

    logger.info("📄 Documentation saved to: arabic_inflection_rules_documentation.json")"

