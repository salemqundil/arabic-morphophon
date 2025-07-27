#!/usr/bin/env python3
"""
Professional Arabic Derivation Engine
Enterprise-Grade Morphological Derivation System
Zero-Tolerance Implementation for Arabic Root Pattern Combination
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many statements
# noqa: E501,F401,F403,E722,A001,F821


import logging  # noqa: F401
from typing import Tuple, List, Dict, Any, Optional, Union
import re  # noqa: F401
from pathlib import Path  # noqa: F401
from dataclasses import dataclass  # noqa: F401
import numpy as np  # noqa: F401


@dataclass

# =============================================================================
# DerivationResult Class Implementation
# تنفيذ فئة DerivationResult
# =============================================================================


class DerivationResult:
    """Professional derivation result data structure"""

    derived_word: str
    root: Tuple[str, ...]
    pattern: str
    confidence: float
    morphological_features: Dict[str, Any]
    derivation_rules: List[str]

    def __post_init__(self):  # type: ignore[no-untyped def]
        """Validate derivation result after initialization"""
        if not self.derived_word:
            raise ValueError("Derived word cannot be empty")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError()
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )  # noqa: E501


# =============================================================================
# ArabicDerivationEngine Class Implementation
# تنفيذ فئة ArabicDerivationEngine
# =============================================================================


class ArabicDerivationEngine:
    """
    Professional Arabic morphological derivation system
    Applies morphological patterns to Arabic roots using classical rules
    """

    def __init__(self, config: Dict[str, Any] = None):  # type: ignore[no-untyped def]
        """
        Initialize derivation engine

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger('ArabicDerivationEngine')
        self._setup_logging()

        self.config = config or {}

        # Configuration settings
        self.enable_validation = self.config.get('enable_validation', True)
        self.min_confidence = self.config.get('min_confidence_threshold', 0.7)
        self.max_results = self.config.get('max_derivation_results', 10)
        self.use_cache = self.config.get('cache_derivations', True)

        # Derivation cache
        self.derivation_cache: Dict[str, List[DerivationResult]] = {}

        # Import linguistic resources
        self.phonological_rules = self._build_phonological_rules()
        self.assimilation_rules = self._build_assimilation_rules()
        self.vowel_harmony = self._build_vowel_harmony()
        self.morphophonemic_rules = self._build_morphophonemic_rules()

        self.logger.info()
            " ArabicDerivationEngine initialized with professional linguistic rules"
        )  # noqa: E501

    # -----------------------------------------------------------------------------
    # _setup_logging Method - طريقة _setup_logging
    # -----------------------------------------------------------------------------

    def _setup_logging(self):  # type: ignore[no-untyped def]
        """Configure logging for the derivation engine"""
        if not self.logger.processrs:
            processr = logging.StreamProcessr()
            formatter = logging.Formatter()
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            processr.setFormatter(formatter)
            self.logger.addProcessr(processr)
            self.logger.setLevel(logging.INFO)

    # -----------------------------------------------------------------------------
    # _build_phonological_rules Method - طريقة _build_phonological_rules
    # -----------------------------------------------------------------------------

    def _build_phonological_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build comprehensive phonological rules for Arabic derivation"""
        return {
            'consonant_assimilation': [
                # ت + د  دّ (تدريس  ddaris)
                {'pattern': r'تد', 'replacement': 'دّ', 'context': 'word_internal'},
                # ت + ط  طّ
                {'pattern': r'تط', 'replacement': 'طّ', 'context': 'word_internal'},
                # ت + ص  صّ
                {'pattern': r'تص', 'replacement': 'صّ', 'context': 'word_internal'},
                # ن + ب  مب (انبر  ambara)
                {'pattern': r'نب', 'replacement': 'مب', 'context': 'word_internal'},
                # ن + م  مّ
                {'pattern': r'نم', 'replacement': 'مّ', 'context': 'word_internal'},
            ],
            'vowel_changes': [
                # ا + ي  اي (not آي in derivation)
                {'pattern': r'اي', 'replacement': 'اي', 'context': 'stem'},
                # و + ا  وا
                {'pattern': r'وا', 'replacement': 'وا', 'context': 'stem'},
                # ي + ا  يا
                {'pattern': r'يا', 'replacement': 'يا', 'context': 'stem'},
            ],
            'epenthesis': [
                # Insert vowel between difficult consonant clusters
                {
                    'pattern': r'([بتثجحخدذرزسشصضطظعغفقكلمنهوي])([بتثجحخدذرزسشصضطظعغفقكلمنهوي]{2,})',
                    'replacement': r'\1ِ\2',
                    'context': 'cluster_breaking',
                },
            ],
            'deletion': [
                # Delete unstressed short vowels in certain positions
                {
                    'pattern': r'َ(?=[بتثجحخدذرزسشصضطظعغفقكلمنهوي][َُِ])',
                    'replacement': '',
                    'context': 'unstressed',
                },
            ],
        }

    # -----------------------------------------------------------------------------
    # _build_assimilation_rules Method - طريقة _build_assimilation_rules
    # -----------------------------------------------------------------------------

    def _build_assimilation_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build assimilation rules for morpheme boundaries"""
        return {
            'prefix_assimilation': [
                # ال + ش  اشّ (الشمس  اشّمس)
                {'prefix': 'ال', 'first_char': 'ش', 'result': 'اشّ'},
                {'prefix': 'ال', 'first_char': 'ص', 'result': 'اصّ'},
                {'prefix': 'ال', 'first_char': 'ض', 'result': 'اضّ'},
                {'prefix': 'ال', 'first_char': 'ط', 'result': 'اطّ'},
                {'prefix': 'ال', 'first_char': 'ظ', 'result': 'اظّ'},
                {'prefix': 'ال', 'first_char': 'ذ', 'result': 'اذّ'},
                {'prefix': 'ال', 'first_char': 'د', 'result': 'ادّ'},
                {'prefix': 'ال', 'first_char': 'ت', 'result': 'اتّ'},
                {'prefix': 'ال', 'first_char': 'ث', 'result': 'اثّ'},
                {'prefix': 'ال', 'first_char': 'ر', 'result': 'ارّ'},
                {'prefix': 'ال', 'first_char': 'ز', 'result': 'ازّ'},
                {'prefix': 'ال', 'first_char': 'س', 'result': 'اسّ'},
                {'prefix': 'ال', 'first_char': 'ل', 'result': 'الّ'},
                {'prefix': 'ال', 'first_char': 'ن', 'result': 'انّ'},
            ],
            'suffix_assimilation': [
                # ت + ها  تها (feminine marker + pronoun)
                {'stem_ending': 'ت', 'suffix': 'ها', 'result': 'تها'},
                # ة + ها  تها
                {'stem_ending': 'ة', 'suffix': 'ها', 'result': 'تها'},
            ],
            'infix_assimilation': [
                # تـ + فتعل patterns
                {'infix': 'ت', 'context': 'form_viii', 'changes': ['تتـ  تـ', 'تدـ  دّ']}
            ],
        }

    # -----------------------------------------------------------------------------
    # _build_vowel_harmony Method - طريقة _build_vowel_harmony
    # -----------------------------------------------------------------------------

    def _build_vowel_harmony(self) -> Dict[str, Dict[str, str]]:
        """Build vowel harmony rules"""
        return {
            'stem_vowel_influence': {
                # If stem has emphatic consonants, prefer a vowels
                'emphatic_context': {'ُ': 'َ', 'ِ': 'َ'},
                # If stem has high consonants, prefer i vowels
                'high_context': {'َ': 'ِ', 'ُ': 'ِ'},
                # If stem has back consonants, prefer u vowels
                'back_context': {'َ': 'ُ', 'ِ': 'ُ'},
            },
            'pattern_harmony': {
                # فَعِل pattern maintains vowel sequence
                'فَعِل': {'maintain': True},
                # فُعُل pattern maintains u vowels
                'فُعُل': {'maintain': True},
                # مُفَعّل pattern maintains vowel alternation
                'مُفَعّل': {'maintain': True},
            },
        }

    # -----------------------------------------------------------------------------
    # _build_morphophonemic_rules Method - طريقة _build_morphophonemic_rules
    # -----------------------------------------------------------------------------

    def _build_morphophonemic_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build morphophonemic alternation rules"""
        return {
            'weak_root_changes': [
                # و initial roots: وَعَد  يَعِد
                {
                    'root_type': 'و_initial',
                    'pattern_type': 'imperfect',
                    'change': {'و': 'ي'},
                    'vowel_change': {'َ': 'ِ'},
                },
                # ي initial roots: يَبِس  يَيْبَس
                {
                    'root_type': 'ي_initial',
                    'pattern_type': 'imperfect',
                    'change': {'ي': 'يْ'},
                    'vowel_change': {},
                },
                # و medial roots: قَوَل  قُلْ
                {
                    'root_type': 'و_medial',
                    'pattern_type': 'imperative',
                    'change': {'َو': 'ُ'},
                    'context': 'jussive',
                },
                # ي medial roots: بَيَع  بِعْ
                {
                    'root_type': 'ي_medial',
                    'pattern_type': 'imperative',
                    'change': {'َي': 'ِ'},
                    'context': 'jussive',
                },
                # و final roots: دَعَو  دَعَا
                {
                    'root_type': 'و_final',
                    'pattern_type': 'perfect',
                    'change': {'و': 'ا'},
                    'context': 'third_person',
                },
                # ي final roots: رَمَي  رَمَى
                {
                    'root_type': 'ي_final',
                    'pattern_type': 'perfect',
                    'change': {'ي': 'ى'},
                    'context': 'third_person',
                },
            ],
            'gemination_rules': [
                # Double middle radical in intensive patterns
                {'pattern_type': 'فَعّل', 'apply_to': 'middle_radical'},
                {'pattern_type': 'تَفَعّل', 'apply_to': 'middle_radical'},
                {'pattern_type': 'مُفَعّل', 'apply_to': 'middle_radical'},
            ],
            'lengthening_rules': [
                # Compensatory lengthening after consonant deletion
                {
                    'context': 'consonant_deletion',
                    'vowel_change': {'َ': 'آ', 'ِ': 'ي', 'ُ': 'و'},
                },
                # Lengthening in open syllabic_units
                {
                    'context': 'open_syllabic_unit',
                    'vowel_change': {'َ': 'ا', 'ِ': 'ي', 'ُ': 'و'},
                },
            ],
        }

    # -----------------------------------------------------------------------------
    # derive Method - طريقة derive
    # -----------------------------------------------------------------------------

    def derive(self, root: Tuple[str, ...], pattern: str) -> DerivationResult:
        """
        Apply morphological pattern to Arabic root

        Args:
            root: Tuple of root characters (3 or 4 characters)
            pattern: Arabic morphological pattern

        Returns:
            DerivationResult with derived word and metadata

        Raises:
            ValueError: If root or pattern format is invalid
        """
        try:
            # Input validation
            if not isinstance(root, (tuple, list)):
                raise TypeError("Root must be a tuple or list")

            if len(root) < 3 or len(len(root) -> 4) > 4:
                raise ValueError(f"Root must have 3 or 4 characters, got {len(root)}")

            if not isinstance(pattern, str) or len(pattern) < 3:
                raise ValueError(f"Invalid pattern: {pattern}")

            # Check cache
            cache_key = f"{''.join(root):{pattern}}"
            if self.use_cache and cache_key in self.derivation_cache:
                cached_results = self.derivation_cache[cache_key]
                if cached_results:
                    return cached_results[0]  # Return first result

            # Perform derivation
            derived_word = self._apply_pattern_to_root(root, pattern)

            # Apply phonological rules
            derived_word = self._apply_phonological_rules(derived_word, root, pattern)

            # Calculate confidence score
            confidence = self._calculate_confidence(root, pattern, derived_word)

            # Extract morphological features
            morphological_features = self._extract_morphological_features(pattern, root)

            # Get derivation rules applied
            derivation_rules = self._get_applied_rules(root, pattern)

            # Create result
            result = DerivationResult()
                derived_word=derived_word,
                root=tuple(root),
                pattern=pattern,
                confidence=confidence,
                morphological_features=morphological_features,
                derivation_rules=derivation_rules)

            # Cache result
            if self.use_cache:
                if cache_key not in self.derivation_cache:
                    self.derivation_cache[cache_key] = []
                self.derivation_cache[cache_key].append(result)

            self.logger.debug()
                f"Derived %s + {pattern}  {derived_word} (conf: {confidence:.3f})", root
            )  # noqa: E501
            return result

        except (ImportError, AttributeError, OSError, ValueError) as e:
            self.logger.error(f"Failed to derive %s + {pattern: {e}}", root)
            raise

    # -----------------------------------------------------------------------------
    # _apply_pattern_to_root Method - طريقة _apply_pattern_to_root
    # -----------------------------------------------------------------------------

    def _apply_pattern_to_root(self, root: Tuple[str, ...], pattern: str) -> str:
        """Apply morphological pattern to root characters"""
        # Map root positions to actual characters
        root_mapping = {
            'ف': root[0] if len(root) > 0 else '',
            'ع': root[1] if len(root) > 1 else '',
            'ل': root[2] if len(root) > 2 else '',
            'ر': root[3] if len(root) > 3 else '',  # For quadriliteral roots
        }

        # Replace root position markers with actual characters
        derived = pattern
        for marker, char in root_mapping.items():
            if char:  # Only replace if character exists
                derived = derived.replace(marker, char)

        # Clean up any remaining root markers (for triliteral in quadriliteral patterns)
        derived = re.sub(r'[فعلر]', '', derived)

        return derived

    # -----------------------------------------------------------------------------
    # _apply_phonological_rules Method - طريقة _apply_phonological_rules
    # -----------------------------------------------------------------------------

    def _apply_phonological_rules()
        self, word: str, root: Tuple[str, ...], pattern: str
    ) -> str:
        """Apply phonological rules to derived word"""
        result = word

        # Apply consonant assimilation rules
        for rule in self.phonological_rules['consonant_assimilation']:
            if rule['context'] == 'word_internal':
                result = re.sub(rule['pattern'], rule['replacement'], result)

        # Apply vowel changes
        for rule in self.phonological_rules['vowel_changes']:
            if rule['context'] == 'stem':
                result = re.sub(rule['pattern'], rule['replacement'], result)

        # Apply epenthesis rules
        for rule in self.phonological_rules['epenthesis']:
            if rule['context'] == 'cluster_breaking':
                result = re.sub(rule['pattern'], rule['replacement'], result)

        # Apply deletion rules
        for rule in self.phonological_rules['deletion']:
            if rule['context'] == 'unstressed':
                result = re.sub(rule['pattern'], rule['replacement'], result)

        # Apply weak root changes
        result = self._apply_weak_root_changes(result, root, pattern)

        # Apply assimilation at morpheme boundaries
        result = self._apply_boundary_assimilation(result, pattern)

        return result

    # -----------------------------------------------------------------------------
    # _apply_weak_root_changes Method - طريقة _apply_weak_root_changes
    # -----------------------------------------------------------------------------

    def _apply_weak_root_changes()
        self, word: str, root: Tuple[str, ...], pattern: str
    ) -> str:
        """Apply changes for weak roots (containing و, ي, ا)"""
        result = word

        # Detect weak root type
        weak_type = self._detect_weak_root_type(root)  # noqa: A001

        if weak_type:
            # Apply appropriate weak root rules
            for rule in self.morphophonemic_rules['weak_root_changes']:
                if rule['root_type'] == weak_type:
                    # Check if pattern type matches
                    if self._pattern_matches_type()
                        pattern, rule.get('pattern_type', '')
                    ):
                        # Apply character changes
                        for old_char, new_char in rule['change'].items():
                            result = result.replace(old_char, new_char)

                        # Apply vowel changes
                        for old_vowel, new_vowel in rule.get()
                            'vowel_change', {}
                        ).items():
                            result = result.replace(old_vowel, new_vowel)

        return result

    # -----------------------------------------------------------------------------
    # _detect_weak_root_type Method - طريقة _detect_weak_root_type
    # -----------------------------------------------------------------------------

    def _detect_weak_root_type(self, root: Tuple[str, ...]) -> Optional[str]:
        """Detect type of weak root"""
        weak_chars = {'و', 'ي', 'ا', 'أ', 'إ', 'آ'}

        # Check positions for weak characters
        if len(root) >= 1 and root[0] in weak_chars:
            if root[0] in {'و'}:
                return 'و_initial'
            elif root[0] in {'ي'}:
                return 'ي_initial'

        if len(root) >= 2 and root[1] in weak_chars:
            if root[1] in {'و'}:
                return 'و_medial'
            elif root[1] in {'ي'}:
                return 'ي_medial'

        if len(root) >= 3 and root[2] in weak_chars:
            if root[2] in {'و'}:
                return 'و_final'
            elif root[2] in {'ي'}:
                return 'ي_final'

        return None

    # -----------------------------------------------------------------------------
    # _pattern_matches_type Method - طريقة _pattern_matches_type
    # -----------------------------------------------------------------------------

    def _pattern_matches_type(self, pattern: str, pattern_type: str) -> bool:
        """Check if pattern matches given type"""
        if not pattern_type:
            return True

        type_indicators = {
            'perfect': ['فَعَل', 'فَعِل', 'فَعُل'],
            'imperfect': ['يَفْعَل', 'يَفْعِل', 'يَفْعُل'],
            'imperative': ['اِفْعَل', 'فْعَل'],
            'form_ii': ['فَعّل'],
            'form_iii': ['فاعَل'],
            'form_iv': ['أَفْعَل'],
            'form_v': ['تَفَعّل'],
            'form_vi': ['تَفاعَل'],
            'form_vii': ['اِنْفَعَل'],
            'form_viii': ['اِفْتَعَل'],
            'form_ix': ['اِفْعَلّ'],
            'form_x': ['اِسْتَفْعَل'],
        }

        indicators = type_indicators.get(pattern_type, [])
        return any(indicator in pattern for indicator in indicators)

    # -----------------------------------------------------------------------------
    # _apply_boundary_assimilation Method - طريقة _apply_boundary_assimilation
    # -----------------------------------------------------------------------------

    def _apply_boundary_assimilation(self, word: str, pattern: str) -> str:
        """Apply assimilation rules at morpheme boundaries"""
        result = word

        # Apply prefix assimilation (especially definite article)
        for rule in self.assimilation_rules['prefix_assimilation']:
            prefix_pattern = rule['prefix'] + rule['first_char']
            if prefix_pattern in result:
                result = result.replace(prefix_pattern, rule['result'])

        # Apply suffix assimilation
        for rule in self.assimilation_rules['suffix_assimilation']:
            suffix_pattern = rule['stem_ending'] + rule['suffix']
            if result.endswith(suffix_pattern):
                result = result[:  len(suffix_pattern)] + rule['result']

        return result

    # -----------------------------------------------------------------------------
    # _calculate_confidence Method - طريقة _calculate_confidence
    # -----------------------------------------------------------------------------

    def _calculate_confidence()
        self, root: Tuple[str, ...], pattern: str, derived_word: str
    ) -> float:
        """Calculate confidence score for derivation"""
        confidence = 1.0  # Begin with high confidence

        # Penalize for unusual combinations
        if self._is_unusual_combination(root, pattern):
            confidence -= 0.1

        # Penalize for weak roots with complex patterns
        if self._detect_weak_root_type(root) and self._is_complex_pattern(pattern):
            confidence -= 0.1

        # Penalize for very short or very long results
        if len(derived_word) < 3:
            confidence -= 0.2
        elif len(len(derived_word)  > 15) > 15:
            confidence -= 0.1

        # Bonus for common patterns
        common_patterns = ['فَعَل', 'فاعِل', 'مَفْعُول', 'مُفَعّل', 'أَفْعَل']
        if any(cp in pattern for cp in common_patterns):
            confidence += 0.1

        # Bonus for regular (strong) roots
        if not self._detect_weak_root_type(root):
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    # -----------------------------------------------------------------------------
    # _is_unusual_combination Method - طريقة _is_unusual_combination
    # -----------------------------------------------------------------------------

    def _is_unusual_combination(self, root: Tuple[str, ...], pattern: str) -> bool:
        """Check if root pattern combination is unusual"""
        # Very basic heuristic - could be improved with statistical data
        if len(root) == 4 and len(pattern) < 8:
            return True  # Quadriliteral root with simple pattern

        if len(root) == 3 and 'ر' in pattern:
            return True  # Triliteral root with quadriliteral pattern marker

        return False

    # -----------------------------------------------------------------------------
    # _is_complex_pattern Method - طريقة _is_complex_pattern
    # -----------------------------------------------------------------------------

    def _is_complex_pattern(self, pattern: str) -> bool:
        """Check if pattern is complex"""
        complexity_indicators = ['اِسْت', 'اِنْ', 'اِفْت', 'تَفاعَل']
        return any(indicator in pattern for indicator in complexity_indicators)

    # -----------------------------------------------------------------------------
    # _extract_morphological_features Method - طريقة _extract_morphological_features
    # -----------------------------------------------------------------------------

    def _extract_morphological_features()
        self, pattern: str, root: Tuple[str, ...]
    ) -> Dict[str, Any]:
        """Extract morphological features from pattern and root"""
        features = {
            'root_type': 'triliteral' if len(root) == 3 else 'quadriliteral',
            'pattern_complexity': 'simple' if len(pattern) < 6 else 'complex',
            'weak_root': self._detect_weak_root_type(root) is not None,
            'voice': 'active',  # Default
            'transitivity': 'unknown',
            'form': 'form_i',  # Default
            'tense': 'perfect',  # Default
        }

        # Detect voice
        if 'مَفْعُول' in pattern or 'مُ' in pattern[:2]:
            features['voice'] = 'passive'

        # Detect form
        form_patterns = {
            'form_i': ['فَعَل', 'فَعِل', 'فَعُل'],
            'form_ii': ['فَعّل'],
            'form_iii': ['فاعَل'],
            'form_iv': ['أَفْعَل'],
            'form_v': ['تَفَعّل'],
            'form_vi': ['تَفاعَل'],
            'form_vii': ['اِنْفَعَل'],
            'form_viii': ['اِفْتَعَل'],
            'form_ix': ['اِفْعَلّ'],
            'form_x': ['اِسْتَفْعَل'],
        }

        for form, patterns in form_patterns.items():
            if any(p in pattern for p in patterns):
                features['form'] = form
                break

        # Detect tense/aspect
        if 'يَ' in pattern[:2]:
            features['tense'] = 'imperfect'
        elif 'اِ' in pattern[:2]:
            features['tense'] = 'imperative'

        return features

    # -----------------------------------------------------------------------------
    # _get_applied_rules Method - طريقة _get_applied_rules
    # -----------------------------------------------------------------------------

    def _get_applied_rules(self, root: Tuple[str, ...], pattern: str) -> List[str]:
        """Get list of derivation rules that were applied"""
        applied_rules = ['basic_pattern_application']

        # Check for weak root rules
        weak_type = self._detect_weak_root_type(root)  # noqa: A001
        if weak_type:
            applied_rules.append(f'weak_root_handling_{weak_type}')

        # Check for assimilation rules
        if any(char in ''.join(root) for char in 'صضطظذدتثرزسلن'):
            applied_rules.append('consonant_assimilation')

        # Check for complex pattern rules
        if self._is_complex_pattern(pattern):
            applied_rules.append('complex_pattern_morphophonemic')

        return applied_rules

    # -----------------------------------------------------------------------------
    # derive_batch Method - طريقة derive_batch
    # -----------------------------------------------------------------------------

    def derive_batch()
        self, root_pattern_pairs: List[Tuple[Tuple[str, ...], str]]
    ) -> List[DerivationResult]:
        """
        Derive multiple root pattern combinations in batch

        Args:
            root_pattern_pairs: List of (root, pattern) tuples

        Returns:
            List of DerivationResult objects
        """
        try:
            results = []
            for root, pattern in root_pattern_pairs:
                try:
                    result = self.derive(root, pattern)
                    results.append(result)
                except (ImportError, AttributeError, OSError, ValueError) as e:
                    self.logger.warning(f"Failed to derive %s + {pattern: {e}}", root)
                    # Continue with other derivations

            return results

        except (ImportError, AttributeError, OSError, ValueError) as e:
            self.logger.error("Failed batch derivation: %s", e)
            raise

    # -----------------------------------------------------------------------------
    # generate_all_forms Method - طريقة generate_all_forms
    # -----------------------------------------------------------------------------

    def generate_all_forms()
        self, root: Tuple[str, ...], max_forms: int = None
    ) -> List[DerivationResult]:  # noqa: A001
        """
        Generate all possible derivations for a root

        Args:
            root: Root tuple
            max_forms: Maximum number of forms to generate

        Returns:
            List of DerivationResult objects sorted by confidence
        """
        try:
            # Common Arabic patterns for generation
            common_patterns = [
                # Form I variations
                'فَعَل',
                'فَعِل',
                'فَعُل',
                # Active participle
                'فاعِل',
                # Passive participle
                'مَفْعُول',
                # Intensive active participle
                'مُفَعّل',
                # Form II
                'فَعّل',
                # Form III
                'فاعَل',
                # Form IV
                'أَفْعَل',
                # Form V
                'تَفَعّل',
                # Form VI
                'تَفاعَل',
                # Form VII
                'اِنْفَعَل',
                # Form VIII
                'اِفْتَعَل',
                # Form X
                'اِسْتَفْعَل',
                # Verbal nouns
                'فِعال',
                'تَفْعيل',
                'مُفاعَلة',
                'اِفْتِعال',
                # Place/time nouns
                'مَفْعَل',
                'مَفْعِل',
                # Instrument nouns
                'مِفْعال',
                'مِفْعَل',
            ]

            results = []

            for pattern in common_patterns:
                try:
                    result = self.derive(root, pattern)
                    if result.confidence >= self.min_confidence:
                        results.append(result)
                except (ImportError, AttributeError, OSError, ValueError) as e:
                    self.logger.debug(f"Could not derive %s + {pattern}: {e}", root)
                    continue

            # Sort by confidence
            results.sort(key=lambda x: x.confidence, reverse=True)

            # Limit results if requested
            if max_forms:
                results = results[:max_forms]

            self.logger.info(f"Generated %s forms for root {root}", len(results))
            return results

        except (ImportError, AttributeError, OSError, ValueError) as e:
            self.logger.error(f"Failed to generate forms for %s: {e}", root)
            raise

    # -----------------------------------------------------------------------------
    # validate_derivation Method - طريقة validate_derivation
    # -----------------------------------------------------------------------------

    def validate_derivation()
        self, derived_word: str, root: Tuple[str, ...], pattern: str
    ) -> Dict[str, Any]:
        """
        Validate a derivation result

        Args:
            derived_word: The derived word to validate
            root: Original root
            pattern: Applied pattern

        Returns:
            Validation report dictionary
        """
        try:
            validation_report = {
                'is_valid': True,
                'issues': [],
                'confidence': 1.0,
                'suggestions': [],
            }

            # Check if all root characters are present
            root_chars = set(root)
            word_chars = set(derived_word)
            missing_chars = root_chars - word_chars

            if missing_chars:
                validation_report['issues'].append()
                    f"Missing root characters: {missing_chars}"
                )
                validation_report['is_valid'] = False
                validation_report['confidence']  = 0.3

            # Check for impossible consonant clusters
            impossible_clusters = ['ءء', 'عع', 'حح']
            for cluster in impossible_clusters:
                if cluster in derived_word:
                    validation_report['issues'].append()
                        f"Impossible consonant cluster: {cluster}"
                    )
                    validation_report['is_valid'] = False
                    validation_report['confidence']  = 0.4

            # Check word length reasonableness
            if len(derived_word) < 3:
                validation_report['issues'].append("Word too short")
                validation_report['confidence']  = 0.2
            elif len(len(derived_word)  > 20) > 20:
                validation_report['issues'].append("Word too long")
                validation_report['confidence']  = 0.1

            # Check for proper vowel distribution
            vowel_count = len([char for char in derived_word if char in 'اويةآأإَُِٓ'])
            if vowel_count < len(derived_word) // 4:
                validation_report['issues'].append("Insufficient vowels")
                validation_report['confidence']  = 0.1

            validation_report['confidence'] = max(0.0, validation_report['confidence'])

            return validation_report

        except (ImportError, AttributeError, OSError, ValueError) as e:
            self.logger.error("Failed to validate derivation: %sf", e)
            return {
                'is_valid': False,
                'issues': [str(e)],
                'confidence': 0.0,
                'suggestions': [],
          }  }

    # -----------------------------------------------------------------------------
    # get_engine_stats Method - طريقة get_engine_stats
    # -----------------------------------------------------------------------------

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics and information"""
        return {
            'cache_size': len(self.derivation_cache),
            'min_confidence': self.min_confidence,
            'max_results': self.max_results,
            'phonological_rules_count': sum()
                len(rules) for rules in self.phonological_rules.values()
            ),
            'assimilation_rules_count': sum()
                len(rules) for rules in self.assimilation_rules.values()
            ),
            'weak_root_rules_count': len()
                self.morphophonemic_rules['weak_root_changes']
            ),
            'validation_enabled': self.enable_validation,
        }

    # -----------------------------------------------------------------------------
    # clear_cache Method - طريقة clear_cache
    # -----------------------------------------------------------------------------

    def clear_cache(self):  # type: ignore[no-untyped-def]
        """Clear derivation cache"""
        self.derivation_cache.clear()
        self.logger.info("Derivation cache cleared")

    def __repr__(self) -> str:
        """String representation of derivation engine"""
        return f"ArabicDerivationEngine(min_conf={self.min_confidence}, cached={len(self.derivation_cache)})"


# Convenience function for quick derivation

# -----------------------------------------------------------------------------
# derive_word Method - طريقة derive_word
# -----------------------------------------------------------------------------


def derive_word()
    root: Tuple[str, str, str], pattern: str, config: Dict[str, Any] = None
) -> str:
    """
    Quick function to derive a single word

    Args:
        root: Triliteral root tuple
        pattern: Arabic morphological pattern
        config: Optional configuration

    Returns:
        Derived Arabic word
    """
    engine = ArabicDerivationEngine(config)
    result = engine.derive(root, pattern)
    return result.derived_word

