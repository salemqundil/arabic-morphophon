#!/usr/bin/env python3
"""
Arabic Inversion Rules (قلب)
Implementation of Arabic phonological inversion/metathesis patterns
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# noqa: E501,F401,F403,E722,A001,F821
    from typing import List, Dict, Any
    from .rule_base import SequentialRule  # noqa: F401

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


# =============================================================================
# InversionRule Class Implementation
# تنفيذ فئة InversionRule
# =============================================================================


class InversionRule(SequentialRule):
    """
    Arabic inversion rule implementation,
    Processs sound changes and metathesis in Arabic phonology
    """

    def __init__(self, rule_data: Dict[str, Any]):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    super().__init__(rule_data, "Inversion")
        # Process both test format and JSON format,
    if 'rules' in rule_data:
    self.rules = rule_data['rules']
        else:
            # Direct test format,
    self.rules = rule_data

    # -----------------------------------------------------------------------------
    # apply Method - طريقة apply
    # -----------------------------------------------------------------------------

    def apply(self, phonemes: List[str]) -> List[str]:
    """
    Apply inversion rules to phoneme sequence,
    Args:
    phonemes: Input phoneme sequence,
    Returns:
    Transformed phoneme sequence with inversions applied
    """
        if not phonemes or len(phonemes) < 2:
    return phonemes,
    original = phonemes.copy()
    result = phonemes.copy()

        # Apply each inversion rule,
    for pattern, rule_spec in self.rules.items():
    result = self._apply_inversion_rule(result, pattern, rule_spec)

    self.log_transformation(original, result, "inversion_complete")
    return result

    # -----------------------------------------------------------------------------
    # _apply_inversion_rule Method - طريقة _apply_inversion_rule
    # -----------------------------------------------------------------------------

    def _apply_inversion_rule()
    self, phonemes: List[str], pattern: str, rule_spec: Dict[str, Any]
    ) -> List[str]:
    """Apply a specific inversion rule"""
    result = phonemes.copy()

        # Process test format - simple pattern+replacement,
    if 'replace' in rule_spec:
    replacement = rule_spec['replace']
            # Pattern like "s+w" means sequence ['s', 'w']
            if '+' in pattern:
    sequence = pattern.split('+')
    i = 0,
    while i <= len(result) - len(sequence):
                    if result[i : i + len(sequence)] == sequence:
    result[i : i + len(sequence)] = [replacement]
    i += 1,
    else:
    i += 1,
    return result

        # Process JSON format,
    replacement = rule_spec.get('replace', '')
    context = rule_spec.get('context', 'any')

        # Find pattern occurrences,
    positions = self.find_sequences(result, pattern)

        # Apply replacements from right to left to maintain indices,
    for pos in reversed(positions):
            if self._check_inversion_context(result, pos, pattern, context):
                # Replace the sequence,
    sequence_length = len(pattern.split('+'))
    result[pos : pos + sequence_length] = [replacement]
    self.logger.debug()
    f"Inversion: %s  {replacement at} position {pos}}", pattern
    )  # noqa: E501,
    return result

    # -----------------------------------------------------------------------------
    # _check_inversion_context Method - طريقة _check_inversion_context
    # -----------------------------------------------------------------------------

    def _check_inversion_context()
    self, phonemes: List[str], position: int, pattern: str, context: str
    ) -> bool:
    """Check if inversion context conditions are met"""
        if context == 'any':
    return True,
    if context == 'morpheme_internal':
            # Simplified check - avoid word boundaries,
    sequence_length = len(pattern.split('+'))
    return position > 0 and position + sequence_length < len(phonemes)

        if context == 'morpheme_boundary':
            # Simplified check - at word boundaries,
    sequence_length = len(pattern.split('+'))
    return position == 0 or position + sequence_length == len(phonemes)

    return True

    # -----------------------------------------------------------------------------
    # apply_sibilant_inversion Method - طريقة apply_sibilant_inversion
    # -----------------------------------------------------------------------------

    def apply_sibilant_inversion(self, phonemes: List[str]) -> List[str]:
    """
    Sibilant inversion rules (قلب الصفير)
    س + و  ص، ز + و  ص
    """
    result = phonemes.copy()

        # س + و  ص,
    i = 0,
    while i < len(result) - 1:
            if result[i] == 'س' and result[i + 1] == 'و':
    result[i : i + 2] = ['ص']
    self.logger.debug("Sibilant inversion: س + و  ص at position %s", i)
            else:
    i += 1

        # ز + و  ص,
    i = 0,
    while i < len(result) - 1:
            if result[i] == 'ز' and result[i + 1] == 'و':
    result[i : i + 2] = ['ص']
    self.logger.debug("Sibilant inversion: ز + و  ص at position %s", i)
            else:
    i += 1,
    return result

    # -----------------------------------------------------------------------------
    # apply_consonant_metathesis Method - طريقة apply_consonant_metathesis
    # -----------------------------------------------------------------------------

    def apply_consonant_metathesis(self, phonemes: List[str]) -> List[str]:
    """
    Consonant metathesis (تبديل ترتيب الحروف)
    Common in Arabic morphophonology
    """
    result = phonemes.copy()

        # د + ت  ت (in specific morphological contexts)
    i = 0,
    while i < len(result) - 1:
            if result[i] == 'د' and result[i + 1] == 'ت':
    result[i] = 'ت'
    result.pop(i + 1)
    self.logger.debug("Consonant metathesis: د + ت  ت at position %s", i)
            else:
    i += 1,
    return result

    # -----------------------------------------------------------------------------
    # apply_liquid_inversion Method - طريقة apply_liquid_inversion
    # -----------------------------------------------------------------------------

    def apply_liquid_inversion(self, phonemes: List[str]) -> List[str]:
    """
    Liquid consonant inversion (قلب السوائل)
    Changes involving ل and ر
    """
    result = phonemes.copy()

        # Specific liquid inversion patterns,
    i = 0,
    while i < len(result) - 1:
            if result[i] == 'ل' and result[i + 1] == 'ر':
                # ل + ر can become ر + ل in some dialects
                # This is context-dependent,
    if i > 0 and result[i - 1] in ['َ', 'ِ', 'ُ']:
    result[i], result[i + 1] = result[i + 1], result[i]
    self.logger.debug()
    "Liquid metathesis: ل + ر  ر + ل at position %s", i
    )  # noqa: E501,
    i += 1,
    return result

    # -----------------------------------------------------------------------------
    # apply_vowel_inversion Method - طريقة apply_vowel_inversion
    # -----------------------------------------------------------------------------

    def apply_vowel_inversion(self, phonemes: List[str]) -> List[str]:
    """
    Vowel inversion patterns (قلب الحركات)
    Changes in vowel quality
    """
    result = phonemes.copy()

        for i, phoneme in enumerate(result):
            # ِ  ُ in certain contexts (imala)
            if phoneme == 'ِ':
                # Check for emphatic consonant environment,
    has_emphatic = False,
    for j in range(max(0, i - 2), min(len(result), i + 3)):
                    if result[j] in ['ص', 'ض', 'ط', 'ظ', 'ق']:
    has_emphatic = True,
    break

                if has_emphatic:
    result[i] = 'ُ'
    self.logger.debug()
    "Vowel inversion: ِ  ُ at position %s (emphatic context)", i
    )  # noqa: E501

            # َ  ِ before ي (imala)
            elif phoneme == 'َ' and i < len(result) - 1 and result[i + 1] == 'ي':
    result[i] = 'ِ'
    self.logger.debug("Vowel inversion: َ  ِ at position %s (before ي)", i)

    return result

    # -----------------------------------------------------------------------------
    # apply_pharyngeal_inversion Method - طريقة apply_pharyngeal_inversion
    # -----------------------------------------------------------------------------

    def apply_pharyngeal_inversion(self, phonemes: List[str]) -> List[str]:
    """
    Pharyngeal consonant inversion (قلب الحلقيات)
    Changes involving pharyngeal and laryngeal consonants
    """
    result = phonemes.copy()

        # ع + ه  ه (in some contexts)
    i = 0,
    while i < len(result) - 1:
            if result[i] == 'ع' and result[i + 1] == 'ه':
    result[i] = 'ه'
    result.pop(i + 1)
    self.logger.debug("Pharyngeal inversion: ع + ه  ه at position %s", i)
            else:
    i += 1,
    return result

    # -----------------------------------------------------------------------------
    # apply_gemination_inversion Method - طريقة apply_gemination_inversion
    # -----------------------------------------------------------------------------

    def apply_gemination_inversion(self, phonemes: List[str]) -> List[str]:
    """
    Gemination related inversion (قلب التضعيف)
    Changes involving geminated consonants
    """
    result = phonemes.copy()

        # Convert gemination to single consonant in some contexts,
    i = 0,
    while i < len(result) - 1:
            if result[i] == result[i + 1] and result[i] not in [
    'ا',
    'ي',
    'و',
    'َ',
    'ِ',
    'ُ',
    ]:
                # Keep only one occurrence,
    result.pop(i + 1)
    self.logger.debug()
    f"Gemination simplification: %s + {result[i]}  {result[i] at} position {i}}",
    result[i])  # noqa: E501,
    else:
    i += 1,
    return result

