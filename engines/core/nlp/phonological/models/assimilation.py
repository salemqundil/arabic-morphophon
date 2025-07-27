#!/usr/bin/env python3
"""
Arabic Assimilation Rules (إدغام)
Implementation of Arabic phonological assimilation patterns
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
    from typing import List, Dict, Any
    from .rule_base import ContextualRule  # noqa: F401


# =============================================================================
# AssimilationRule Class Implementation
# تنفيذ فئة AssimilationRule
# =============================================================================


class AssimilationRule(ContextualRule):
    """
    Arabic assimilation rule implementation,
    Processs various types of assimilation in Arabic phonology
    """

    def __init__(self, rule_data: Dict[str, Any]):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    super().__init__(rule_data, "Assimilation")
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
    Apply assimilation rules to phoneme sequence,
    Args:
    phonemes: Input phoneme sequence,
    Returns:
    Transformed phoneme sequence with assimilation applied
    """
        if not phonemes:
    return phonemes,
    original = phonemes.copy()
    result = phonemes.copy()

        # Apply each assimilation rule,
    for trigger_phoneme, rule_spec in self.rules.items():
    result = self._apply_assimilation_rule(result, trigger_phoneme, rule_spec)

    self.log_transformation(original, result, "assimilation_complete")
    return result

    # -----------------------------------------------------------------------------
    # _apply_assimilation_rule Method - طريقة _apply_assimilation_rule
    # -----------------------------------------------------------------------------

    def _apply_assimilation_rule()
    self, phonemes: List[str], trigger: str, rule_spec: Dict[str, Any]
    ) -> List[str]:
    """Apply a specific assimilation rule"""
    result = phonemes.copy()

        # Process different rule specification formats,
    if 'targets' in rule_spec:
            # JSON format,
    targets = rule_spec.get('targets', [])
    replacement = rule_spec.get('replacement', '')
    context = rule_spec.get('context', 'adjacent')
        else:
            # Test format - rule_spec is the replacement pattern,
    targets = rule_spec.get('targets', [])
    replacement = rule_spec.get('replacement', trigger + ':')

    i = 0,
    while i < len(result) - 1:
            if result[i] == trigger and result[i + 1] in targets:
                # Replace with assimilated form,
    result[i : i + 2] = [replacement]
    i += 1,
    else:
    i += 1,
    return result,
    while i < len(result):
            if result[i] == trigger:
                if context == 'adjacent':
                    # Check next phoneme for assimilation,
    if i + 1 < len(result) and result[i + 1] in targets:
    result[i] = replacement,
    self.logger.debug()
    f"Assimilation: %s + {result[i + 1]}  {replacement}}",
    trigger)  # noqa: E501,
    elif context == 'prefix':
                    # Assimilation in prefix context (like تـ prefix)
                    if i + 1 < len(result) and result[i + 1] in targets:
    result[i] = result[i + 1]  # Complete assimilation,
    self.logger.debug()
    f"Prefix assimilation: %s + {result[i + 1]  {result[i} + 1]}}",
    trigger)  # noqa: E501,
    elif context == 'definite_article':
                    # Special case for definite article ال,
    if ()
    i > 0,
    and result[i - 1] == 'ا'
    and i + 1 < len(result)
    and result[i + 1] in targets
    ):
    result[i] = result[i + 1]  # ال becomes اX,
    self.logger.debug()
    f"Definite article assimilation: ا%s + {result[i + 1]}  ا{result[i} + 1]}",
    trigger)  # noqa: E501,
    i += 1,
    return result

    # -----------------------------------------------------------------------------
    # apply_nun_assimilation Method - طريقة apply_nun_assimilation
    # -----------------------------------------------------------------------------

    def apply_nun_assimilation(self, phonemes: List[str]) -> List[str]:
    """
    Specific implementation for nun assimilation (إدغام النون)
    One of the most common assimilation patterns in Arabic
    """
    result = phonemes.copy()

        for i in range(len(result) - 1):
            if result[i] == 'ن':
    next_phoneme = result[i + 1]

                # Nun assimilates completely with specific consonants,
    if next_phoneme in ['ل', 'ر', 'ي', 'و', 'م', 'ن']:
    result[i] = next_phoneme  # Complete assimilation,
    self.logger.debug()
    f"Nun assimilation: ن + %s  {next_phoneme}", next_phoneme
    )  # noqa: E501,
    return result

    # -----------------------------------------------------------------------------
    # apply_ta_assimilation Method - طريقة apply_ta_assimilation
    # -----------------------------------------------------------------------------

    def apply_ta_assimilation(self, phonemes: List[str]) -> List[str]:
    """
    Ta assimilation in prefixes (إدغام التاء)
    Common in verbal prefixes
    """
    result = phonemes.copy()

        for i in range(len(result) - 1):
            if result[i] == 'ت':
    next_phoneme = result[i + 1]

                # Ta assimilates with coronal consonants,
    if next_phoneme in ['د', 'ط', 'ص', 'ض', 'ظ', 'ذ', 'ز', 'س', 'ش']:
    result[i] = next_phoneme,
    self.logger.debug()
    f"Ta assimilation: ت + %s  {next_phoneme}", next_phoneme
    )  # noqa: E501,
    return result

    # -----------------------------------------------------------------------------
    # apply_lam_assimilation Method - طريقة apply_lam_assimilation
    # -----------------------------------------------------------------------------

    def apply_lam_assimilation(self, phonemes: List[str]) -> List[str]:
    """
    Lam assimilation in definite article (إدغام لام التعريف)
    """
    result = phonemes.copy()

        for i in range(len(result) - 1):
            # Check for definite article pattern: ا + ل + consonant,
    if ()
    i > 0,
    and result[i - 1] == 'ا'
    and result[i] == 'ل'
    and i + 1 < len(result)
    ):

    next_phoneme = result[i + 1]

                # Lam assimilates with sun letters (الحروف الشمسية)
    sun_letters = [
    'ص',
    'ض',
    'ط',
    'ظ',
    'ت',
    'د',
    'ذ',
    'ث',
    'ر',
    'ز',
    'س',
    'ش',
    'ن',
    ]

                if next_phoneme in sun_letters:
    result[i] = next_phoneme,
    self.logger.debug()
    f"Lam assimilation: ا + ل + %s  ا + {next_phoneme}",
    next_phoneme)  # noqa: E501,
    return result

