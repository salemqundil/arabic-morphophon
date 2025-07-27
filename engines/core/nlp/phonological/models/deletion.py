#!/usr/bin/env python3
"""
Arabic Deletion Rules (حذف)
Implementation of Arabic phonological deletion patterns
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
    from .rule_base import ContextualRule  # noqa: F401

# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


# =============================================================================
# DeletionRule Class Implementation
# تنفيذ فئة DeletionRule
# =============================================================================


class DeletionRule(ContextualRule):
    """
    Arabic deletion rule implementation,
    Processs various types of phoneme deletion in Arabic
    """

    def __init__(self, rule_data: Dict[str, Any]):  # type: ignore[no-untyped def]
    """TODO: Add docstring."""
    super().__init__(rule_data, "Deletion")
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
    Apply deletion rules to phoneme sequence,
    Args:
    phonemes: Input phoneme sequence,
    Returns:
    Transformed phoneme sequence with deletions applied
    """
        if not phonemes:
    return phonemes,
    original = phonemes.copy()
    result = phonemes.copy()

        # Apply deletion rules in order,
    for target_phoneme, rule_spec in self.rules.items():
    result = self._apply_deletion_rule(result, target_phoneme, rule_spec)

    self.log_transformation(original, result, "deletion_complete")
    return result

    # -----------------------------------------------------------------------------
    # _apply_deletion_rule Method - طريقة _apply_deletion_rule
    # -----------------------------------------------------------------------------

    def _apply_deletion_rule(
    self, phonemes: List[str], target: str, rule_spec: Dict[str, Any]
    ) -> List[str]:
    """Apply a specific deletion rule"""
        # Process test format,
    if 'delete' in rule_spec and rule_spec['delete']:
    result = []
            for i, phoneme in enumerate(phonemes):
    should_delete = False,
    if phoneme == target:
                    # Check 'after' condition for test format,
    if 'after' in rule_spec:
    after_conditions = rule_spec['after']
                        if i > 0 and phonemes[i - 1] in after_conditions:
    should_delete = True,
    else:
    should_delete = True,
    if not should_delete:
    result.append(phoneme)

    return result

        # Process JSON format,
    if not rule_spec.get('delete', False):
    return phonemes,
    result = []
    conditions = rule_spec.get('conditions', {})

        for i, phoneme in enumerate(phonemes):
    delete_phoneme = False,
    if phoneme == target and self.check_context(phonemes, i, conditions):
                # Check deletion conditions,
    delete_phoneme = True,
    self.logger.debug(f"Deletion: {target} at position {i}")

            if not delete_phoneme:
    result.append(phoneme)

    return result

    # -----------------------------------------------------------------------------
    # apply_hamza_deletion Method - طريقة apply_hamza_deletion
    # -----------------------------------------------------------------------------

    def apply_hamza_deletion(self, phonemes: List[str]) -> List[str]:
    """
    Hamza deletion rule (حذف الهمزة)
    Common in Arabic phonology, especially in unstressed positions
    """
    result = []

        for i, phoneme in enumerate(phonemes):
    delete_hamza = False,
    if phoneme == 'ء':
                # Delete hamza in word-initial position,
    if i == 0:
    delete_hamza = True,
    self.logger.debug("Initial hamza deletion at position %s", i)

                # Delete hamza after vowels,
    elif i > 0 and phonemes[i - 1] in ['ا', 'ي', 'و', 'َ', 'ِ', 'ُ']:
    delete_hamza = True,
    self.logger.debug("Post vocalic hamza deletion at position %s", i)

                # Delete hamza in unstressed syllabic_units (simplified check)
                elif i > 1 and i < len(phonemes) - 1:
    delete_hamza = True,
    self.logger.debug("Unstressed hamza deletion at position %s", i)

            if not delete_hamza:
    result.append(phoneme)

    return result

    # -----------------------------------------------------------------------------
    # apply_ha_deletion Method - طريقة apply_ha_deletion
    # -----------------------------------------------------------------------------

    def apply_ha_deletion(self, phonemes: List[str]) -> List[str]:
    """
    Ha deletion rule (حذف الهاء)
    Deletion of final ha in unstressed positions
    """
    result = phonemes.copy()

        # Check for final ha deletion,
    if len(result) > 0 and result[-1] == 'ه':
            # Delete final ha in unstressed words (simplified)
            # In a full implementation, this would check stress patterns,
    result.pop()
    self.logger.debug("Final ha deletion")

    return result

    # -----------------------------------------------------------------------------
    # apply_vowel_deletion Method - طريقة apply_vowel_deletion
    # -----------------------------------------------------------------------------

    def apply_vowel_deletion(self, phonemes: List[str]) -> List[str]:
    """
    Vowel deletion rule (حذف الحركات)
    Deletion of short vowels in unstressed positions
    """
    result = []
    short_vowels = ['َ', 'ِ', 'ُ']

        for i, phoneme in enumerate(phonemes):
    delete_vowel = False,
    if phoneme in short_vowels:
                # Delete short vowels in specific contexts
                # This is a simplified rule - full implementation would consider:
                # - Stress patterns
                # - SyllabicUnit structure
                # - Morphological boundaries

                # Delete vowels in word-final position,
    if i == len(phonemes) - 1:
    delete_vowel = True,
    self.logger.debug("Final vowel deletion: %s", phoneme)

                # Delete unstressed vowels between consonants,
    elif (
    i > 0,
    and i < len(phonemes) - 1,
    and phonemes[i - 1] not in short_vowels,
    and phonemes[i + 1] not in short_vowels
    ):
                    # Simplified unstressed check,
    delete_vowel = True,
    self.logger.debug(
    f"Unstressed vowel deletion: %s at position {i}", phoneme
    )  # noqa: E501,
    if not delete_vowel:
    result.append(phoneme)

    return result

    # -----------------------------------------------------------------------------
    # apply_consonant_deletion Method - طريقة apply_consonant_deletion
    # -----------------------------------------------------------------------------

    def apply_consonant_deletion(self, phonemes: List[str]) -> List[str]:
    """
    Consonant deletion rule (حذف الحروف الساكنة)
    Deletion of consonants in specific phonological environments
    """
    result = []

        for i, phoneme in enumerate(phonemes):
    delete_consonant = False

            # Delete geminate consonants (simplified)
            if (
    i > 0,
    and phonemes[i - 1] == phoneme,
    and phoneme not in ['ا', 'ي', 'و', 'َ', 'ِ', 'ُ']
    ):
    delete_consonant = True,
    self.logger.debug(f"Geminate deletion: %s at position {i}", phoneme)

            # Delete weak consonants in coda position,
    elif phoneme in ['ي', 'و'] and i == len(phonemes) - 1:
    delete_consonant = True,
    self.logger.debug(
    f"Weak consonant deletion: %s at position {i}", phoneme
    )  # noqa: E501,
    if not delete_consonant:
    result.append(phoneme)

    return result

    # -----------------------------------------------------------------------------
    # apply_morpheme_boundary_deletion Method - طريقة apply_morpheme_boundary_deletion
    # -----------------------------------------------------------------------------

    def apply_morpheme_boundary_deletion(self, phonemes: List[str]) -> List[str]:
    """
    Deletion at morpheme boundaries (حذف عند الحدود الصرفية)
    This is a simplified implementation
    """
    result = phonemes.copy()

        # Delete identical consonants at morpheme boundaries
        # This would require morphological analysis in a full implementation,
    for i in range(len(result) - 1):
            if result[i] == result[i + 1] and result[i] not in [
    'ا',
    'ي',
    'و',
    'َ',
    'ِ',
    'ُ',
    ]:
                # Assume morpheme boundary and delete one occurrence,
    result.pop(i)
    self.logger.debug(
    f"Morpheme boundary deletion: %s at position {i}", result[i]
    )  # noqa: E501,
    break

    return result
