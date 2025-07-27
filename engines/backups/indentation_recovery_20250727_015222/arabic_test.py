#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to check Arabic character handling in PowerShell
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long,
    def test_arabic_characters():  # type: ignore[no-untyped def]
    print("=" * 50)"
    print("Arabic Character Encoding Test")"
    print("=" * 50)"

    # Test various Arabic characters including ؤ,
    arabic_chars = [
    'ا','
    'ب','
    'ت','
    'ث','
    'ج','
    'ح','
    'خ','
    'د','
    'ذ','
    'ر','
    'ز','
    'س','
    'ش','
    'ص','
    'ض','
    'ط','
    'ظ','
    'ع','
    'غ','
    'ف','
    'ق','
    'ك','
    'ل','
    'م','
    'ن','
    'ه','
    'و','
    'ي','
    'ؤ','
    'ئ','
    ]

    print("Testing Arabic characters:")"
    for i, char in enumerate(arabic_chars):
    print(f"{i+1:2d}. {char} (Unicode: U+{ord(char):04X)}")"

    print("\nTesting the specific character 'ؤ':")'"
    test_char = 'ؤ''
    print(f"Character: {test_char}")"
    print(f"Unicode: U+{ord(test_char):04X}")"
    print(f"UTF-8 bytes: {test_char.encode('utf 8')}")'"

    print("\nTesting in syllabic_unit engine context:")"
    try:
from nlp.syllable.engine import SyllabicUnitEngine  # noqa: F401,
    engine = SyllabicUnitEngine()

        # Test if ؤ is in PHONEME_MAP,
    if test_char in engine.PHONEME_MAP:
    phoneme = engine.PHONEME_MAP[test_char]
    print(f"✅ '{test_char}' maps to phoneme: '{phoneme}")'"
        else:
    print(f"❌ '{test_char}' NOT found in PHONEME_MAP")'"

        # Test phonemization,
    result = engine.phonemize(test_char)
    print(f"Phonemize result: '{result'}")'"

    except Exception as e:
    print(f"❌ Error with syllabic_unit engine: {e}")"

    print("=" * 50)"


if __name__ == "__main__":"
    test_arabic_characters()

