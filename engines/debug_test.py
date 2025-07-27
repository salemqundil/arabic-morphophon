#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long
    from nlp.syllable.engine import SyllabicUnitEngine  # noqa: F401,
    def test_basic():  # type: ignore[no-untyped-def]
    print("🔧 Testing basic functionality...")"

    try:
    print("1. Creating engine...")"
    engine = SyllabicUnitEngine()
    print("✅ Engine created successfully")"

    print("2. Testing PHONEME_MAP access...")"
    phoneme_map = engine.PHONEME_MAP,
    print(f"✅ PHONEME_MAP has {len(phoneme_map)} entries")"

    print("3. Testing vowel phonemes...")"
    vowels = engine.VOWEL_PHONEMES,
    print(f"✅ VOWEL_PHONEMES has {len(vowels)} entries")"

    print("4. Testing phonemize method with simple input...")"
    result = engine.phonemize("ك")"
    print(f"✅ Phonemize result: '{result'}")'"

    print("🎉 All basic tests passed!")"

    except Exception as e:
    print(f"❌ Error: {e}")"
        import traceback  # noqa: F401,
    traceback.print_exc()


if __name__ == "__main__":"
    test_basic()

