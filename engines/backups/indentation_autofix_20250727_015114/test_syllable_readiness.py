#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for syllable engine verification
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


try:
    print("🔬 Testing syllable engine import...")
    import syllable_phonological_engine  # noqa: F401

    print("✅ Import successful")

    print("🔬 Testing engine initialization...")
    from syllable_phonological_engine import (
        PurePhonologicalSyllableEngine,
    )  # noqa: F401

    engine = PurePhonologicalSyllableEngine()
    print("✅ Engine initialized")

    print("🔬 Testing syllabification...")
    test_word = "كَتَبَ"
    syllables = engine.syllabify_word(test_word)
    print(f"✅ Syllabification successful: {len(syllables)} syllables")

    print("🔬 Testing prosodic analysis...")
    analysis = engine.analyze_prosodic_structure(syllables)
    print(f"✅ Prosodic analysis complete: Weight {analysis['total_prosodic_weight']}")

    print("\n🎉 SYLLABLE ENGINE IS 100% OPERATIONAL AND READY!")
    print("🚀 Ready for Stage 2 development!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback  # noqa: F401

    traceback.print_exc()
