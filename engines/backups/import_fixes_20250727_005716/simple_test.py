#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import sys  # noqa: F401
import traceback  # noqa: F401

try:
    print("🔧 Importing SyllabicUnitEngine...")
    from nlp.syllable.engine import SyllabicUnitEngine  # noqa: F401

    print("✅ Import successful")

    print("🔧 Creating engine instance...")
    engine = SyllabicUnitEngine()
    print("✅ Engine created")

    print("🔧 Testing phonemization...")
    text = "كتاب"
    phonemes = engine.phonemize(text)
    print(f"✅ Phonemes for '{text}': {phonemes}")

    print("🔧 Testing syllabification...")
    result = engine.syllabify_text(text)
    print(f"✅ Syllabification result: {result}")

except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()
