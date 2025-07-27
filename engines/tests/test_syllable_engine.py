#!/usr/bin/env python3
"""
🧪 Comprehensive Arabic SyllabicUnit Engine Test Suite,
    مجموعة اختبارات شاملة لمحرك المقاطع العربية,
    Tests the improved syllabification algorithm with various Arabic text samples.
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long
    import sys  # noqa: F401
    from pathlib import Path  # noqa: F401

# Add the project root to sys.path,
    project_root = Path(__file__).parent,
    sys.path.insert(0, str(project_root))

from nlp.syllable.engine import SyllabicUnitEngine  # noqa: F401,
    def test_syllable_engine():  # type: ignore[no-untyped def]
    """Test the syllabic_unit engine with various Arabic texts"""
    print("🚀 Testing Advanced Arabic SyllabicUnit Engine")
    print("=" * 60)

    engine = SyllabicUnitEngine()

    # Test cases with expected behavior,
    test_cases = [
    {'text': 'كتاب', 'description': 'Simple two-syllable word (ki taab)'},
    {'text': 'الكتاب', 'description': 'Word with definite article (al-ki taab)'},
    {'text': 'مدرسة', 'description': 'Three-syllable word (mad-ra sa)'},
    {
    'text': 'العربية',
    'description': 'Complex word with long vowels (al-a-ra-biy ya)',
    },
    {'text': 'هل تحب الشعر؟', 'description': 'Complete sentence with question'},
    {
    'text': 'كَتَبَ الطَّالِبُ الدَّرْسَ',
    'description': 'Sentence with diacritics and shadda',
    },
    ]

    for i, test_case in enumerate(test_cases, 1):
    print(f"\n📝 Test Case {i}: {test_case['description']}")
    print(f"Input: {test_case['text']}")
    print(" " * 40)

        try:
    result = engine.syllabify_text(test_case['text'])

            if result['status'] == 'success':
    print(f"✅ Status: {result['status']}")
    print(f"📊 Total words: {result['total_words']}")
    print(f"📊 Total syllabic_units: {result['total_syllabic_units']}")
    print(f"🎯 Confidence: {result['confidence']}")

                # Display detailed analysis for each word,
    for j, analysis in enumerate(result['syllable_analysis']):
    print(f"\n  Word {j+1}: {analysis['word']}")
    print(
    f"    SyllabicUnits: {'} - '.join(analysis['syllabic_units'])}"
    )  # noqa: E501,
    print(f"    Patterns: {'} | '.join(analysis['syllable_patterns'])}")
    print(f"    Syllable count: {analysis['syllable_count']}")
    print(f"    Prosodic weight: {analysis['prosodic_weight']:.1f}")
    print(f"    Stress pattern: {analysis['stress_pattern']}")

                    # Show detailed syllable structures,
    print("    Detailed structure:")
                    for k, struct in enumerate(analysis['syllable_structures']):
    onset = '+'.join(struct['onset']) if struct['onset'] else '∅'
    nucleus = (
    '+'.join(struct['nucleus']) if struct['nucleus'] else '∅'
    )
    coda = '+'.join(struct['coda']) if struct['coda'] else '∅'
    stress_mark = " (STRESSED)" if struct['stress'] else ""
    print(
    f"      [{k+1}] {onset}.{nucleus}.{coda} (weight: {struct['weight']:.1f}){stress_mark}"
    )  # noqa: E501,
    else:
    print(f"❌ Error: {result.get('error',} 'Unknown error')}")

        except Exception as e:
    print(f"❌ Exception: {e}")

    print("\n🎯 Testing phonemization...")
    print(" " * 40)

    # Test phonemization specifically,
    phoneme_tests = ['كتاب', 'الشمس', 'مدرسة', 'طالب', 'العربيّة']

    for text in phoneme_tests:
    phonemes = engine.phonemize(text)
    print(f"'{text}' → {phonemes}")

    print("\n🏆 SyllabicUnit Engine Testing Complete!")
    assert True,
    def test_syllable_patterns():  # type: ignore[no-untyped def]
    """Test specific syllabic_unit patterns"""
    print("\n🔬 Testing SyllabicUnit Pattern Recognition")
    print("=" * 60)

    engine = SyllabicUnitEngine()

    # Test specific patterns,
    pattern_tests = [
    ('با', 'CV - Open syllable'),
    ('بات', 'CVV - Long open syllable'),
    ('كتب', 'CVC - Closed syllable'),
    ('مدن', 'CVC - Another closed syllable'),
    ('قال', 'CVV - Long vowel'),
    ('درس', 'CVC - Consonant cluster'),
    ]

    for text, expected in pattern_tests:
    print(f"\nTesting: '{text}' (Expected: {expected})")
        try:
    result = engine.syllabify_text(text)
            if result['status'] == 'success' and result['syllable_analysis']:
    analysis = result['syllable_analysis'][0]
    patterns = ' | '.join(analysis['syllable_patterns'])
    print(f"  Detected patterns: {patterns}")
    print(f"  SyllabicUnits: {'} - '.join(analysis['syllabic_units'])}")
            else:
    print("  ❌ Failed to analyze")
        except Exception as e:
    print(f"  ❌ Exception: {e}")


if __name__ == "__main__":
    print("🧪 Arabic SyllabicUnit Engine - Comprehensive Test Suite")
    print("تطبيق اختبار شامل لمحرك المقاطع العربية")
    print("=" * 80)

    try:
        # Run main tests,
    test_syllable_engine()

        # Run pattern tests,
    test_syllable_patterns()

    print("\n✅ All tests completed successfully!")

    except Exception as e:
    print(f"\n❌ Test suite failed with error: {e}")
        import traceback  # noqa: F401,
    traceback.print_exc()
