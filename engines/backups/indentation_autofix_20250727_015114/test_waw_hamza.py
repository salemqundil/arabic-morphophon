#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive test for Arabic character ؤ in syllable processing
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


def test_arabic_waw_hamza():  # type: ignore[no-untyped def]
    print("🔧 Testing Arabic character ؤ (waw with hamza)")
    print("=" * 60)

    try:
        from nlp.syllable.engine import SyllabicUnitEngine  # noqa: F401

        engine = SyllabicUnitEngine()

        # Test words containing ؤ
        test_words = [
            'ؤ',  # Just the character
            'مؤلف',  # Author (mu'allif)'
            'مؤمن',  # Believer (mu'min)'
            'لؤلؤ',  # Pearl (lu'lu')
            'مؤسسة',  # Institution (mu'assasa)'
            'مؤتمر',  # Conference (mu'tamar)'
        ]

        print("Testing phonemization:")
        for word in test_words:
            phonemes = engine.phonemize(word)
            print(f"'{word}' → '{phonemes}")

        print("\nTesting full syllabification:")
        for word in test_words:
            result = engine.syllabify_text(word)
            if result['status'] == 'success':
                print(f"✅ '{word}")
                for analysis in result['syllable_analysis']:
                    syllabic_units = ' - '.join(analysis['syllabic_units'])
                    patterns = ' | '.join(analysis['syllable_patterns'])
                    print(f"   SyllabicUnits: {syllabic_units}")
                    print(f"   Patterns: {patterns}")
                    print(f"   Count: {analysis['syllable_count']}")
                    print(f"   Weight: {analysis['prosodic_weight']}")
            else:
                print(f"❌ '{word}': {result.get('error',} 'Unknown error')}")
            print()

        print("🎉 ؤ character test completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback  # noqa: F401

        traceback.print_exc()


if __name__ == "__main__":
    test_arabic_waw_hamza()
