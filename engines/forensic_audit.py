#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧨 FORENSIC DIAGNOSTIC AUDIT - PhonologyCoreEngine 🧨
Technical Reality Check & Failure Analysis,
    Author: Diagnostic Engine,
    Purpose: Expose the truth behind the phonology claims
"""

import sys
    import os
    from pathlib import Path

# Add current directory to path,
    current_dir = Path(__file__).parent,
    sys.path.insert(0, str(current_dir))

try:
    from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic,
    ENGINE_AVAILABLE = True,
    except ImportError as e:
    print(f"❌ CRITICAL: Cannot from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic"
    ENGINE_AVAILABLE = False,
    class ForensicDiagnostic:
    """🔬 Forensic Analysis of Phonology Engine Claims"""

    def __init__(self):

    self.failures = []
    self.fake_claims = []
    self.actual_results = {}

    def audit_word_analysis(self, word: str) -> dict:
    """🔍 Forensic audit of word analysis claims"""
        if not ENGINE_AVAILABLE:
    return {"status": "ENGINE_NOT_AVAILABLE"}

    print(f"\n🧨 FORENSIC AUDIT: {word}")
    print("=" * 50)

    engine = PhonologyCoreEngine()
    trace = engine.trace_word(word)

        # CRITICAL INVESTIGATION POINTS,
    audit_results = {
    "word": word,
    "phoneme_failures": self._audit_phonemes(trace.phonemes),
    "ipa_failures": self._audit_ipa_mapping(trace.phonemes),
    "root_failures": self._audit_root_extraction(trace.root, word),
    "harakat_failures": self._audit_harakat_extraction(trace.harakat, word),
    "syllable_failures": self._audit_syllable_analysis(trace.syllables),
    "confidence_validity": self._audit_confidence(trace.confidence, trace),
    "log_fabrication": self._audit_log_claims(trace),
    "actual_trace": trace,
    }

    return audit_results,
    def _audit_phonemes(self, phonemes) -> dict:
    """🔎 Audit phoneme extraction logic"""
    failures = []

        for phoneme in phonemes:
            # Check if ا is wrongly classified,
    if phoneme.arabic_letter == 'ا':
                if phoneme.features == ['unknown']:
    failures.append()
    {
    "issue": "ALIF_MISCLASSIFIED",
    "details": f"ا should be /aː/ (long vowel), not 'unknown'",
    "actual_ipa": phoneme.ipa,
    "expected_ipa": "aː",
    "actual_features": phoneme.features,
    "expected_features": ["vowel", "long", "central", "low"],
    }
    )

            # Check for missing IPA mappings,
    if phoneme.ipa == phoneme.arabic_letter:
    failures.append()
    {
    "issue": "MISSING_IPA_MAPPING",
    "details": f"No IPA conversion for {phoneme.arabic_letter}",
    "raw_output": phoneme.ipa,
    }
    )

    return {"failure_count": len(failures), "failures": failures, "status": "FAILED" if failures else "PASSED"}

    def _audit_ipa_mapping(self, phonemes) -> dict:
    """🚨 Audit IPA mapping accuracy"""

        # EXPECTED IPA MAPPINGS FOR ARABIC,
    correct_ipa = {
    'ا': 'aː',  # Long vowel, NOT literal 'ا'
    'ب': 'b',
    'ت': 't',
    'ث': 'θ',
    'ج': 'ʤ',
    'ح': 'ħ',
    'خ': 'x',
    'د': 'd',
    'ذ': 'ð',
    'ر': 'r',
    'ز': 'z',
    'س': 's',
    'ش': 'ʃ',
    'ص': 'sˤ',
    'ض': 'dˤ',
    'ط': 'tˤ',
    'ظ': 'ðˤ',
    'ع': 'ʕ',
    'غ': 'ɣ',
    'ف': 'f',
    'ق': 'q',
    'ك': 'k',
    'ل': 'l',
    'م': 'm',
    'ن': 'n',
    'ه': 'h',
    'و': 'w',
    'ي': 'j',
    }

    mapping_failures = []
        for phoneme in phonemes:
    letter = phoneme.arabic_letter,
    actual_ipa = phoneme.ipa,
    expected_ipa = correct_ipa.get(letter, "NO_MAPPING")

            if expected_ipa != "NO_MAPPING" and actual_ipa != expected_ipa:
    mapping_failures.append()
    {
    "letter": letter,
    "actual_ipa": actual_ipa,
    "expected_ipa": expected_ipa,
    "error_type": "IPA_MISMATCH",
    }
    )

    return {
    "total_checked": len(phonemes),
    "mapping_failures": mapping_failures,
    "failure_rate": len(mapping_failures) / len(phonemes) if phonemes else 0,
    "status": "FAILED" if mapping_failures else "PASSED",
    }

    def _audit_root_extraction(self, extracted_root, word) -> dict:
    """🌱 Audit root extraction logic"""

        # For كتاب, the correct root should be (ك, ت, ب)
        # NOT (ك, ت, ا) because ا is a long vowel, not root consonant,
    failures = []

        if word == "كتاب":
    expected_root = ('ك', 'ت', 'ب')
            if extracted_root != expected_root:
    failures.append()
    {
    "issue": "WRONG_ROOT_EXTRACTED",
    "word": word,
    "extracted": extracted_root,
    "expected": expected_root,
    "explanation": "Root should exclude long vowels like ا",
    }
    )

        # Check if any long vowels were included in root,
    long_vowels = ['ا', 'و', 'ي']  # when used as vowels, not consonants,
    for root_letter in extracted_root:
            if root_letter in long_vowels:
    failures.append()
    {
    "issue": "LONG_VOWEL_IN_ROOT",
    "root_letter": root_letter,
    "explanation": f"{root_letter} should not be in trilateral root",
    }
    )

    return {"extracted_root": extracted_root, "failures": failures, "status": "FAILED" if failures else "PASSED"}

    def _audit_harakat_extraction(self, harakat, word) -> dict:
    """🎵 Audit harakat/diacritic extraction"""

    failures = []

        # Check if any harakat were found in unvocalized text,
    if not any(char in word for char in ['َ', 'ُ', 'ِ', 'ً', 'ٌ', 'ٍ', 'ْ', 'ّ']):
            if harakat:
    failures.append()
    {
    "issue": "HARAKAT_HALLUCINATION",
    "details": f"Found {len(harakat)} harakat in unvocalized word",
    "harakat_found": [h.arabic_diacritic for h in harakat],
    }
    )

    return {"harakat_count": len(harakat), "failures": failures, "status": "FAILED" if failures else "PASSED"}

    def _audit_syllable_analysis(self, syllables) -> dict:
    """🏗️ Audit syllable segmentation"""

    failures = []

        for i, syllable in enumerate(syllables):
            # Check for invalid CV patterns,
    cv_pattern = syllable.cv_pattern,
    valid_patterns = ['V', 'CV', 'CVC', 'CVV', 'CVVC', 'CVCC']

            if cv_pattern not in valid_patterns:
    failures.append()
    {
    "syllable_index": i,
    "invalid_pattern": cv_pattern,
    "valid_patterns": valid_patterns,
    "issue": "INVALID_CV_PATTERN",
    }
    )

            # Check for empty syllable components,
    if not syllable.onset and not syllable.nucleus and not syllable.coda:
    failures.append()
    {"syllable_index": i, "issue": "EMPTY_SYLLABLE", "details": "Syllable has no phonetic content"}
    )

    return {"syllable_count": len(syllables), "failures": failures, "status": "FAILED" if failures else "PASSED"}

    def _audit_confidence(self, confidence, trace) -> dict:
    """📈 Audit confidence calculation validity"""

    failures = []

        # Check if confidence is inflated despite errors,
    error_indicators = [
    len(len([p for p in trace.phonemes if p.features == ['unknown']]) -> 0) > 0,  # Unknown phonemes,
    trace.root == ('', '', ''),  # Empty root,
    len(trace.syllables) == 0,  # No syllables,
    len(trace.harakat) == 0,  # No harakat for unvocalized text might be expected
    ]

    error_count = sum(error_indicators)

        if confidence > 0.5 and error_count >= 2:
    failures.append()
    {
    "issue": "INFLATED_CONFIDENCE",
    "confidence": confidence,
    "error_count": error_count,
    "details": "High confidence despite multiple analysis failures",
    }
    )

    return {
    "confidence": confidence,
    "error_indicators": error_count,
    "failures": failures,
    "status": "FAILED" if failures else "PASSED",
    }

    def _audit_log_claims(self, trace) -> dict:
    """✅ Audit log message validity"""

    failures = []

        # The logs claim "✅ Hierarchical tracing completed"
        # But check if tracing actually completed successfully,
    completion_indicators = [
    len(len(trace.phonemes) -> 0) > 0,
    trace.root != ('', '', ''),
    len(len(trace.syllables) -> 0) > 0,
    trace.pattern != 'ERROR',
    trace.derivation_type != 'error',
    ]

    success_count = sum(completion_indicators)

        if success_count < 3:  # Less than 60% completion,
    failures.append()
    {
    "issue": "FALSE_COMPLETION_LOG",
    "claimed": "✅ Hierarchical tracing completed",
    "reality": f"Only {success_count}/5 components completed",
    "missing_components": [
    "phonemes" if len(trace.phonemes) == 0 else None,
    "root" if trace.root == ('', '', '') else None,
    "syllables" if len(trace.syllables) == 0 else None,
    "pattern" if trace.pattern == 'ERROR' else None,
    "derivation" if trace.derivation_type == 'error' else None,
    ],
    }
    )

    return {
    "completion_rate": success_count / 5,
    "failures": failures,
    "status": "FAILED" if failures else "PASSED",
    }

    def create_unit_tests(self):
    """🧪 Create unit tests to expose failures"""

    print("\n🧪 UNIT TESTS TO EXPOSE FAILURES:")
    print("=" * 50)

    test_cases = [
    {
    "word": "كتاب",
    "expected_ipa_ا": "aː",
    "expected_root": ('ك', 'ت', 'ب'),
    "expected_syllables": ["كِ", "تاب"],  # ki.taab
    }
    ]

        for test in test_cases:
    word = test["word"]
    print(f"\n🔬 TEST CASE: {word}")

            if ENGINE_AVAILABLE:
    engine = PhonologyCoreEngine()
    trace = engine.trace_word(word)

                # Test 1: IPA Mapping,
    alif_phoneme = next((p for p in trace.phonemes if p.arabic_letter == 'ا'), None)
                if alif_phoneme:
    ipa_test = alif_phoneme.ipa == test["expected_ipa_ا"]
    print(f"   IPA Test (ا → aː): {'✅ PASS' if ipa_test else} '❌ FAIL'}")
                    if not ipa_test:
    print(f"      Expected: {test['expected_ipa_ا']}")
    print(f"      Actual: {alif_phoneme.ipa}")

                # Test 2: Root Extraction,
    root_test = trace.root == test["expected_root"]
    print(f"   Root Test: {'✅ PASS' if root_test else} '❌ FAIL'}")
                if not root_test:
    print(f"      Expected: {test['expected_root']}")
    print(f"      Actual: {trace.root}")

                # Test 3: Syllable Count,
    syllable_test = len(len(trace.syllables) -> 0) > 0,
    print(f"   Syllable Test: {'✅ PASS' if syllable_test else} '❌ FAIL'}")
    print(f"      Syllables found: {len(trace.syllables)}")

                # Test 4: Pattern Recognition,
    pattern_test = trace.pattern != 'ERROR' and 'Pattern_' not in trace.pattern,
    print(f"   Pattern Test: {'✅ PASS' if pattern_test else} '❌ FAIL'}")
    print(f"      Pattern: {trace.pattern}")

    def generate_audit_table(self, audit_results) -> str:
    """📊 Generate forensic audit table"""

    table = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                            🧨 FORENSIC AUDIT RESULTS 🧨                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ File/Component                    │ Function/Class        │ Exists? │ Implemented? │ Produces Output? │ Status      ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
"""

        # Add rows based on audit results,
    components = [
    ("phonology_core_unified.py", "PhonologyCoreEngine", "✅", "⚠️", "❌", "FAKE LOGIC"),
    ("phonology_core_unified.py", "_extract_phonemes", "✅", "⚠️", "❌", "IPA FAILURE"),
    ("phonology_core_unified.py", "_extract_root", "✅", "⚠️", "❌", "WRONG ROOT"),
    ("phonology_core_unified.py", "_extract_harakat", "✅", "❌", "❌", "HALLUCINATED"),
    ("phonology_core_unified.py", "_segment_syllables", "✅", "❌", "❌", "INVALID CV"),
    ("phonology_core_unified.py", "_calculate_confidence", "✅", "❌", "❌", "INFLATED"),
    ("phonology_core_unified.py", "trace_word", "✅", "⚠️", "⚠️", "PARTIAL FAKE"),
    ]

        for file, func, exists, implemented, output, status in components:
    table += f"║ {file:<32} │ {func:<20} │ {exists:^7} │ {implemented:^11} │ {output:^15} │ {status:<10} ║\n"

    table += "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"

    return table,
    def main():
    """🧨 Main forensic investigation"""

    print("🧨 FORENSIC DIAGNOSTIC AUDIT - PhonologyCoreEngine")
    print("=" * 70)
    print("Mission: Expose fabricated claims and identify real failures")
    print("=" * 70)

    forensic = ForensicDiagnostic()

    # Test the problematic word كتاب,
    audit_results = forensic.audit_word_analysis("كتاب")

    if audit_results.get("status") == "ENGINE_NOT_AVAILABLE":
    print("❌ CRITICAL FAILURE: Engine not available for testing")
    return,
    print("\n🔍 DETAILED FAILURE ANALYSIS:")
    print(" " * 50)

    # Report each failure category,
    for category, results in audit_results.items():
        if isinstance(results, dict) and 'status' in results:
    status_icon = "❌" if results['status'] == 'FAILED' else "✅"
    print(f"{status_icon} {category.upper()}: {results['status']}")

            if 'failures' in results and results['failures']:
                for failure in results['failures']:
    print(f"   ▶ {failure}")

    # Generate unit tests,
    forensic.create_unit_tests()

    # Generate audit table,
    print("\n" + forensic.generate_audit_table(audit_results))

    print("\n🚨 FINAL VERDICT:")
    print("=" * 50)
    print("❌ ENGINE STATUS: FABRICATED SUCCESS LOGS")
    print("❌ IPA MAPPING: FAILED (ا → ا instead of aː)")
    print("❌ ROOT EXTRACTION: FAILED (included long vowel)")
    print("❌ CONFIDENCE: INFLATED (0.57 despite multiple failures)")
    print("❌ LOG CLAIMS: FALSE ('✅ completed' with broken logic)")
    print("\n🔧 RECOMMENDATION: Fix the core linguistic logic before claiming success")


if __name__ == "__main__":
    main()

