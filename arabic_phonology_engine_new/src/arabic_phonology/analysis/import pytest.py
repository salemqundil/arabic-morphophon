import pytest
from .syllabifier import syllabify, _pattern, _weight, _flush, Syllable
from .classifier import Phoneme


class TestPattern:
    """Test the _pattern function for CV pattern generation."""

    def test_simple_consonant(self):
        """Test single consonant pattern."""
        phoneme = Phoneme(char="p", meta={"type": "C"})
        result = _pattern([phoneme])
        assert result == "C"

    def test_simple_vowel(self):
        """Test single vowel pattern."""
        phoneme = Phoneme(char="a", meta={"type": "V"})
        result = _pattern([phoneme])
        assert result == "V"

    def test_cv_pattern(self):
        """Test consonant-vowel pattern."""
        phonemes = [
            Phoneme(char="p", meta={"type": "C"}),
            Phoneme(char="a", meta={"type": "V"}),
        ]
        result = _pattern(phonemes)
        assert result == "CV"

    def test_geminated_consonant(self):
        """Test geminated consonant creates double C pattern."""
        phoneme = Phoneme(char="t", meta={"type": "C", "geminated": True})
        result = _pattern([phoneme])
        assert result == "CC"

    def test_geminated_vowel(self):
        """Test geminated vowel creates double V pattern."""
        phoneme = Phoneme(char="a", meta={"type": "V", "geminated": True})
        result = _pattern([phoneme])
        assert result == "VV"

    def test_missing_type_defaults_to_consonant(self):
        """Test phoneme without type metadata defaults to consonant."""
        phoneme = Phoneme(char="x", meta={})
        result = _pattern([phoneme])
        assert result == "C"

    def test_missing_geminated_defaults_to_false(self):
        """Test phoneme without geminated metadata defaults to single."""
        phoneme = Phoneme(char="p", meta={"type": "C"})
        result = _pattern([phoneme])
        assert result == "C"

    def test_empty_sequence(self):
        """Test empty phoneme sequence."""
        result = _pattern([])
        assert result == ""


class TestWeight:
    """Test the _weight function for acoustic weight calculation."""

    def test_single_phoneme_weight(self):
        """Test weight calculation for single phoneme."""
        phoneme = Phoneme(char="a", meta={"acoustic_weight": 1.5})
        result = _weight([phoneme])
        assert result == 1.5

    def test_multiple_phonemes_weight(self):
        """Test weight calculation for multiple phonemes."""
        phonemes = [
            Phoneme(char="p", meta={"acoustic_weight": 0.5}),
            Phoneme(char="a", meta={"acoustic_weight": 1.0}),
            Phoneme(char="t", meta={"acoustic_weight": 0.6}),
        ]
        result = _weight(phonemes)
        assert result == 2.1

    def test_missing_weight_defaults(self):
        """Test phoneme without weight metadata defaults to 0.5."""
        phoneme = Phoneme(char="x", meta={})
        result = _weight([phoneme])
        assert result == 0.5

    def test_empty_sequence_weight(self):
        """Test weight calculation for empty sequence."""
        result = _weight([])
        assert result == 0.0


class TestFlush:
    """Test the _flush function for syllable creation."""

    def test_flush_creates_syllable(self):
        """Test that flush creates a syllable and clears buffer."""
        buf = [
            Phoneme(char="p", meta={"type": "C", "acoustic_weight": 0.5}),
            Phoneme(char="a", meta={"type": "V", "acoustic_weight": 1.0}),
        ]
        out = []

        _flush(buf, out)

        assert len(out) == 1
        assert len(buf) == 0  # Buffer should be cleared
        assert out[0].pattern == "CV"
        assert out[0].weight == 1.5
        assert len(out[0].phones) == 2

    def test_flush_copies_buffer(self):
        """Test that flush copies buffer content, not references."""
        buf = [Phoneme(char="a", meta={"type": "V"})]
        out = []

        _flush(buf, out)

        # Modify original buffer
        buf.append(Phoneme(char="b", meta={"type": "C"}))

        # Syllable should still have only original phoneme
        assert len(out[0].phones) == 1
        assert out[0].phones[0].char == "a"


class TestSyllabify:
    """Test the main syllabify function."""

    def test_single_vowel(self):
        """Test syllabification of single vowel."""
        phones = [Phoneme(char="a", meta={"type": "V"})]
        result = syllabify(phones)

        assert len(result) == 1
        assert result[0].pattern == "V"

    def test_cv_syllable(self):
        """Test consonant-vowel syllable."""
        phones = [
            Phoneme(char="p", meta={"type": "C"}),
            Phoneme(char="a", meta={"type": "V"}),
        ]
        result = syllabify(phones)

        assert len(result) == 1
        assert result[0].pattern == "CV"

    def test_cvc_syllable(self):
        """Test consonant-vowel-consonant syllable."""
        phones = [
            Phoneme(char="p", meta={"type": "C"}),
            Phoneme(char="a", meta={"type": "V"}),
            Phoneme(char="t", meta={"type": "C"}),
        ]
        result = syllabify(phones)

        assert len(result) == 1
        assert result[0].pattern == "CVC"

    def test_two_syllables_vcv(self):
        """Test VCV pattern creates two syllables (V.CV)."""
        phones = [
            Phoneme(char="a", meta={"type": "V"}),
            Phoneme(char="p", meta={"type": "C"}),
            Phoneme(char="a", meta={"type": "V"}),
        ]
        result = syllabify(phones)

        assert len(result) == 2
        assert result[0].pattern == "V"
        assert result[1].pattern == "CV"

    def test_two_syllables_vccv(self):
        """Test VCCV pattern creates two syllables (VC.CV)."""
        phones = [
            Phoneme(char="a", meta={"type": "V"}),
            Phoneme(char="p", meta={"type": "C"}),
            Phoneme(char="t", meta={"type": "C"}),
            Phoneme(char="a", meta={"type": "V"}),
        ]
        result = syllabify(phones)

        assert len(result) == 2
        assert result[0].pattern == "VC"
        assert result[1].pattern == "CV"

    def test_complex_word_pataka(self):
        """Test complex word 'pataka' syllabification."""
        phones = [
            Phoneme(char="p", meta={"type": "C", "acoustic_weight": 0.5}),
            Phoneme(char="a", meta={"type": "V", "acoustic_weight": 1.0}),
            Phoneme(char="t", meta={"type": "C", "acoustic_weight": 0.6}),
            Phoneme(char="a", meta={"type": "V", "acoustic_weight": 1.0}),
            Phoneme(char="k", meta={"type": "C", "acoustic_weight": 0.7}),
            Phoneme(char="a", meta={"type": "V", "acoustic_weight": 1.0}),
        ]
        result = syllabify(phones)

        assert len(result) == 3
        assert result[0].pattern == "CV"  # pa
        assert result[1].pattern == "CV"  # ta
        assert result[2].pattern == "CV"  # ka

    def test_vowel_sequence(self):
        """Test vowel sequence creates separate syllables."""
        phones = [
            Phoneme(char="a", meta={"type": "V"}),
            Phoneme(char="e", meta={"type": "V"}),
            Phoneme(char="i", meta={"type": "V"}),
        ]
        result = syllabify(phones)

        assert len(result) == 3
        for syl in result:
            assert syl.pattern == "V"

    def test_consonant_cluster_start(self):
        """Test word starting with consonant cluster."""
        phones = [
            Phoneme(char="s", meta={"type": "C"}),
            Phoneme(char="p", meta={"type": "C"}),
            Phoneme(char="a", meta={"type": "V"}),
        ]
        result = syllabify(phones)

        assert len(result) == 1
        assert result[0].pattern == "CCV"

    def test_missing_metadata_handling(self):
        """Test handling of phonemes with missing metadata."""
        phones = [
            Phoneme(char="p", meta={}),  # No metadata
            Phoneme(char="a", meta={"type": "V"}),
            Phoneme(char="t", meta={}),  # No metadata
        ]
        result = syllabify(phones)

        assert len(result) == 1
        assert result[0].pattern == "CVC"  # Defaults to consonant

    def test_empty_input(self):
        """Test empty phoneme sequence."""
        result = syllabify([])
        assert result == []

    def test_final_consonants_included(self):
        """Test that final consonants are included in last syllable."""
        phones = [
            Phoneme(char="a", meta={"type": "V"}),
            Phoneme(char="p", meta={"type": "C"}),
            Phoneme(char="t", meta={"type": "C"}),
            Phoneme(char="s", meta={"type": "C"}),
        ]
        result = syllabify(phones)

        assert len(result) == 1
        assert result[0].pattern == "VCCC"


class TestSyllableDataclass:
    """Test the Syllable dataclass."""

    def test_syllable_creation(self):
        """Test creating a Syllable instance."""
        phones = [Phoneme(char="a", meta={"type": "V"})]
        syl = Syllable(phones=phones, pattern="V", weight=1.0)

        assert syl.phones == phones
        assert syl.pattern == "V"
        assert syl.weight == 1.0

    def test_syllable_equality(self):
        """Test Syllable equality comparison."""
        phones = [Phoneme(char="a", meta={"type": "V"})]
        syl1 = Syllable(phones=phones, pattern="V", weight=1.0)
        syl2 = Syllable(phones=phones, pattern="V", weight=1.0)

        assert syl1 == syl2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_only_consonants(self):
        """Test sequence with only consonants."""
        phones = [
            Phoneme(char="p", meta={"type": "C"}),
            Phoneme(char="t", meta={"type": "C"}),
            Phoneme(char="k", meta={"type": "C"}),
        ]
        result = syllabify(phones)

        assert len(result) == 1
        assert result[0].pattern == "CCC"

    def test_geminated_in_syllabification(self):
        """Test geminated phonemes in syllabification context."""
        phones = [
            Phoneme(char="a", meta={"type": "V"}),
            Phoneme(char="t", meta={"type": "C", "geminated": True}),
            Phoneme(char="a", meta={"type": "V"}),
        ]
        result = syllabify(phones)

        assert len(result) == 2
        assert result[0].pattern == "VCC"  # Geminated t contributes to both syllables
        assert result[1].pattern == "V"

    def test_none_metadata_handling(self):
        """Test phonemes with None as metadata values."""
        phoneme = Phoneme(char="x", meta={"type": None, "acoustic_weight": None})

        # Should not crash and use defaults
        pattern_result = _pattern([phoneme])
        weight_result = _weight([phoneme])

        assert pattern_result == "C"  # Default to consonant
        assert weight_result == 0.5  # Default weight
