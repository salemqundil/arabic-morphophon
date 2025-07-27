from __future__ import annotations

"""Arabic Phoneme Generator
===========================
Self‑contained engine for generating Arabic syllables, applying
phonological rules, and producing basic morphological patterns **without
heavy external dependencies**.  Previous runs failed with
``ModuleNotFoundError: No module named 'micropip'`` because the runtime
attempted to auto‑install packages (e.g. *matplotlib*, *numpy*) via
*micropip* (used by Pyodide environments).  This revision removes those
optional libraries and falls back gracefully when plotting is requested.

Highlights
----------
* Eliminates hard dependency on **numpy** and **matplotlib**; uses the
  standard ``random`` module instead.
* Lazy‑loads ``matplotlib`` inside ``plot_syllable`` only if available.
* Keeps and **extends** the minimal test suite.
"""

from dataclasses import dataclass, field
from random import choice
from typing import Dict, List, Sequence, Tuple, Optional, Any, Union
import itertools
import sys

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhonemeFeature:
    """Represents phonological features of an Arabic sound."""

    place: str | None = None
    manner: str | None = None
    voicing: str | None = None
    type: str | None = None  # "vowel", "diacritic", …
    duration: float | None = None


# ---------------------------------------------------------------------------
#  Core generator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ArabicPhonemeGenerator:
    """Generate and analyse Arabic phoneme combinations."""

    # -----------------------------
    #   Inventories
    # -----------------------------

    consonants: List[str] = field(
        default_factory=lambda: list("بتثجحخدذرزسشصضطظعغفقكلمنهء")
    )
    vowels: List[str] = field(default_factory=lambda: ["َ", "ُ", "ِ"])
    long_vowels: List[str] = field(default_factory=lambda: ["ا", "و", "ي"])
    sukoon: List[str] = field(default_factory=lambda: ["ْ"])
    shadda: List[str] = field(default_factory=lambda: ["ّ"])
    tanwin: List[str] = field(default_factory=lambda: ["ً", "ٌ", "ٍ"])

    # Articulatory feature space (simplified)
    features: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "place": ["bilabial", "dental", "velar", "pharyngeal", "glottal"],
            "manner": ["stop", "fricative", "trill", "approximant"],
            "voicing": ["voiced", "voiceless"],
        }
    )

    # Classical Arabic morphological weights (sample)
    weights: Dict[str, str] = field(
        default_factory=lambda: {
            "فعل مجرد ماضي": "فَعَلَ",
            "مصدر قياسي": "فَعْل",
            "اسم مشتق": "فَاعِل",
        }
    )

    # Computed attributes (set in __post_init__)
    phoneme_map: Dict[str, PhonemeFeature] = field(init=False)
    _position_map: Dict[str, List[str]] = field(init=False)

    # -------------------------------------------------------------------
    #  Construction
    # -------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Initialize computed attributes after instance creation."""
        self.phoneme_map = self._create_phoneme_map()
        self._position_map = {
            "C": self.consonants,
            "V": self.vowels,
            "L": self.long_vowels,
            "D": self.sukoon + self.shadda + self.tanwin,
            "S": self.shadda,
        }

    # -------------------------------------------------------------------
    #  Mapping helpers
    # -------------------------------------------------------------------

    def _create_phoneme_map(self) -> Dict[str, PhonemeFeature]:
        """Create a mapping of phonemes to their phonological features."""
        mapping: Dict[str, PhonemeFeature] = {}
        rand = choice  # local ref for speed

        # Map consonants to their features
        for c in self.consonants:
            mapping[c] = PhonemeFeature(
                place=rand(self.features["place"]),
                manner=rand(self.features["manner"]),
                voicing=rand(self.features["voicing"]),
            )

        # Map vowels to their features
        for v in (*self.vowels, *self.long_vowels):
            mapping[v] = PhonemeFeature(
                type="vowel", duration=1.0 if v in self.long_vowels else 0.5
            )

        # Map diacritics to their features
        for sym in (*self.sukoon, *self.shadda, *self.tanwin):
            mapping[sym] = PhonemeFeature(type="diacritic")

        return mapping

    # -------------------------------------------------------------------
    #  Combination generation
    # -------------------------------------------------------------------

    def generate_all_combinations(self, template: str = "CVC") -> List[Tuple[str, ...]]:
        """Generate all possible phoneme combinations based on a template.

        Args:
            template: A string representing the syllable structure,
                     where C=consonant, V=vowel, L=long vowel, etc.

        Returns:
            A list of tuples, each containing a possible phoneme combination.

        Raises:
            ValueError: If the template contains unknown symbols.
        """
        try:
            pools = [self._position_map[ch] for ch in template]
        except KeyError as err:
            raise ValueError(f"Unknown symbol in template: {err}") from None
        return list(itertools.product(*pools))

    # -------------------------------------------------------------------
    #  Phonological filter
    # -------------------------------------------------------------------

    def _valid(self, combo: Tuple[str, ...]) -> bool:
        """Check if a phoneme combination is phonologically valid in Arabic.

        Applies the following constraints:
        - No gemination (shadda) can follow sukoon
        - Tanwin requires a vowel
        - No geminated long vowels
        - No bare consonant clusters without vowels

        Args:
            combo: A tuple of phonemes to check

        Returns:
            True if the combination is valid, False otherwise
        """
        s = "".join(combo)

        # Check for shadda after sukoon (not allowed)
        if "ّْ" in s:
            return False  # Gemination cannot follow sukoon

        # Check for tanwin without vowel (not allowed)
        if any(t in s for t in self.tanwin) and not any(v in s for v in self.vowels):
            return False  # Tanwin requires vowel

        # Check for geminated long vowels (not allowed)
        if "ّ" in s and any(lv in s for lv in self.long_vowels):
            return False  # No geminated long vowels

        # Check for consonant clusters without vowels (not allowed)
        for a, b in zip(s, s[1:]):
            if (
                a in self.consonants
                and b in self.consonants
                and not any(v in s for v in (*self.vowels, *self.long_vowels))
            ):
                return False  # No bare CC clusters

        return True

    def apply_phonological_rules(self, combos: List[Tuple[str, ...]]) -> List[str]:
        """Apply phonological rules to filter valid combinations.

        Args:
            combos: A list of phoneme combinations to filter

        Returns:
            A list of valid phoneme combinations as strings
        """
        return ["".join(c) for c in combos if self._valid(c)]

    # -------------------------------------------------------------------
    #  Morphological patterns
    # -------------------------------------------------------------------

    def generate_morph_patterns(self, pattern_type: str) -> List[str]:
        """Generate morphological patterns based on a template type.

        Args:
            pattern_type: The type of morphological pattern (key in self.weights)

        Returns:
            A list of strings representing valid morphological patterns
        """
        weight = self.weights.get(pattern_type)
        if not weight:
            return []

        structure: List[Sequence[str]] = []
        for ch in weight:
            if ch in "فعلا":
                structure.append(self.consonants)
            elif ch in "َُِ":
                structure.append(self.vowels)
            elif ch in "اوي":
                structure.append(self.long_vowels)
            else:
                structure.append([ch])

        return ["".join(p) for p in itertools.product(*structure)]

    # -------------------------------------------------------------------
    #  Temporal analysis & optional plot
    # -------------------------------------------------------------------

    def temporal_analysis(self, syllable: str) -> Dict[str, Any]:
        """Analyze the temporal structure of a syllable.

        Args:
            syllable: The syllable to analyze

        Returns:
            A dictionary with syllable, duration, and component information
        """
        t = 0.0
        parts: List[Tuple[str, str, float]] = []

        for ch in syllable:
            if ch in self.consonants:
                t += 0.3
                parts.append(("C", ch, 0.3))
            elif ch in self.vowels:
                t += 0.5
                parts.append(("V", ch, 0.5))
            elif ch in self.long_vowels:
                t += 1.0
                parts.append(("Vː", ch, 1.0))
            elif ch == "ّ":
                t += 0.3
                parts.append(("Gem", ch, 0.3))
            elif ch == "ْ":
                parts.append(("Suk", ch, 0.0))
            elif ch in self.tanwin:
                t += 0.2
                parts.append(("Tan", ch, 0.2))

        return {"syllable": syllable, "duration": t, "components": parts}

    def plot_syllable(self, syllable: str) -> Dict[str, Any]:
        """Display a simple horizontal bar chart if matplotlib is present.

        Args:
            syllable: The syllable to plot

        Returns:
            The temporal analysis data (same as temporal_analysis())
        """
        data = self.temporal_analysis(syllable)

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("matplotlib not available – skipping plot.", file=sys.stderr)
            return data

        labels = [f"{c[1]}\n({c[0]})" for c in data["components"]]
        durs = [c[2] for c in data["components"]]

        fig, ax = plt.subplots()
        ax.barh(labels, durs)
        ax.set_xlabel("Duration (AU)")
        ax.set_title(syllable)
        plt.tight_layout()
        plt.show()

        return data

    # -------------------------------------------------------------------
    #  Full dataset builder
    # -------------------------------------------------------------------

    def build_dataset(self, templates: Sequence[str] | None = None) -> List[str]:
        """Build a complete dataset of valid syllables from templates.

        Args:
            templates: A list of syllable templates to use (default: ["CV", "CVC", "VC"])

        Returns:
            A list of all valid syllables
        """
        templates = templates or ["CV", "CVC", "VC"]
        out: List[str] = []

        for tp in templates:
            out.extend(
                self.apply_phonological_rules(self.generate_all_combinations(tp))
            )

        return out


# ---------------------------------------------------------------------------
#  Self‑tests
# ---------------------------------------------------------------------------


def _basic_tests() -> None:
    """Run basic tests to verify the generator works correctly."""
    gen = ArabicPhonemeGenerator()

    # 1. _position_map exists
    assert hasattr(gen, "_position_map"), "_position_map attribute not found"

    # 2. generate_all_combinations works
    combos = gen.generate_all_combinations("CV")
    assert len(combos) == len(gen.consonants) * len(
        gen.vowels
    ), "Incorrect number of combinations"

    # 3. filter reduces or equals size
    filtered = gen.apply_phonological_rules(combos)
    assert (
        0 < len(filtered) <= len(combos)
    ), "Filtering did not produce expected results"

    # 4. morphological pattern size non‑zero
    patterns = gen.generate_morph_patterns("فعل مجرد ماضي")
    assert patterns, "No patterns generated for فعل مجرد ماضي"

    # 5. Test temporal analysis
    analysis = gen.temporal_analysis("بَر")
    assert analysis["duration"] > 0, "Temporal analysis failed"

    print(
        "✅ tests passed –",
        len(filtered),
        "valid CV syllables,",
        len(patterns),
        "patterns for 'فعل مجرد ماضي'.",
    )


if __name__ == "__main__":
    _basic_tests()
    # demonstration
    g = ArabicPhonemeGenerator()
    syllables = g.build_dataset(["CV"])[:20]
    print("Sample CV syllables:", syllables)

    if len(syllables) > 0:
        # Demonstrate temporal analysis with the first syllable
        analysis = g.temporal_analysis(syllables[0])
        print(
            f"Temporal analysis of '{syllables[0]}': {analysis['duration']}s duration"
        )

        # Try to plot if available
        g.plot_syllable(syllables[0])
