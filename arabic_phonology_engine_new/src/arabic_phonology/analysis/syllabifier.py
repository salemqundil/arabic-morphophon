from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))
from phonology.classifier import Phoneme


@dataclass
class Syllable:
    phones: Sequence[Phoneme]
    pattern: str
    weight: float


def _pattern(chunk: Sequence[Phoneme]) -> str:
    cv = []
    for p in chunk:
        # Safe metadata access with default fallback
        phoneme_type = p.meta.get("type", "C")  # Default to consonant
        t = "C" if phoneme_type == "C" else "V"
        is_geminated = p.meta.get("geminated", False)  # Default to False
        cv.extend([t, t] if is_geminated else [t])
    return "".join(cv)


def _weight(chunk: Sequence[Phoneme]) -> float:
    # Safe metadata access with default fallback
    total_weight = 0.0
    for p in chunk:
        weight = p.meta.get("acoustic_weight", 0.5)  # Default weight
        # Handle None values safely
        if weight is None:
            weight = 0.5
        try:
            total_weight += float(weight)
        except (TypeError, ValueError):
            # If weight is not convertible to float, use default
            total_weight += 0.5
    return total_weight


def syllabify(phones: Sequence[Phoneme]) -> List[Syllable]:
    out: List[Syllable] = []
    buf: List[Phoneme] = []

    i = 0
    while i < len(phones):
        buf.append(phones[i])
        # Safe metadata access with default fallback
        current_type = phones[i].meta.get("type", "C")
        if current_type == "V":
            nxt = phones[i + 1] if i + 1 < len(phones) else None
            nxt2 = phones[i + 2] if i + 2 < len(phones) else None
            # After hitting a vowel (V):
            # Case 1: V + None or V + V → flush (end syllable)
            # Case 2: V + C + (None or V) → take consonant and flush
            # Case 3: V + C + C → leave consonant for next syllable
            if nxt is None or nxt.meta.get("type", "C") == "V":
                _flush(buf, out)
            elif nxt.meta.get("type", "C") == "C" and (
                nxt2 is None or nxt2.meta.get("type", "C") == "V"
            ):
                buf.append(nxt)
                i += 1
                _flush(buf, out)
        i += 1
    if buf:
        _flush(buf, out)
    return out


def _flush(buf: List[Phoneme], out: List[Syllable]) -> None:
    out.append(Syllable(buf.copy(), _pattern(buf), _weight(buf)))
    buf.clear()


# Test functionality when run directly
if __name__ == "__main__":
    # Creates test_phones variable
    # Tests syllabify function
    # Shows results
    test_phones = [
        Phoneme(
            char="p",
            meta={"type": "C", "acoustic_weight": 0.5, "geminated": False},
        ),
        Phoneme(
            char="a",
            meta={"type": "V", "acoustic_weight": 1.0, "geminated": False},
        ),
        Phoneme(
            char="t",
            meta={"type": "C", "acoustic_weight": 0.6, "geminated": False},
        ),
        Phoneme(
            char="a",
            meta={"type": "V", "acoustic_weight": 1.0, "geminated": False},
        ),
        Phoneme(
            char="k",
            meta={"type": "C", "acoustic_weight": 0.7, "geminated": False},
        ),
        Phoneme(
            char="a",
            meta={"type": "V", "acoustic_weight": 1.0, "geminated": False},
        ),
    ]

    print("🧪 Testing Syllabifier...")
    result = syllabify(test_phones)

    print(f"✅ Input: {''.join(p.char for p in test_phones)}")
    print(f"✅ Syllables: {len(result)}")

    for i, syl in enumerate(result):
        phones_str = "".join(p.char for p in syl.phones)
        print(f"   {i+1}. {phones_str} - Pattern: {syl.pattern} - Weight: {syl.weight}")

    print("🎯 Syllabifier working correctly!")
