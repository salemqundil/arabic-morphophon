# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc

from __future__ import annotations
import re  # noqa: F401
from collections import Counter  # noqa: F401
from typing import List, Dict

import matplotlib.pyplot as plt  # noqa: F401
import base64  # noqa: F401
from io import BytesIO  # noqa: F401

# ------------------ 1. ثابتات ----------------------------------------- #
SHORT = {"َ": "a", "ُ": "u", "ِ": "i"}
LONG = {"ا": "ā", "و": "ū", "ي": "ī"}

PHONEMES: Dict[str, Dict[str, str]] = {
    "ب": {"cv": "C", "place": "labial"},
    "م": {"cv": "C", "place": "labial"},
    "ف": {"cv": "C", "place": "labiodental"},
    "ت": {"cv": "C", "place": "dental"},
    "د": {"cv": "C", "place": "dental"},
    "ط": {"cv": "C", "place": "dental"},
    "ث": {"cv": "C", "place": "inter dental"},
    "ذ": {"cv": "C", "place": "inter dental"},
    "س": {"cv": "C", "place": "alveolar"},
    "ص": {"cv": "C", "place": "alveolar"},
    "ز": {"cv": "C", "place": "alveolar"},
    "ش": {"cv": "C", "place": "alveo palatal"},
    "ج": {"cv": "C", "place": "palatal"},
    "ي": {"cv": "V", "place": "palatal"},
    "ك": {"cv": "C", "place": "velar"},
    "ق": {"cv": "C", "place": "uvular"},
    "غ": {"cv": "C", "place": "uvular"},
    "خ": {"cv": "C", "place": "uvular"},
    "ح": {"cv": "C", "place": "pharyngeal"},
    "ع": {"cv": "C", "place": "pharyngeal"},
    "ه": {"cv": "C", "place": "glottal"},
    "ʔ": {"cv": "C", "place": "glottal"},  # همزة قطع
    "ɂ": {"cv": "C", "place": "glottal"},  # همزة وصل
    "َ": {"cv": "v", "place": "vowel", "type": "fatha"},
    "ُ": {"cv": "v", "place": "vowel", "type": "damma"},
    "ِ": {"cv": "v", "place": "vowel", "type": "kasra"},
    "ْ": {"cv": "", "place": "sukūn"},
    "ا": {"cv": "V", "place": "pharyngeal", "type": "long"},
    "و": {"cv": "V", "place": "labial", "type": "long"},
}

HAMZA_QAT = r"[ءأإؤئ]"
HAMZA_WASL = r"(?:ٱ|ا(?![ء ي]))"  # ألف بداية بدون همزة تليها لام


# ------------------ 2. Fonemizer -------------------------------------- #
def _normalise(text: str) -> str:
    """TODO: Add docstring."""
    text = re.sub(HAMZA_QAT, "ʔ", text)
    text = re.sub(r"\b" + HAMZA_WASL, "ɂ", text)
    return text


def phonemize(text: str) -> List[Dict[str, str]]:
    """TODO: Add docstring."""
    text = _normalise(text)
    tokens: List[Dict[str, str]] = []

    for ch in text:
    info = PHONEMES.get(ch)
        if not info:
    continue
    tokens.append({"char": ch, **info})
    return tokens


def syllable_pattern(tokens: List[Dict[str, str]]) -> str:
    """TODO: Add docstring."""
    return "".join(t["cv"] for t in tokens if t["cv"])


# ------------------ 3. Heat-map --------------------------------------- #
def articulation_heatmap(text: str, *, show: bool = False) -> str:
    """TODO: Add docstring."""
    try:
    phs = phonemize(text)
    places = [t["place"] for t in phs if t["cv"] == "C"]
    counts = Counter(places)

    labels = list(counts)
    values = [counts[p] for p in labels]

    fig, ax = plt.subplots(figsize=(4, 1))
    ax.imshow([values], aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
        for spine in ax.spines.values():
    spine.set_visible(False)

    buf = BytesIO()
        try:
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            if show:
                import PIL.Image as Image  # noqa: F401

    Image.open(buf).show()
    return base64.b64encode(buf.getvalue()).decode()
    finally:
    buf.close()
    plt.close(fig)
    except Exception as e:
    raise RuntimeError(f"Failed to generate articulation heatmap: {e}")
