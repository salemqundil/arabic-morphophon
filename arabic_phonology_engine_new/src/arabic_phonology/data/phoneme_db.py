"""
Arabic Phoneme Database - Core Implementation
Expert-level phonological database for Arabic language processing
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PhonemeType(Enum):
    """Phoneme classification types."""
    CONSONANT = "consonant"
    VOWEL = "vowel"
    DIACRITIC = "diacritic"
    GLOTTAL = "glottal"

class ArticulationPlace(Enum):
    """Places of articulation for consonants."""
    BILABIAL = "bilabial"
    DENTAL = "dental"
    ALVEOLAR = "alveolar"
    PALATAL = "palatal"
    VELAR = "velar"
    UVULAR = "uvular"
    PHARYNGEAL = "pharyngeal"
    GLOTTAL = "glottal"

class ArticulationManner(Enum):
    """Manners of articulation for consonants."""
    STOP = "stop"
    FRICATIVE = "fricative"
    NASAL = "nasal"
    LIQUID = "liquid"
    GLIDE = "glide"

@dataclass
class Phoneme:
    """Arabic phoneme with linguistic properties."""
    symbol: str
    ipa: str
    arabic: str
    type: PhonemeType
    place: Optional[ArticulationPlace] = None
    manner: Optional[ArticulationManner] = None
    frequency: float = 0.0
    
class PhonemeDatabase:
    """Expert Arabic Phoneme Database with zero tolerance architecture."""
    
    def __init__(self):
        self._phonemes = self._initialize_phonemes()
        self._symbol_to_phoneme = {p.symbol: p for p in self._phonemes}
        self._arabic_to_phoneme = {p.arabic: p for p in self._phonemes}
        
    def _initialize_phonemes(self) -> List[Phoneme]:
        """Initialize comprehensive Arabic phoneme inventory."""
        return [
            # Root consonants
            Phoneme("b", "b", "ب", PhonemeType.CONSONANT, ArticulationPlace.BILABIAL, ArticulationManner.STOP, 0.4),
            Phoneme("t", "t", "ت", PhonemeType.CONSONANT, ArticulationPlace.ALVEOLAR, ArticulationManner.STOP, 0.5),
            Phoneme("j", "d͡ʒ", "ج", PhonemeType.CONSONANT, ArticulationPlace.PALATAL, ArticulationManner.FRICATIVE, 0.3),
            Phoneme("d", "d", "د", PhonemeType.CONSONANT, ArticulationPlace.ALVEOLAR, ArticulationManner.STOP, 0.4),
            Phoneme("r", "r", "ر", PhonemeType.CONSONANT, ArticulationPlace.ALVEOLAR, ArticulationManner.LIQUID, 0.5),
            Phoneme("z", "z", "ز", PhonemeType.CONSONANT, ArticulationPlace.ALVEOLAR, ArticulationManner.FRICATIVE, 0.3),
            Phoneme("s", "s", "س", PhonemeType.CONSONANT, ArticulationPlace.ALVEOLAR, ArticulationManner.FRICATIVE, 0.4),
            Phoneme("sh", "ʃ", "ش", PhonemeType.CONSONANT, ArticulationPlace.PALATAL, ArticulationManner.FRICATIVE, 0.3),
            Phoneme("f", "f", "ف", PhonemeType.CONSONANT, ArticulationPlace.BILABIAL, ArticulationManner.FRICATIVE, 0.4),
            Phoneme("q", "q", "ق", PhonemeType.CONSONANT, ArticulationPlace.UVULAR, ArticulationManner.STOP, 0.3),
            Phoneme("k", "k", "ك", PhonemeType.CONSONANT, ArticulationPlace.VELAR, ArticulationManner.STOP, 0.6),
            Phoneme("l", "l", "ل", PhonemeType.CONSONANT, ArticulationPlace.ALVEOLAR, ArticulationManner.LIQUID, 0.8),
            Phoneme("m", "m", "م", PhonemeType.CONSONANT, ArticulationPlace.BILABIAL, ArticulationManner.NASAL, 0.7),
            Phoneme("n", "n", "ن", PhonemeType.CONSONANT, ArticulationPlace.ALVEOLAR, ArticulationManner.NASAL, 0.6),
            Phoneme("h", "h", "ه", PhonemeType.CONSONANT, ArticulationPlace.GLOTTAL, ArticulationManner.FRICATIVE, 0.5),
            Phoneme("w", "w", "و", PhonemeType.CONSONANT, ArticulationPlace.VELAR, ArticulationManner.GLIDE, 0.6),
            Phoneme("y", "j", "ي", PhonemeType.CONSONANT, ArticulationPlace.PALATAL, ArticulationManner.GLIDE, 0.6),
            
            # Vowels
            Phoneme("a", "a", "ا", PhonemeType.VOWEL, frequency=0.9),
            Phoneme("i", "i", "ي", PhonemeType.VOWEL, frequency=0.6),
            Phoneme("u", "u", "و", PhonemeType.VOWEL, frequency=0.6),
            Phoneme("aa", "aː", "آ", PhonemeType.VOWEL, frequency=0.7),
            Phoneme("ii", "iː", "ی", PhonemeType.VOWEL, frequency=0.5),
            Phoneme("uu", "uː", "ؤ", PhonemeType.VOWEL, frequency=0.5),
        ]
    
    def get_phoneme(self, symbol: str) -> Optional[Phoneme]:
        """Get phoneme by symbol."""
        return self._symbol_to_phoneme.get(symbol)
    
    def get_phoneme_by_arabic(self, arabic: str) -> Optional[Phoneme]:
        """Get phoneme by Arabic character."""
        return self._arabic_to_phoneme.get(arabic)
    
    def get_all_phonemes(self) -> List[Phoneme]:
        """Get all phonemes."""
        return self._phonemes.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        consonants = [p for p in self._phonemes if p.type == PhonemeType.CONSONANT]
        vowels = [p for p in self._phonemes if p.type == PhonemeType.VOWEL]
        
        return {
            "total_phonemes": len(self._phonemes),
            "consonants": len(consonants),
            "vowels": len(vowels),
            "coverage": "Arabic phoneme inventory"
        }

# Global database instance
_phoneme_db = PhonemeDatabase()

def get_phoneme_database() -> PhonemeDatabase:
    """Get the global phoneme database instance."""
    return _phoneme_db

def get_phoneme(symbol: str) -> Optional[Phoneme]:
    """Get phoneme by symbol."""
    return _phoneme_db.get_phoneme(symbol)

def analyze_phonemes(phoneme_list: List[str]) -> Dict[str, Any]:
    """Analyze a list of phonemes."""
    results = []
    for symbol in phoneme_list:
        if phoneme := get_phoneme(symbol):
            results.append({
                "symbol": symbol,
                "ipa": phoneme.ipa,
                "arabic": phoneme.arabic,
                "type": phoneme.type.value,
                "frequency": phoneme.frequency
            })
        else:
            results.append({
                "symbol": symbol,
                "ipa": "unknown",
                "arabic": "unknown",
                "type": "unknown",
                "frequency": 0.0
            })
    
    return {
        "analysis": results,
        "total_phonemes": len(phoneme_list),
        "recognized": len([r for r in results if r["type"] != "unknown"])
    }

def to_ipa(phoneme_symbols: List[str]) -> str:
    """Convert phoneme symbols to IPA representation."""
    ipa_chars = []
    for symbol in phoneme_symbols:
        if phoneme := get_phoneme(symbol):
            ipa_chars.append(phoneme.ipa)
        else:
            ipa_chars.append(f"[{symbol}]")
    return "".join(ipa_chars)

def to_arabic(phoneme_symbols: List[str]) -> str:
    """Convert phoneme symbols to Arabic representation."""
    arabic_chars = []
    for symbol in phoneme_symbols:
        if phoneme := get_phoneme(symbol):
            arabic_chars.append(phoneme.arabic)
        else:
            arabic_chars.append(f"[{symbol}]")
    return "".join(arabic_chars)

def get_database_stats() -> Dict[str, Any]:
    """Get phoneme database statistics."""
    return _phoneme_db.get_stats()

def validate_environment() -> bool:
    """Validate phoneme database environment."""
    try:
        stats = get_database_stats()
        return stats["total_phonemes"] > 0
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False