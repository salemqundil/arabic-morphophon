# Arabic Phoneme Analyzer
# Analyzes Arabic text for phonological properties

import re

# Arabic phoneme classification data
ARABIC_CONSONANTS = {
    'ب': {'type': 'consonant', 'place': 'bilabial', 'manner_primary': 'stop', 'voicing': 'voiced', 'emphatic': False},
    'ت': {'type': 'consonant', 'place': 'dental', 'manner_primary': 'stop', 'voicing': 'voiceless', 'emphatic': False},
    'ث': {'type': 'consonant', 'place': 'dental', 'manner_primary': 'fricative', 'voicing': 'voiceless', 'emphatic': False},
    'ج': {'type': 'consonant', 'place': 'post-alveolar', 'manner_primary': 'affricate', 'voicing': 'voiced', 'emphatic': False},
    'ح': {'type': 'consonant', 'place': 'pharyngeal', 'manner_primary': 'fricative', 'voicing': 'voiceless', 'emphatic': False},
    'خ': {'type': 'consonant', 'place': 'uvular', 'manner_primary': 'fricative', 'voicing': 'voiceless', 'emphatic': False},
    'د': {'type': 'consonant', 'place': 'dental', 'manner_primary': 'stop', 'voicing': 'voiced', 'emphatic': False},
    'ذ': {'type': 'consonant', 'place': 'dental', 'manner_primary': 'fricative', 'voicing': 'voiced', 'emphatic': False},
    'ر': {'type': 'consonant', 'place': 'alveolar', 'manner_primary': 'trill', 'voicing': 'voiced', 'emphatic': False},
    'ز': {'type': 'consonant', 'place': 'alveolar', 'manner_primary': 'fricative', 'voicing': 'voiced', 'emphatic': False},
    'س': {'type': 'consonant', 'place': 'alveolar', 'manner_primary': 'fricative', 'voicing': 'voiceless', 'emphatic': False},
    'ش': {'type': 'consonant', 'place': 'post-alveolar', 'manner_primary': 'fricative', 'voicing': 'voiceless', 'emphatic': False},
    'ص': {'type': 'consonant', 'place': 'alveolar', 'manner_primary': 'fricative', 'voicing': 'voiceless', 'emphatic': True},
    'ض': {'type': 'consonant', 'place': 'dental', 'manner_primary': 'stop', 'voicing': 'voiced', 'emphatic': True},
    'ط': {'type': 'consonant', 'place': 'dental', 'manner_primary': 'stop', 'voicing': 'voiceless', 'emphatic': True},
    'ظ': {'type': 'consonant', 'place': 'dental', 'manner_primary': 'fricative', 'voicing': 'voiced', 'emphatic': True},
    'ع': {'type': 'consonant', 'place': 'pharyngeal', 'manner_primary': 'fricative', 'voicing': 'voiced', 'emphatic': False},
    'غ': {'type': 'consonant', 'place': 'uvular', 'manner_primary': 'fricative', 'voicing': 'voiced', 'emphatic': False},
    'ف': {'type': 'consonant', 'place': 'labio-dental', 'manner_primary': 'fricative', 'voicing': 'voiceless', 'emphatic': False},
    'ق': {'type': 'consonant', 'place': 'uvular', 'manner_primary': 'stop', 'voicing': 'voiceless', 'emphatic': False},
    'ك': {'type': 'consonant', 'place': 'velar', 'manner_primary': 'stop', 'voicing': 'voiceless', 'emphatic': False},
    'ل': {'type': 'consonant', 'place': 'alveolar', 'manner_primary': 'lateral', 'voicing': 'voiced', 'emphatic': False},
    'م': {'type': 'consonant', 'place': 'bilabial', 'manner_primary': 'nasal', 'voicing': 'voiced', 'emphatic': False},
    'ن': {'type': 'consonant', 'place': 'alveolar', 'manner_primary': 'nasal', 'voicing': 'voiced', 'emphatic': False},
    'ه': {'type': 'consonant', 'place': 'glottal', 'manner_primary': 'fricative', 'voicing': 'voiceless', 'emphatic': False},
    'و': {'type': 'semi-vowel', 'place': 'labio-velar', 'manner_primary': 'approximant', 'voicing': 'voiced', 'emphatic': False},
    'ي': {'type': 'semi-vowel', 'place': 'palatal', 'manner_primary': 'approximant', 'voicing': 'voiced', 'emphatic': False},
    'ء': {'type': 'consonant', 'place': 'glottal', 'manner_primary': 'stop', 'voicing': 'voiceless', 'emphatic': False},
}

ARABIC_VOWELS = {
    'َ': {'type': 'short_vowel', 'name': 'fatha', 'quality': 'open'},
    'ِ': {'type': 'short_vowel', 'name': 'kasra', 'quality': 'close'},
    'ُ': {'type': 'short_vowel', 'name': 'damma', 'quality': 'close-back'},
    'ً': {'type': 'tanween', 'name': 'fathatan', 'quality': 'open'},
    'ٍ': {'type': 'tanween', 'name': 'kasratan', 'quality': 'close'},
    'ٌ': {'type': 'tanween', 'name': 'dammatan', 'quality': 'close-back'},
    'ْ': {'type': 'sukun', 'name': 'sukun', 'quality': 'none'},
    'ّ': {'type': 'gemination', 'name': 'shadda', 'quality': 'geminate'},
}

def analyze_phonemes(text):
    """
    Analyze Arabic text for phonological properties
    
    Args:
        text (str): Input Arabic text
        
    Returns:
        list: List of tuples containing (character, phonological_data)
    """
    if not text:
        return []
    
    analysis = []
    
    for char in text:
        if char in ARABIC_CONSONANTS:
            phoneme_data = ARABIC_CONSONANTS[char].copy()
            phoneme_data['morph_class'] = 'core'
            analysis.append((char, phoneme_data))
        elif char in ARABIC_VOWELS:
            phoneme_data = ARABIC_VOWELS[char].copy()
            phoneme_data['morph_class'] = 'functional'
            analysis.append((char, phoneme_data))
        elif char == ' ':
            phoneme_data = {'type': 'space', 'morph_class': 'boundary'}
            analysis.append((char, phoneme_data))
        elif char.isdigit():
            phoneme_data = {'type': 'digit', 'morph_class': 'extra'}
            analysis.append((char, phoneme_data))
        else:
            # Unknown character
            phoneme_data = {'type': 'unknown', 'morph_class': 'unknown'}
            analysis.append((char, phoneme_data))
    
    return analysis

def get_phoneme_features(char):
    """
    Get phonological features for a single character
    
    Args:
        char (str): Single Arabic character
        
    Returns:
        dict: Phonological features
    """
    if char in ARABIC_CONSONANTS:
        return ARABIC_CONSONANTS[char]
    elif char in ARABIC_VOWELS:
        return ARABIC_VOWELS[char]
    else:
        return {'type': 'unknown', 'morph_class': 'unknown'}

def is_emphatic(char):
    """
    Check if character is emphatic (pharyngealized)
    
    Args:
        char (str): Arabic character
        
    Returns:
        bool: True if emphatic, False otherwise
    """
    return ARABIC_CONSONANTS.get(char, {}).get('emphatic', False)

def get_consonant_type(char):
    """
    Get the type of consonant (stop, fricative, etc.)
    
    Args:
        char (str): Arabic character
        
    Returns:
        str: Consonant type or None if not a consonant
    """
    return ARABIC_CONSONANTS.get(char, {}).get('manner_primary', None)
