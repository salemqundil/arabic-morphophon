# Arabic Syllable Encoder
# Encodes Arabic text into syllable structures

import re
from .normalizer import normalize_text

def encode_syllables(text):
    """
    Encode Arabic text into syllable structures
    
    Args:
        text (str): Input Arabic text
        
    Returns:
        list: List of syllable encoding data
    """
    if not text:
        return []
    
    encoding = []
    
    for char in text:
        syllable_data = {
            'letter': char,
            'vowel': None,
            'syllable_code': 'unknown',
            'classification': 'unknown'
        }
        
        # Classify the character
        if char in 'اأإآ':
            syllable_data['syllable_code'] = 'V'  # Vowel
            syllable_data['classification'] = 'core'
            syllable_data['vowel'] = 'long_a'
        elif char in 'وؤ':
            syllable_data['syllable_code'] = 'V'  # Vowel
            syllable_data['classification'] = 'core'
            syllable_data['vowel'] = 'long_u'
        elif char in 'يئى':
            syllable_data['syllable_code'] = 'V'  # Vowel
            syllable_data['classification'] = 'core'
            syllable_data['vowel'] = 'long_i'
        elif char in 'بتثجحخدذرزسشصضطظعغفقكلمنهءة':
            syllable_data['syllable_code'] = 'C'  # Consonant
            syllable_data['classification'] = 'core'
        elif char in 'ًٌٍَُِّْ':
            syllable_data['syllable_code'] = 'D'  # Diacritic
            syllable_data['classification'] = 'functional'
            syllable_data['vowel'] = get_diacritic_name(char)
        elif char == ' ':
            syllable_data['syllable_code'] = 'S'  # Space
            syllable_data['classification'] = 'boundary'
        elif char.isdigit():
            syllable_data['syllable_code'] = 'N'  # Number
            syllable_data['classification'] = 'extra'
        else:
            syllable_data['syllable_code'] = 'X'  # Unknown
            syllable_data['classification'] = 'unknown'
        
        encoding.append(syllable_data)
    
    return encoding

def get_diacritic_name(char):
    """
    Get the name of Arabic diacritic
    
    Args:
        char (str): Diacritic character
        
    Returns:
        str: Name of the diacritic
    """
    diacritics = {
        'َ': 'fatha',
        'ِ': 'kasra', 
        'ُ': 'damma',
        'ً': 'fathatan',
        'ٍ': 'kasratan',
        'ٌ': 'dammatan',
        'ْ': 'sukun',
        'ّ': 'shadda'
    }
    return diacritics.get(char, 'unknown')

def get_syllable_pattern(text):
    """
    Get simplified syllable pattern (CV pattern)
    
    Args:
        text (str): Input Arabic text
        
    Returns:
        str: CV pattern string
    """
    encoding = encode_syllables(text)
    pattern = ""
    
    for item in encoding:
        code = item['syllable_code']
        if code in ['C', 'V']:
            pattern += code
        elif code == 'D':
            # Diacritics modify the previous consonant
            if pattern and pattern[-1] == 'C':
                pattern += 'V'  # Add vowel after consonant
    
    return pattern

def syllabify(text):
    """
    Divide text into syllables
    
    Args:
        text (str): Input Arabic text
        
    Returns:
        list: List of syllables
    """
    # Simple syllabification based on CV patterns
    pattern = get_syllable_pattern(text)
    syllables = []
    
    current_syllable = ""
    for i, char in enumerate(pattern):
        current_syllable += char
        
        # Basic syllable boundary detection
        if char == 'V':  # Vowel ends a syllable
            if i + 1 < len(pattern) and pattern[i + 1] == 'C':
                # CV.C or CVC pattern
                syllables.append(current_syllable)
                current_syllable = ""
    
    if current_syllable:
        syllables.append(current_syllable)
    
    return syllables
