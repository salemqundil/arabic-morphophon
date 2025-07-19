# Arabic Morphological Classifier
# Classifies Arabic characters morphologically

def classify_morphology(text):
    """
    Classify Arabic text morphologically
    
    Args:
        text (str): Input Arabic text
        
    Returns:
        dict: Morphological classification data
    """
    if not text:
        return {'classification': 'empty', 'features': []}
    
    # Basic morphological classification
    core_letters = 'اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوؤيئىء'
    functional_marks = 'ًٌٍَُِّْ'
    
    stats = {
        'core_letters': 0,
        'functional_marks': 0,
        'spaces': 0,
        'digits': 0,
        'other': 0,
        'total_chars': len(text)
    }
    
    features = []
    
    for char in text:
        if char in core_letters:
            stats['core_letters'] += 1
            features.append({'char': char, 'type': 'core', 'category': 'letter'})
        elif char in functional_marks:
            stats['functional_marks'] += 1
            features.append({'char': char, 'type': 'functional', 'category': 'diacritic'})
        elif char == ' ':
            stats['spaces'] += 1
            features.append({'char': char, 'type': 'boundary', 'category': 'space'})
        elif char.isdigit():
            stats['digits'] += 1
            features.append({'char': char, 'type': 'extra', 'category': 'digit'})
        else:
            stats['other'] += 1
            features.append({'char': char, 'type': 'unknown', 'category': 'other'})
    
    # Determine overall classification
    if stats['core_letters'] > 0:
        classification = 'arabic_text'
    elif stats['digits'] > 0:
        classification = 'mixed_content'
    else:
        classification = 'non_arabic'
    
    return {
        'classification': classification,
        'statistics': stats,
        'features': features,
        'complexity': calculate_complexity(stats)
    }

def calculate_complexity(stats):
    """
    Calculate text complexity based on character distribution
    
    Args:
        stats (dict): Character statistics
        
    Returns:
        str: Complexity level (simple, moderate, complex)
    """
    total = stats['total_chars']
    if total == 0:
        return 'simple'
    
    # Calculate ratios
    diacritic_ratio = stats['functional_marks'] / total
    variety_score = sum(1 for v in stats.values() if v > 0)
    
    if diacritic_ratio > 0.3 and variety_score >= 4:
        return 'complex'
    elif diacritic_ratio > 0.1 or variety_score >= 3:
        return 'moderate'
    else:
        return 'simple'
