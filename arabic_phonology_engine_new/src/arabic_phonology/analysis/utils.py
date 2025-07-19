"""
Phonological utilities for Arabic text analysis.
"""

from typing import Dict, Any

def get_phoneme_info(char: str, phoneme_db: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get phoneme information from database with safe fallback.
    
    Args:
        char: Character to look up
        phoneme_db: Phoneme database dictionary
        
    Returns:
        Dictionary with phoneme information
    """
    return phoneme_db.get(char, {
        "type": "unknown",
        "frequency": 0.0,
        "features": [],
        "morph_class": None
    }).copy()

def validate_phoneme_database(phoneme_db: Dict[str, Any]) -> bool:
    """
    Validate phoneme database structure.
    
    Args:
        phoneme_db: Database to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(phoneme_db, dict):
        return False
    
    required_fields = ["type", "frequency"]
    
    for char, info in phoneme_db.items():
        if not isinstance(info, dict):
            return False
        
        for field in required_fields:
            if field not in info:
                return False
    
    return True
