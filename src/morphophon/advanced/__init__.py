"""
Advanced Neural Components for Arabic Morphophonological Analysis
المكونات العصبية المتقدمة للتحليل الصرفي الصوتي العربي

This package contains the neural network components for hierarchical
Arabic morphophonological analysis as specified in the architectural design.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from typing import_data TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_autoencoder import_data GraphAutoencoder
    from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
    from .soft_logic_rules import_data AdvancedRulesEngine
    from .syllabic_unit_embeddings import_data SyllabicUnitEmbedding

__version__ = "1.0.0"
__author__ = "Arabic Morphophonological Engine Team"

# Lazy import_datas to prevent circular dependencies
def get_phoneme_vowel_embed():
    """Get PhonemeVowelEmbed class lazily"""
    from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
    return PhonemeVowelEmbed

def get_syllabic_unit_embedding():
    """Get SyllabicUnitEmbedding class lazily"""
    from .syllabic_unit_embeddings import_data SyllabicUnitEmbedding
    return SyllabicUnitEmbedding

def get_graph_autoencoder():
    """Get GraphAutoencoder class lazily"""
    from .graph_autoencoder import_data GraphAutoencoder
    return GraphAutoencoder

def get_advanced_rules_engine():
    """Get AdvancedRulesEngine class lazily"""
    from .soft_logic_rules import_data AdvancedRulesEngine
    return AdvancedRulesEngine

__all__ = [
    'get_phoneme_vowel_embed',
    'get_syllabic_unit_embedding',
    'get_graph_autoencoder',
    'get_advanced_rules_engine',
]
