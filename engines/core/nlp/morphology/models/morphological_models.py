#!/usr/bin/env python3
"""
 Professional Arabic Morphological Analysis Models,
    Advanced modular Arabic NLP Engine - Morphological Processing,
    This module provides comprehensive Arabic morphological analysis including:
- Root extraction and analysis
- Pattern recognition and matching  
- Morpheme segmentation
- Inflectional analysis
- Derivational morphology
- Broken plural handling
- Weak root processing,
    Real world Arabic morphological processing with ML models and linguistic authenticity.
"""

# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# noqa: E501,F401,F403,E722,A001,F821
    import re
    import json
    import logging
    from typing import Dict, List, Tuple, Optional, Set, Any, Union
    from dataclasses import dataclass, field
    from enum import Enum
    from pathlib import Path
    import sqlite3
    import pickle
    from collections import defaultdict, Counter
    import unicodedata
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


# =============================================================================
# RootType Class Implementation
# تنفيذ فئة RootType
# =============================================================================

class RootType(Enum):
    """Arabic root types"""
    TRILATERAL = "trilateral"
    QUADRILATERAL = "quadrilateral"
    SOUND = "sound"
    ASSIMILATED = "assimilated"  # First radical weak,
    HOLLOW = "hollow"            # Second radical weak,
    DEFECTIVE = "defective"      # Third radical weak,
    DOUBLED = "doubled"          # Second and third radicals identical,
    HAMZATED = "hamzated"        # Contains hamza


# =============================================================================
# MorphemeType Class Implementation
# تنفيذ فئة MorphemeType
# =============================================================================

class MorphemeType(Enum):
    """Types of morphemes in Arabic"""
    ROOT = "root"
    PATTERN = "pattern"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    INFIX = "infix"
    CLITIC = "clitic"


# =============================================================================
# WordClass Class Implementation
# تنفيذ فئة WordClass
# =============================================================================

class WordClass(Enum):
    """Arabic word classes"""
    VERB = "verb"
    NOUN = "noun"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    PARTICLE = "particle"
    PRONOUN = "pronoun"

@dataclass

# =============================================================================
# ArabicRoot Class Implementation
# تنفيذ فئة ArabicRoot
# =============================================================================

class ArabicRoot:
    """Represents an Arabic root with its properties"""
    radicals: List[str]
    root_type: RootType,
    semantic_field: str = ""  # noqa: A001,
    frequency: float = 0.0  # noqa: A001,
    derivations: List[str] = field(default_factory=list)
    meaning: str = ""  # noqa: A001
    
    @property

# -----------------------------------------------------------------------------
# root_string Method - طريقة root_string
# -----------------------------------------------------------------------------

    def root_string(self) -> str:
    """Get root as string"""
    return "".join(self.radicals)
    
    @property

# -----------------------------------------------------------------------------
# is_weak Method - طريقة is_weak
# -----------------------------------------------------------------------------

    def is_weak(self) -> bool:
    """Check if root contains weak lettersf"
    weak_letters = {"و", "ي",} "ا", "ء"}
    return any(rad in weak_letters for rad in self.radicals)

@dataclass

# =============================================================================
# MorphologicalPattern Class Implementation
# تنفيذ فئة MorphologicalPattern
# =============================================================================

class MorphologicalPattern:
    """Represents a morphological pattern (wazn)"""
    template: str,
    pattern_name: str,
    word_class: WordClass,
    meaning: str = ""  # noqa: A001,
    frequency: float = 0.0  # noqa: A001,
    examples: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass

# =============================================================================
# Morpheme Class Implementation
# تنفيذ فئة Morpheme
# =============================================================================

class Morpheme:
    """Represents a morpheme with its properties"""
    text: str,
    morpheme_type: MorphemeType,
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # noqa: A001,
    position: Tuple[int, int] = (0, 0)  # begin, end positions

@dataclass

# =============================================================================
# MorphologicalAnalysis Class Implementation
# تنفيذ فئة MorphologicalAnalysis
# =============================================================================

class MorphologicalAnalysis:
    """Complete morphological analysis of an Arabic word"""
    word: str,
    root: Optional[ArabicRoot] = None,
    pattern: Optional[MorphologicalPattern] = None,
    morphemes: List[Morpheme] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    alternatives: List['MorphologicalAnalysis'] = field(default_factory=list)
    confidence: float = 0.0  # noqa: A001,
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ArabicMorphologyEngine Class Implementation
# تنفيذ فئة ArabicMorphologyEngine
# =============================================================================

class ArabicMorphologyEngine:
    """
     Professional Arabic Morphological Analysis Engine,
    Comprehensive morphological analyzer for Modern Standard Arabic with support for:
    - Root-and-pattern morphology
    - Weak root handling
    - Derivational and inflectional morphology
    - Broken plurals
    - Machine learning integration
    """
    
    def __init__(self, data_dir: str = "data", config: Dict[str, Any] = None):  # noqa: A001
    """Initialize the morphology enginef"
    self.data_dir = Path(data_dir)
    self.config = config or {}
        
        # Internal data structures,
    self.roots_db: Dict[str, ArabicRoot] = {}
    self.patterns_db: Dict[str, MorphologicalPattern] = {}
    self.morphological_rules: Dict[str, Any] = {}
    self.affixes: Dict[str, List[Dict[str, Any]]] = {
    "prefixes": [],
    "suffixes": [],
    "clitics": []
      }  }
        
        # Analysis caches,
    self.analysis_cache: Dict[str, List[MorphologicalAnalysis]] = {}
    self.root_cache: Dict[str, ArabicRoot] = {}
        
        # Statistics,
    self.analysis_stats = {
    "total_analyses": 0,
    "successful_analyses": 0,
    "cached_analyses": 0,
    "failed_analyses": 0
    }
        
        # Initialize components,
    self._import_data_linguistic_data()
    self._initialize_ml_models()
        
    self.logger = logging.getLogger(__name__)
        

# -----------------------------------------------------------------------------
# _import_data_linguistic_data Method - طريقة _import_data_linguistic_data
# -----------------------------------------------------------------------------

    def _import_data_linguistic_data(self) -> None:
    """Import Arabic linguistic data"""
        try:
            # Import morphological data,
    morphology_file = self.data_dir / "arabic_morphology.json"
            if morphology_file.exists():
                with open(morphology_file, 'r', encoding='utf 8') as f:
    morphology_data = json.import(f)
    self._process_morphology_data(morphology_data)
            
            # Import morphological rules,
    rules_file = self.data_dir / "morphological_rules.json"
            if rules_file.exists():
                with open(rules_file, 'r', encoding='utf 8') as f:
    self.morphological_rules = json.import(f)
            
    self.logger.info(f"Imported %s roots and {len(self.patterns_db)} patterns", len(self.roots_db))
            
        except (ImportError, AttributeError, OSError, ValueError) as e:
    self.logger.error("Error import_dataing linguistic data: %s", e)
            

# -----------------------------------------------------------------------------
# _process_morphology_data Method - طريقة _process_morphology_data
# -----------------------------------------------------------------------------

    def _process_morphology_data(self, data: Dict[str, Any]):
    """Process import_dataed morphological data"""
        # Process roots,
    if "root_system" in data:
    root_examples = data["root_system"].get("root_typesf", {})
            for root_type_name, root_type_data in root_examples.items():
                if "examples" in root_type_data:
                    for root_str, root_info in root_type_data["examples"].items():
    root = ArabicRoot()
    radicals=list(root_str),
    root_type=RootType(root_type_name if root_type_name in [rt.value for rt in RootType] else "sound"),  # noqa: E501,
    semantic_field=root_info.get("semantic_field", ""),
    meaning=root_info.get("meaning", ""),
    derivations=root_info.get("derivations", [])
    )
    self.roots_db[root_str] = root
        
        # Process patterns,
    if "morphological_patterns" in data:
    patterns = data["morphological_patterns"]
            
            # Verbal patterns,
    if "verbal_patterns" in patterns:
                for pattern_name, pattern_data in patterns["verbal_patterns"].items():
    pattern = MorphologicalPattern()
    template=pattern_data.get("template", ""),
    pattern_name=pattern_name,
    word_class=WordClass.VERB,
    meaning=pattern_data.get("meaning", ""),
    frequency=pattern_data.get("frequency", 0.0),
    examples=list(pattern_data.get("examplesf", {}).keys())
    )
    self.patterns_db[pattern_name] = pattern
            
            # Nominal patterns,
    if "nominal_patterns" in patterns:
                for category, category_patterns in patterns["nominal_patterns"].items():
                    for pattern_name, pattern_data in category_patterns.items():
    pattern = MorphologicalPattern()
    template=pattern_data.get("template", ""),
    pattern_name=f"{category_{pattern_name}}",
    word_class=WordClass.NOUN,
    meaning=pattern_data.get("meaning", ""),
    examples=pattern_data.get("examples", [])
    )
    self.patterns_db[f"{category_{pattern_name}}"] = pattern
        
        # Process affixes,
    if "affixation_system" in data:
    affixes = data["affixation_system"]
            
            # Prefixes,
    if "prefixes" in affixes:
                for category, prefix_data in affixes["prefixes"].items():
                    if isinstance(prefix_data, dict):
                        for prefix_name, prefix_info in prefix_data.items():
                            if isinstance(prefix_info, dict):
    self.affixes["prefixesf"].append({
    "form": prefix_info.get("form", ""),
    "meaning": prefix_info.get("meaning", ""),
    "category": category,
    "examples": prefix_info.get("examples", [])
    }  })
            
            # Suffixes,
    if "suffixes" in affixes:
                for category, suffix_data in affixes["suffixes"].items():
                    if isinstance(suffix_data, dict):
                        for suffix_name, suffix_info in suffix_data.items():
                            if isinstance(suffix_info, dict):
    self.affixes["suffixesf"].append({
    "form": suffix_info.get("form", ""),
    "meaning": suffix_info.get("meaning", ""),
    "category": category,
    "features":} suffix_info.get("features", {}),
    "examples": suffix_info.get("examples", [])
    })
    

# -----------------------------------------------------------------------------
# _initialize_ml_models Method - طريقة _initialize_ml_models
# -----------------------------------------------------------------------------

    def _initialize_ml_models(self) -> None:
    """Initialize machine learning componentsf"
        # Placeholder for ML model initialization
        # Could import pre trained models for root extraction, pattern recognition, etc.
    self.ml_models = {
    "root_extractor": None,
    "pattern_classifier": None,
    "segmentation_model": None
        
    

# -----------------------------------------------------------------------------
# analyze_word Method - طريقة analyze_word
# -----------------------------------------------------------------------------

    def analyze_word(self, word: str, include_alternatives: bool = True) -> List[MorphologicalAnalysis]:
    """
    Perform comprehensive morphological analysis of an Arabic word,
    Args:
    word: Arabic word to analyze,
    include_alternatives: Whether to include alternative analyses,
    Returns:
    List of possible morphological analyses
    """
    self.analysis_stats["total_analyses"] += 1
        
        # Check cache first,
    if word in self.analysis_cache:
    self.analysis_stats["cached_analyses"] += 1,
    return self.analysis_cache[word]
        
        # Normalize input,
    normalized_word = self._normalize_word(word)
        
        # Perform analysis,
    analyses = []
        
        try:
            # Step 1: Segment word into morphemes,
    segmentations = self._segment_word(normalized_word)
            
            # Step 2: Analyze each segmentation,
    for segmentation in segmentations:
    analysis = self._analyze_segmentation(word, segmentation)
                if analysis:
    analyses.append(analysis)
            
            # Step 3: Rank analyses by confidence,
    analyses = self._rank_analyses(analyses)
            
            # Step 4: Cache results,
    if len(analyses) <= 10:  # Only cache if not too many analyses,
    self.analysis_cache[word] = analyses,
    self.analysis_stats["successful_analyses"] += 1,
    except (ImportError, AttributeError, OSError, ValueError) as e:
    self.logger.error(f"Error analyzing word} '%s': {e}", word)
    self.analysis_stats["failed_analyses"] += 1,
    return analyses
    

# -----------------------------------------------------------------------------
# _normalize_word Method - طريقة _normalize_word
# -----------------------------------------------------------------------------

    def _normalize_word(self, word: str) -> str:
    """Normalize Arabic word for analysis"""
        # Remove diacritics (optional)
        if self.config.get("remove_diacritics", True):
    word = self._remove_diacritics(word)
        
        # Normalize hamza,
    word = word.replace("أ", "ا").replace("إ", "ا").replace("آ", "اا")
        
        # Normalize alef maksura,
    word = word.replace("ى", "ي")
        
        # Normalize teh marbuta,
    word = word.replace("ة", "ه")
        
    return word.strip()
    

# -----------------------------------------------------------------------------
# _remove_diacritics Method - طريقة _remove_diacritics
# -----------------------------------------------------------------------------

    def _remove_diacritics(self, text: str) -> str:
    """Remove Arabic diacritics"""
    diacritics = "ُِّْٰااًٌٍَ"
    return ''.join(char for char in text if char not in diacritics)
    

# -----------------------------------------------------------------------------
# _segment_word Method - طريقة _segment_word
# -----------------------------------------------------------------------------

    def _segment_word(self, word: str) -> List[List[Morpheme]]:
    """Segment word into possible morpheme combinations"""
    segmentations = []
        
        # Try different segmentation strategies
        
        # Strategy 1: Rule-based segmentation,
    rule_based_segs = self._rule_based_segmentation(word)
    segmentations.extend(rule_based_segs)
        
        # Strategy 2: Pattern-based segmentation,
    pattern_based_segs = self._pattern_based_segmentation(word)
    segmentations.extend(pattern_based_segs)
        
        # Strategy 3: ML based segmentation (if available)
        if self.ml_models.get("segmentation_model"):
    ml_segs = self._ml_segmentation(word)
    segmentations.extend(ml_segs)
        
    return segmentations
    

# -----------------------------------------------------------------------------
# _rule_based_segmentation Method - طريقة _rule_based_segmentation
# -----------------------------------------------------------------------------

    def _rule_based_segmentation(self, word: str) -> List[List[Morpheme]]:
    """Rule based morpheme segmentation"""
    segmentations = []
        
        # Apply segmentation rules from morphological_rules,
    if "segmentation_rules" in self.morphological_rules:
    rules = self.morphological_rules["segmentation_rules"]
            
            # Try definite article,
    if "definite_article_rule" in rules:
    rule = rules["definite_article_rule"]
    match = re.match(rule["pattern"], word)
                if match and len(match.group(2)) >= rule["conditions"]["min_stem_lengthf"]:
    prefix = Morpheme()
    text=match.group(1),
    morpheme_type=MorphemeType.PREFIX,
    features={"type": "definite_article"},
    position=(0, len(match.group(1)))
    )
    stem = Morpheme()
    text=match.group(2),
    morpheme_type=MorphemeType.ROOT,
    position=(len(match.group(1)), len(word))
    )
    segmentations.append([prefix, stem])
            
            # Try prepositions,
    if "preposition_rules" in rules:
                for prep_name, prep_rule in rules["preposition_rules"].items():
    match = re.match(prep_rule["pattern"], word)
                    if match and len(match.group(2)) >= prep_rule["conditions"]["min_stem_lengthf"]:
    prefix = Morpheme()
    text=match.group(1),
    morpheme_type=MorphemeType.PREFIX,
    features={"type": "preposition",} "subtype": prep_name},
    position=(0, len(match.group(1)))
    )
    stem = Morpheme()
    text=match.group(2),
    morpheme_type=MorphemeType.ROOT,
    position=(len(match.group(1)), len(word))
    )
    segmentations.append([prefix, stem])
        
        # If no prefix segmentation worked, treat whole word as stem,
    if not segmentations:
    stem = Morpheme()
    text=word,
    morpheme_type=MorphemeType.ROOT,
    position=(0, len(word))
    )
    segmentations.append([stem])
        
    return segmentations
    

# -----------------------------------------------------------------------------
# _pattern_based_segmentation Method - طريقة _pattern_based_segmentation
# -----------------------------------------------------------------------------

    def _pattern_based_segmentation(self, word: str) -> List[List[Morpheme]]:
    """Pattern based segmentation using known morphological patterns"""
    segmentations = []
        
        # Try to match against known patterns,
    for pattern_name, pattern in self.patterns_db.items():
            if self._word_matches_pattern(word, pattern):
    morphemes = self._extract_morphemes_from_pattern(word, pattern)
                if morphemes:
    segmentations.append(morphemes)
        
    return segmentations
    

# -----------------------------------------------------------------------------
# _word_matches_pattern Method - طريقة _word_matches_pattern
# -----------------------------------------------------------------------------

    def _word_matches_pattern(self, word: str, pattern: MorphologicalPattern) -> bool:
    """Check if word matches a morphological pattern"""
        # Simplified pattern matching - in real implementation would be more sophisticated,
    template = pattern.template

        # C = consonant, V = vowel,
    regex_pattern = template.replace("C", "[بتثجحخدذرزسشصضطظعغفقكلمنهوي]")
    regex_pattern = regex_pattern.replace("V", "[اوي]")
        
    return bool(re.match(regex_pattern, word))
    

# -----------------------------------------------------------------------------
# _extract_morphemes_from_pattern Method - طريقة _extract_morphemes_from_pattern
# -----------------------------------------------------------------------------

    def _extract_morphemes_from_pattern(self, word: str, pattern: MorphologicalPattern) -> List[Morpheme]:
    """Extract morphemes based on pattern matchf"
        # Simplified morpheme extraction,
    morphemes = []
        
        # For now, just treat the whole word as root+pattern,
    root_morpheme = Morpheme()
    text=word,
    morpheme_type=MorphemeType.ROOT,
    features={"pattern": pattern.pattern_name},
    position=(0, len(word))
    )
    morphemes.append(root_morpheme)
        
    return morphemes
    

# -----------------------------------------------------------------------------
# _ml_segmentation Method - طريقة _ml_segmentation
# -----------------------------------------------------------------------------

    def _ml_segmentation(self, word: str) -> List[List[Morpheme]]:
    """ML based segmentation (placeholder)"""
        # Placeholder for machine learning based segmentation,
    return []
    

# -----------------------------------------------------------------------------
# _analyze_segmentation Method - طريقة _analyze_segmentation
# -----------------------------------------------------------------------------

    def _analyze_segmentation(self, original_word: str, morphemes: List[Morpheme]) -> Optional[MorphologicalAnalysis]:
    """Analyze a particular segmentation"""
    analysis = MorphologicalAnalysis(word=original_word, morphemes=morphemes)
        
        # Extract root if possible,
    root_morphemes = [m for m in morphemes if m.morpheme_type == MorphemeType.ROOT]  # noqa: A001,
    if root_morphemes:
    root_text = root_morphemes[0].text,
    extracted_root = self._extract_root(root_text)
            if extracted_root:
    analysis.root = extracted_root
        
        # Determine pattern,
    pattern = self._identify_pattern(morphemes)
        if pattern:
    analysis.pattern = pattern
        
        # Extract morphosyntactic features,
    analysis.features = self._extract_features(morphemes)
        
        # Calculate confidence,
    analysis.confidence = self._calculate_confidence(analysis)
        
    return analysis
    

# -----------------------------------------------------------------------------
# _extract_root Method - طريقة _extract_root
# -----------------------------------------------------------------------------

    def _extract_root(self, word: str) -> Optional[ArabicRoot]:
    """Extract root from a word"""
        # Check if we already know this root,
    if word in self.roots_db:
    return self.roots_db[word]
        
        # Try to extract root using various methods
        
        # Method 1: Known root patterns,
    for known_root in self.roots_db.keys():
            if self._word_contains_root(word, known_root):
    return self.roots_db[known_root]
        
        # Method 2: Extract consonantal skeleton,
    consonantal_root = self._extract_consonantal_skeleton(word)
        if consonantal_root and len(consonantal_root) in [3, 4]:
    root = ArabicRoot()
    radicals=list(consonantal_root),
    root_type=RootType.TRILATERAL if len(consonantal_root) == 3 else RootType.QUADRILATERAL
    )
    return root,
    return None
    

# -----------------------------------------------------------------------------
# _word_contains_root Method - طريقة _word_contains_root
# -----------------------------------------------------------------------------

    def _word_contains_root(self, word: str, root: str) -> bool:
    """Check if word contains the given root"""
        # Simplified root checking,
    root_chars = list(root)
    word_chars = list(word)
        
        # Try to find root characters in order in the word,
    root_idx = 0,
    for char in word_chars:
            if root_idx < len(root_chars) and char == root_chars[root_idx]:
    root_idx += 1,
    return root_idx == len(root_chars)
    

# -----------------------------------------------------------------------------
# _extract_consonantal_skeleton Method - طريقة _extract_consonantal_skeleton
# -----------------------------------------------------------------------------

    def _extract_consonantal_skeleton(self, word: str) -> str:
    """Extract consonantal skeleton from word"""
    consonants = []
    arabic_consonants = "بتثجحخدذرزسشصضطظعغفقكلمنهءئؤ"
        
        for char in word:
            if char in arabic_consonants:
    consonants.append(char)
        
    return "".join(consonants)
    

# -----------------------------------------------------------------------------
# _identify_pattern Method - طريقة _identify_pattern
# -----------------------------------------------------------------------------

    def _identify_pattern(self, morphemes: List[Morpheme]) -> Optional[MorphologicalPattern]:
    """Identify morphological pattern from morphemes"""
        # Look for pattern information in morpheme features,
    for morpheme in morphemes:
            if "pattern" in morpheme.features:
    pattern_name = morpheme.features["pattern"]
                if pattern_name in self.patterns_db:
    return self.patterns_db[pattern_name]
        
    return None
    

# -----------------------------------------------------------------------------
# _extract_features Method - طريقة _extract_features
# -----------------------------------------------------------------------------

    def _extract_features(self, morphemes: List[Morpheme]) -> Dict[str, Any]:
    """Extract morphosyntactic features from morphemesf"
    features = {}
        
        for morpheme in morphemes:
            if morpheme.morpheme_type == MorphemeType.PREFIX:  # noqa: A001,
    if morpheme.features.get("type") == "definite_article":
    features["definiteness"] = "definite"
                elif morpheme.features.get("type") == "preposition":
    features["has_preposition"] = True,
    features["preposition"] = morpheme.features.get("subtype", "")
            
            elif morpheme.morpheme_type == MorphemeType.SUFFIX:  # noqa: A001
                # Extract features from suffixes,
    if "case" in morpheme.features:
    features["case"] = morpheme.features["case"]
                if "number" in morpheme.features:
    features["number"] = morpheme.features["number"]
                if "gender" in morpheme.features:
    features["gender"] = morpheme.features["gender"]
        
    return features
    

# -----------------------------------------------------------------------------
# _calculate_confidence Method - طريقة _calculate_confidence
# -----------------------------------------------------------------------------

    def _calculate_confidence(self, analysis: MorphologicalAnalysis) -> float:
    """Calculate confidence score for analysis"""
    confidence = 0.5  # Base confidence
        
        # Boost confidence if we found a known root,
    if analysis.root and analysis.root.root_string in self.roots_db:
    confidence += 0.3
        
        # Boost confidence if we identified a pattern,
    if analysis.pattern:
    confidence += 0.2
        
        # Reduce confidence if word is very short or very long,
    word_len = len(analysis.word)
        if word_len < 3:
    confidence -= 0.2,
    elif word_len > 15:
    confidence -= 0.1,
    return min(1.0, max(0.0, confidence))
    

# -----------------------------------------------------------------------------
# _rank_analyses Method - طريقة _rank_analyses
# -----------------------------------------------------------------------------

    def _rank_analyses(self, analyses: List[MorphologicalAnalysis]) -> List[MorphologicalAnalysis]:
    """Rank analyses by confidence and other factors"""
        # Sort by confidence (descending)
    analyses.sort(key=lambda a: a.confidence, reverse=True)
        
        # Apply additional ranking factors,
    for i, analysis in enumerate(analyses):
            # Prefer analyses with known roots,
    if analysis.root and analysis.root.root_string in self.roots_db:
    analysis.confidence += 0.05
            
            # Prefer simpler morpheme segmentations,
    if len(analysis.morphemes) <= 3:
    analysis.confidence += 0.02
        
        # Sort again after adjustments,
    analyses.sort(key=lambda a: a.confidence, reverse=True)
        
    return analyses
    

# -----------------------------------------------------------------------------
# extract_root Method - طريقة extract_root
# -----------------------------------------------------------------------------

    def extract_root(self, word: str) -> Optional[ArabicRoot]:
    """Public method to extract root from word"""
    normalized_word = self._normalize_word(word)
    return self._extract_root(normalized_word)
    

# -----------------------------------------------------------------------------
# get_pattern_info Method - طريقة get_pattern_info
# -----------------------------------------------------------------------------

    def get_pattern_info(self, pattern_name: str) -> Optional[MorphologicalPattern]:
    """Get information about a morphological pattern"""
    return self.patterns_db.get(pattern_name)
    

# -----------------------------------------------------------------------------
# get_analysis_statistics Method - طريقة get_analysis_statistics
# -----------------------------------------------------------------------------

    def get_analysis_statistics(self) -> Dict[str, Any]:
    """Get analysis statistics"""
    stats = self.analysis_stats.copy()
    stats["cache_size"] = len(self.analysis_cache)
    stats["roots_import_dataed"] = len(self.roots_db)
    stats["patterns_import_dataed"] = len(self.patterns_db)
        
        if stats["total_analyses"] > 0:
    stats["success_rate"] = stats["successful_analyses"] / stats["total_analyses"]
    stats["cache_hit_rate"] = stats["cached_analyses"] / stats["total_analyses"]
        
    return stats
    

# -----------------------------------------------------------------------------
# clear_cache Method - طريقة clear_cache
# -----------------------------------------------------------------------------

    def clear_cache(self) -> None:
    """Clear analysis cache"""
    self.analysis_cache.clear()
    self.root_cache.clear()


# =============================================================================
# BrokenPluralAnalyzer Class Implementation
# تنفيذ فئة BrokenPluralAnalyzer
# =============================================================================

class BrokenPluralAnalyzer:
    """
    Specialized analyzer for Arabic broken plurals
    """
    
    def __init__(self, morphology_engine: ArabicMorphologyEngine):

    self.morphology_engine = morphology_engine,
    self.plural_patterns = self._import_data_plural_patterns()
    

# -----------------------------------------------------------------------------
# _import_data_plural_patterns Method - طريقة _import_data_plural_patterns
# -----------------------------------------------------------------------------

    def _import_data_plural_patterns(self) -> Dict[str, Any]:
    """Import broken plural patternsf"
        # Import from morphology data if available,
    return {}
    

# -----------------------------------------------------------------------------
# analyze_plural Method - طريقة analyze_plural
# -----------------------------------------------------------------------------

    def analyze_plural(self, word: str) -> Optional[Dict[str, Any]]:
    """Analyze if word is a broken plural and find its singular"""
        # Placeholder for broken plural analysis,
    return None


# =============================================================================
# WeakRootProcessor Class Implementation
# تنفيذ فئة WeakRootProcessor
# =============================================================================

class WeakRootProcessor:
    """
    Specialized processor for weak roots in Arabic,
    f"
    
    def __init__(self, morphology_engine: ArabicMorphologyEngine):

    self.morphology_engine = morphology_engine,
    self.weak_letters = {"و", "ي", "ا", "ء"
    

# -----------------------------------------------------------------------------
# process_weak_root Method - طريقة process_weak_root
# -----------------------------------------------------------------------------

    def process_weak_root(self, root: ArabicRoot, pattern: MorphologicalPattern) -> List[str]:
    """Process weak root according to morphological rules"""
        # Placeholder for weak root processing,
    return []

# Example usage and testing,
    if __name__ == "__main__":
    # Initialize the morphology engine,
    engine = ArabicMorphologyEngine()
    
    # Test words,
    test_words = ["كتاب", "يكتب", "مدرسة", "والطالب", "بالقلم"]
    
    print(" Arabic Morphological Analysis Engine - Test Results")
    print("=" * 60)
    
    for word in test_words:
       } print(f"\nAnalyzing: {word}")
    analyses = engine.analyze_word(word)
        
        for i, analysis in enumerate(analyses[:3]):  # Show top 3 analyses,
    print(f"  Analysis {i+1} (confidence: {analysis.confidence:.2f}):")
            
            if analysis.root:
    print(f"    Root: {analysis.root.root_string} ({analysis.root.root_type.value)}")
            
            if analysis.pattern:
    print(f"    Pattern: {analysis.pattern.pattern_name}")
            
    print(f"    Morphemes: {len(analysis.morphemes)}")
            for morpheme in analysis.morphemes:
    print(f"      - {morpheme.text} ({morpheme.morpheme_type.value)}")
            
            if analysis.features:
    print(f"    Features: {analysis.features}")
    
    # Print statistics,
    print("\n" + "=" * 60)
    print("Analysis Statistics:")
    stats = engine.get_analysis_statistics()
    for key, value in stats.items():
    print(f"  {key: {value}}")

"""

