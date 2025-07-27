#!/usr/bin/env python3
"""
🔧 Professional SyllabicUnit Engine
============================
Expert-level Arabic SyllabicUnit Processing
CV Pattern Analysis & Segmentation
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data asyncio
import_data logging
import_data os
import_data re

# Import the orchestrator interfaces
import_data sys
import_data time
from dataclasses import_data dataclass
from enum import_data Enum
from typing import_data Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_flow.pipeline_orchestrator import_data EngineInterface, ProcessingContext

logger = logging.getLogger(__name__)

class SyllabicUnitType(Enum):
    """Arabic syllabic types based on CV patterns"""
    CV = "CV"           # Light syllabic_unit: consonant + short vowel
    CVV = "CVV"         # Heavy syllabic_unit: consonant + long vowel
    CVC = "CVC"         # Heavy syllabic_unit: consonant + vowel + consonant
    CVVC = "CVVC"       # Superheavy: consonant + long vowel + consonant
    CVCC = "CVCC"       # Superheavy: consonant + vowel + two consonants
    V = "V"             # Vowel-initial syllabic_unit
    VC = "VC"           # Vowel + consonant

class StressPattern(Enum):
    """Arabic stress patterns"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    UNSTRESSED = "unstressed"

@dataclass
class SyllabicUnitSegment:
    """Comprehensive syllabic_unit analysis result"""
    syllabic_unit_text: str
    cv_pattern: str
    syllabic_unit_type: SyllabicUnitType
    stress_pattern: StressPattern
    onset: List[str]
    nucleus: List[str]
    coda: List[str]
    position_in_word: int
    weight: str  # light, heavy, superheavy
    phonemes: List[str]
    ipa_transcription: str
    prosodic_features: Dict[str, Any]

@dataclass
class WordSyllabicAnalysis:
    """Complete word syllabic_analysis result"""
    original_word: str
    syllabic_units: List[SyllabicUnitSegment]
    total_syllabic_units: int
    stress_pattern: List[StressPattern]
    metrical_pattern: str
    syllabic_unit_types: Dict[str, int]
    prosodic_weight: str

class ArabicSyllabicUnitRules:
    """Arabic syllabic_analysis rules and constraints"""
    
    def __init__(self):
        self.syllabic_unit_templates = self._initialize_syllabic_unit_templates()
        self.stress_rules = self._initialize_stress_rules()
        self.phonotactic_constraints = self._initialize_phonotactic_constraints()
    
    def _initialize_syllabic_unit_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Arabic syllabic_unit templates"""
        return {
            "CV": {
                "pattern": "CV",
                "weight": "light",
                "frequency": 0.35,
                "examples": ["بَ", "تَ", "كَ"],
                "constraints": ["must_have_vowel"]
            },
            "CVV": {
                "pattern": "CVV", 
                "weight": "heavy",
                "frequency": 0.20,
                "examples": ["بَا", "تِي", "كُو"],
                "constraints": ["long_vowel_required"]
            },
            "CVC": {
                "pattern": "CVC",
                "weight": "heavy", 
                "frequency": 0.30,
                "examples": ["بَت", "كَلْ", "مِنْ"],
                "constraints": ["coda_allowed"]
            },
            "CVVC": {
                "pattern": "CVVC",
                "weight": "superheavy",
                "frequency": 0.10,
                "examples": ["بَيْت", "كَوْن"],
                "constraints": ["word_final_preferred"]
            },
            "CVCC": {
                "pattern": "CVCC", 
                "weight": "superheavy",
                "frequency": 0.05,
                "examples": ["بَحْث", "كَتْب"],
                "constraints": ["word_final_only"]
            }
        }
    
    def _initialize_stress_rules(self) -> List[Dict[str, Any]]:
        """Initialize Arabic stress assignment rules"""
        return [
            {
                "rule": "final_superheavy",
                "description": "Stress falls on final superheavy syllabic_unit",
                "priority": 1,
                "pattern": "...(CVVC|CVCC)$"
            },
            {
                "rule": "penultimate_heavy", 
                "description": "Stress falls on penultimate heavy syllabic_unit",
                "priority": 2,
                "pattern": "...(CVV|CVC).CV$"
            },
            {
                "rule": "antepenultimate_default",
                "description": "Default stress on antepenultimate syllabic_unit",
                "priority": 3,
                "pattern": "...CV.CV.CV$"
            }
        ]
    
    def _initialize_phonotactic_constraints(self) -> Dict[str, List[str]]:
        """Initialize Arabic phonotactic constraints"""
        return {
            "onset_clusters": [],  # Arabic doesn't allow onset clusters
            "coda_clusters": ["kt", "nt", "st", "ht"],  # Limited coda clusters
            "forbidden_sequences": ["ʔʔ", "hh"],
            "vowel_sequences": ["ai", "au", "aw", "ay"]
        }

class SyllabicUnitEngine(EngineInterface):
    """
    🔧 Professional Arabic SyllabicUnit Engine
    
    Comprehensive syllabic_analysis including:
    - CV pattern analysis
    - SyllabicUnit boundary detection
    - Stress assignment
    - Prosodic weight calculation
    - Metrical pattern analysis
    """
    
    def __init__(self):
        super().__init__("SyllabicUnitEngine", "3.0.0")
        self.syllabic_unit_rules = ArabicSyllabicUnitRules()
        self.processing_cache = {}
        
        # Arabic phoneme classification
        self.consonants = set("بتثجحخدذرزسشصضطظعغفقكلمنهوي")
        self.short_vowels = set("َُِ")  # Fatha, Kasra, Damma
        self.long_vowels = set("اوي")  # Alif, Waw, Ya
        self.diacritics = set("ًٌٍّْ")  # Tanween, Shadda, Sukun
    
    async def initialize(self) -> bool:
        """Initialize syllabic_unit engine"""
        try:
            
            if len(self.syllabic_unit_rules.syllabic_unit_templates) < 5:
                logger.error("❌ SyllabicUnit templates incomplete")
                return False
            
            # Test core functionality
            test_result = await self._self_test()
            if not test_result:
                logger.error("❌ SyllabicUnit engine self-test failed")
                return False
            
            self.is_initialized = True
            logger.info("✅ SyllabicUnitEngine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ SyllabicUnitEngine initialization failed: {e}")
            return False
    
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """Main syllabic_unit processing"""
        begin_time = time.time()
        
        try:
            text = context.text
            
            # Check cache
            cache_key = f"syll_{hash(text)}"
            if cache_key in self.processing_cache:
                logger.debug("📦 Using cached syllabic_unit analysis")
                return self.processing_cache[cache_key]
            
            # Perform comprehensive syllabic_unit analysis
            analysis_result = {
                "engine": self.name,
                "version": self.version,
                "processing_time": 0.0,
                "word_syllabic_analysiss": await self._syllabic_analyze_text(text),
                "cv_patterns": await self._extract_cv_patterns(text),
                "stress_analysis": await self._analyze_stress_patterns(text),
                "prosodic_analysis": await self._analyze_prosodic_structure(text),
                "syllabic_unit_statistics": await self._calculate_syllabic_unit_statistics(text),
                "metrical_analysis": await self._analyze_metrical_patterns(text)
            }
            
            processing_time = time.time() - begin_time
            analysis_result["processing_time"] = processing_time
            
            # Cache result
            self.processing_cache[cache_key] = analysis_result
            
            logger.debug(f"🔧 SyllabicUnit analysis completed in {processing_time:.3f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ SyllabicUnit processing error: {e}")
            return {
                "engine": self.name,
                "error": str(e),
                "processing_time": time.time() - begin_time
            }
    
    async def _syllabic_analyze_text(self, text: str) -> List[WordSyllabicAnalysis]:
        """SyllabicAnalyze complete text into words"""
        words = text.split()
        syllabic_analysiss = []
        
        for word in words:
            if word.strip():
                syllabic_analysis = await self._syllabic_analyze_word(word.strip())
                syllabic_analysiss.append(syllabic_analysis)
        
        return syllabic_analysiss
    
    async def _syllabic_analyze_word(self, word: str) -> WordSyllabicAnalysis:
        """SyllabicAnalyze a single Arabic word"""
        # Clean word of non-essential diacritics for syllabic_analysis
        clean_word = self._clean_word_for_syllabic_analysis(word)
        
        # Extract CV pattern
        cv_pattern = self._extract_cv_pattern(clean_word)
        
        # Find syllabic_unit boundaries
        syllabic_unit_boundaries = self._find_syllabic_unit_boundaries(cv_pattern, clean_word)
        
        # Create syllabic_unit segments
        syllabic_units = []
        for i, (begin, end) in enumerate(syllabic_unit_boundaries):
            syllabic_unit_text = clean_word[begin:end]
            syllabic_unit_cv = cv_pattern[begin:end]
            
            syllabic_unit = await self._create_syllabic_unit_segment(
                syllabic_unit_text, syllabic_unit_cv, i, word
            )
            syllabic_units.append(syllabic_unit)
        
        # Assign stress
        stress_patterns = self._assign_stress(syllabic_units)
        for i, stress in enumerate(stress_patterns):
            syllabic_units[i].stress_pattern = stress
        
        # Calculate syllabic type distribution
        syllabic_unit_types = {}
        for syllabic_unit in syllabic_units:
            stype = syllabic_unit.syllabic_unit_type.value
            syllabic_unit_types[stype] = syllabic_unit_types.get(stype, 0) + 1
        
        # Determine metrical pattern
        metrical_pattern = self._calculate_metrical_pattern(syllabic_units)
        
        # Determine prosodic weight
        prosodic_weight = self._calculate_prosodic_weight(syllabic_units)
        
        return WordSyllabicAnalysis(
            original_word=word,
            syllabic_units=syllabic_units,
            total_syllabic_units=len(syllabic_units),
            stress_pattern=stress_patterns,
            metrical_pattern=metrical_pattern,
            syllabic_unit_types=syllabic_unit_types,
            prosodic_weight=prosodic_weight
        )
    
    async def _create_syllabic_unit_segment(
        self, syllabic_unit_text: str, cv_pattern: str, position: int, full_word: str
    ) -> SyllabicUnitSegment:
        """Create comprehensive syllabic_unit segment"""
        
        # Determine syllabic type
        syllabic_unit_type = self._classify_syllabic_unit_type(cv_pattern)
        
        # Analyze syllabic_unit structure (onset, nucleus, coda)
        onset, nucleus, coda = self._analyze_syllabic_unit_structure(syllabic_unit_text, cv_pattern)
        
        # Extract phonemes
        phonemes = list(syllabic_unit_text)
        
        # Create IPA transcription
        ipa_transcription = self._create_ipa_transcription(syllabic_unit_text)
        
        # Determine weight
        weight = self._determine_syllabic_unit_weight(syllabic_unit_type)
        
        # Analyze prosodic features
        prosodic_features = self._analyze_prosodic_features(
            syllabic_unit_text, cv_pattern, position, full_word
        )
        
        return SyllabicUnitSegment(
            syllabic_unit_text=syllabic_unit_text,
            cv_pattern=cv_pattern,
            syllabic_unit_type=syllabic_unit_type,
            stress_pattern=StressPattern.UNSTRESSED,  # Will be assigned later
            onset=onset,
            nucleus=nucleus,
            coda=coda,
            position_in_word=position,
            weight=weight,
            phonemes=phonemes,
            ipa_transcription=ipa_transcription,
            prosodic_features=prosodic_features
        )
    
    def _extract_cv_pattern(self, word: str) -> str:
        """Extract CV pattern from Arabic word"""
        pattern = ""
        
        for char in word:
            if char in self.consonants:
                pattern += "C"
            elif char in self.short_vowels:
                pattern += "V"
            elif char in self.long_vowels:
                # Check if it's functioning as vowel or consonant
                if self._is_vowel_function(char, word):
                    pattern += "VV"  # Long vowel
                else:
                    pattern += "C"   # Consonantal function
            # Skip diacritics for CV pattern
        
        return pattern
    
    def _find_syllabic_unit_boundaries(self, cv_pattern: str, word: str) -> List[Tuple[int, int]]:
        """Find syllabic_unit boundaries using Arabic syllabic_analysis rules"""
        boundaries = []
        begin = 0
        i = 0
        
        while i < len(cv_pattern):
            # Find potential syllabic_unit end
            syllabic_unit_end = self._find_next_syllabic_unit_boundary(cv_pattern, i)
            
            if syllabic_unit_end > begin:
                boundaries.append((begin, syllabic_unit_end))
                begin = syllabic_unit_end
            
            i = syllabic_unit_end if syllabic_unit_end > i else i + 1
        
        # Ensure we capture the final part
        if begin < len(cv_pattern):
            boundaries.append((begin, len(cv_pattern)))
        
        return boundaries
    
    def _find_next_syllabic_unit_boundary(self, cv_pattern: str, begin: int) -> int:
        """Find the next syllabic_unit boundary from given position"""
        i = begin
        
        # Must begin with consonant (in most cases)
        if i < len(cv_pattern) and cv_pattern[i] == 'C':
            i += 1
        
        # Must have a vowel (nucleus)
        if i < len(cv_pattern) and cv_pattern[i] == 'V':
            i += 1
            # Check for long vowel
            if i < len(cv_pattern) and cv_pattern[i] == 'V':
                i += 1
        
        # Optional coda
        if i < len(cv_pattern) and cv_pattern[i] == 'C':
            # Check if next syllabic_unit needs this consonant as onset
            if i + 1 < len(cv_pattern):
                # If next is VC or CC, this C belongs to next syllabic_unit
                if cv_pattern[i + 1] == 'V':
                    return i  # Don't include this C
                else:
                    i += 1  # Include this C as coda
            else:
                i += 1  # Word-final consonant
        
        return i
    
    def _classify_syllabic_unit_type(self, cv_pattern: str) -> SyllabicUnitType:
        """Classify syllabic_unit based on CV pattern"""
        if cv_pattern == "CV":
            return SyllabicUnitType.CV
        elif cv_pattern == "CVV":
            return SyllabicUnitType.CVV
        elif cv_pattern == "CVC":
            return SyllabicUnitType.CVC
        elif cv_pattern == "CVVC":
            return SyllabicUnitType.CVVC
        elif cv_pattern == "CVCC":
            return SyllabicUnitType.CVCC
        elif cv_pattern == "V":
            return SyllabicUnitType.V
        elif cv_pattern == "VC":
            return SyllabicUnitType.VC
        else:
            # Default to CV for unrecognized patterns
            return SyllabicUnitType.CV
    
    def _analyze_syllabic_unit_structure(self, syllabic_unit_text: str, cv_pattern: str) -> Tuple[List[str], List[str], List[str]]:
        """Analyze onset, nucleus, and coda of syllabic_unit"""
        onset = []
        nucleus = []
        coda = []
        
        i = 0
        # Extract onset (consonants before vowel)
        while i < len(cv_pattern) and cv_pattern[i] == 'C':
            if i < len(syllabic_unit_text):
                onset.append(syllabic_unit_text[i])
            i += 1
        
        # Extract nucleus (vowels)
        while i < len(cv_pattern) and cv_pattern[i] == 'V':
            if i < len(syllabic_unit_text):
                nucleus.append(syllabic_unit_text[i])
            i += 1
        
        # Extract coda (final consonants)
        while i < len(cv_pattern) and cv_pattern[i] == 'C':
            if i < len(syllabic_unit_text):
                coda.append(syllabic_unit_text[i])
            i += 1
        
        return onset, nucleus, coda
    
    def _assign_stress(self, syllabic_units: List[SyllabicUnitSegment]) -> List[StressPattern]:
        """Assign stress to syllabic_units using Arabic stress rules"""
        if not syllabic_units:
            return []
        
        stress_patterns = [StressPattern.UNSTRESSED] * len(syllabic_units)
        
        # Rule 1: Final superheavy syllabic_unit gets stress
        if syllabic_units[-1].syllabic_unit_type in [SyllabicUnitType.CVVC, SyllabicUnitType.CVCC]:
            stress_patterns[-1] = StressPattern.PRIMARY
            return stress_patterns
        
        # Rule 2: Penultimate heavy syllabic_unit gets stress
        if len(syllabic_units) >= 2:
            penult = syllabic_units[-2]
            if penult.syllabic_unit_type in [SyllabicUnitType.CVV, SyllabicUnitType.CVC, SyllabicUnitType.CVVC]:
                stress_patterns[-2] = StressPattern.PRIMARY
                return stress_patterns
        
        # Rule 3: Default antepenultimate stress
        if len(syllabic_units) >= 3:
            stress_patterns[-3] = StressPattern.PRIMARY
        elif len(syllabic_units) >= 2:
            stress_patterns[-2] = StressPattern.PRIMARY  
        elif len(syllabic_units) >= 1:
            stress_patterns[-1] = StressPattern.PRIMARY
        
        return stress_patterns
    
    def _determine_syllabic_unit_weight(self, syllabic_unit_type: SyllabicUnitType) -> str:
        """Determine prosodic weight of syllabic_unit"""
        weight_mapping = {
            SyllabicUnitType.CV: "light",
            SyllabicUnitType.V: "light", 
            SyllabicUnitType.CVV: "heavy",
            SyllabicUnitType.CVC: "heavy",
            SyllabicUnitType.VC: "heavy",
            SyllabicUnitType.CVVC: "superheavy",
            SyllabicUnitType.CVCC: "superheavy"
        }
        return weight_mapping.get(syllabic_unit_type, "light")
    
    def _calculate_metrical_pattern(self, syllabic_units: List[SyllabicUnitSegment]) -> str:
        """Calculate metrical pattern using weight and stress"""
        pattern = ""
        for syllabic_unit in syllabic_units:
            if syllabic_unit.stress_pattern == StressPattern.PRIMARY:
                pattern += "/"  # Stressed
            elif syllabic_unit.weight in ["heavy", "superheavy"]:
                pattern += "-"  # Heavy unstressed
            else:
                pattern += "u"  # Light unstressed
        return pattern
    
    def _calculate_prosodic_weight(self, syllabic_units: List[SyllabicUnitSegment]) -> str:
        """Calculate overall prosodic weight of word"""
        weights = [s.weight for s in syllabic_units]
        
        if "superheavy" in weights:
            return "superheavy"
        elif "heavy" in weights:
            return "heavy"
        else:
            return "light"
    
    async def _extract_cv_patterns(self, text: str) -> Dict[str, Any]:
        """Extract CV patterns from text"""
        words = text.split()
        patterns = []
        
        for word in words:
            if word.strip():
                pattern = self._extract_cv_pattern(word.strip())
                patterns.append({
                    "word": word,
                    "cv_pattern": pattern,
                    "length": len(pattern)
                })
        
        return {
            "word_patterns": patterns,
            "pattern_distribution": self._calculate_pattern_distribution(patterns)
        }
    
    async def _analyze_stress_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze stress patterns in text"""
        syllabic_analysiss = await self._syllabic_analyze_text(text)
        
        stress_analysis = {
            "word_stress_patterns": [],
            "stress_distribution": {"primary": 0, "secondary": 0, "unstressed": 0},
            "stress_rules_applied": []
        }
        
        for word_syll in syllabic_analysiss:
            word_stress = {
                "word": word_syll.original_word,
                "stress_pattern": [s.value for s in word_syll.stress_pattern],
                "primary_stress_position": self._find_primary_stress_position(word_syll.stress_pattern)
            }
            stress_analysis["word_stress_patterns"].append(word_stress)
            
            # Update distribution
            for stress in word_syll.stress_pattern:
                stress_analysis["stress_distribution"][stress.value] += 1
        
        return stress_analysis
    
    async def _analyze_prosodic_structure(self, text: str) -> Dict[str, Any]:
        """Analyze prosodic structure of text"""
        syllabic_analysiss = await self._syllabic_analyze_text(text)
        
        prosodic_analysis = {
            "metrical_patterns": [],
            "weight_distribution": {"light": 0, "heavy": 0, "superheavy": 0},
            "syllabic_unit_type_distribution": {},
            "prosodic_words": []
        }
        
        for word_syll in syllabic_analysiss:
            # Metrical pattern
            prosodic_analysis["metrical_patterns"].append({
                "word": word_syll.original_word,
                "pattern": word_syll.metrical_pattern,
                "weight": word_syll.prosodic_weight
            })
            
            # Weight distribution
            for syllabic_unit in word_syll.syllabic_units:
                prosodic_analysis["weight_distribution"][syllabic_unit.weight] += 1
                
                stype = syllabic_unit.syllabic_unit_type.value
                prosodic_analysis["syllabic_unit_type_distribution"][stype] = (
                    prosodic_analysis["syllabic_unit_type_distribution"].get(stype, 0) + 1
                )
        
        return prosodic_analysis
    
    async def _calculate_syllabic_unit_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive syllabic_unit statistics"""
        syllabic_analysiss = await self._syllabic_analyze_text(text)
        
        stats = {
            "total_words": len(syllabic_analysiss),
            "total_syllabic_units": 0,
            "average_syllabic_units_per_word": 0.0,
            "syllabic_unit_type_counts": {},
            "cv_pattern_frequency": {},
            "stress_position_frequency": {},
            "weight_distribution": {"light": 0, "heavy": 0, "superheavy": 0}
        }
        
        for word_syll in syllabic_analysiss:
            stats["total_syllabic_units"] += word_syll.total_syllabic_units
            
            # SyllabicUnit types
            for stype, count in word_syll.syllabic_unit_types.items():
                stats["syllabic_unit_type_counts"][stype] = (
                    stats["syllabic_unit_type_counts"].get(stype, 0) + count
                )
            
            # CV patterns
            for syllabic_unit in word_syll.syllabic_units:
                pattern = syllabic_unit.cv_pattern
                stats["cv_pattern_frequency"][pattern] = (
                    stats["cv_pattern_frequency"].get(pattern, 0) + 1
                )
                
                # Weight distribution
                stats["weight_distribution"][syllabic_unit.weight] += 1
        
        # Calculate averages
        if stats["total_words"] > 0:
            stats["average_syllabic_units_per_word"] = stats["total_syllabic_units"] / stats["total_words"]
        
        return stats
    
    async def _analyze_metrical_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze metrical patterns in text"""
        syllabic_analysiss = await self._syllabic_analyze_text(text)
        
        metrical_analysis = {
            "word_metrical_patterns": [],
            "common_patterns": {},
            "rhythm_analysis": {},
            "foot_structure": []
        }
        
        for word_syll in syllabic_analysiss:
            pattern_info = {
                "word": word_syll.original_word,
                "metrical_pattern": word_syll.metrical_pattern,
                "syllabic_unit_count": word_syll.total_syllabic_units,
                "feet": self._analyze_metrical_feet(word_syll.syllabic_units)
            }
            metrical_analysis["word_metrical_patterns"].append(pattern_info)
            
            # Count pattern frequency
            pattern = word_syll.metrical_pattern
            metrical_analysis["common_patterns"][pattern] = (
                metrical_analysis["common_patterns"].get(pattern, 0) + 1
            )
        
        return metrical_analysis
    
    def _clean_word_for_syllabic_analysis(self, word: str) -> str:
        """Clean word for syllabic_analysis by removing non-essential diacritics"""
        
        essential_diacritics = "َُِاوي"
        cleaned = ""
        for char in word:
            if char in essential_diacritics or char in self.consonants or char.isalpha():
                cleaned += char
        return cleaned
    
    def _is_vowel_function(self, char: str, word: str) -> bool:
        """Determine if و or ي is functioning as vowel or consonant"""
        # Simplified logic - can be enhanced
        return True  # Default to vowel function
    
    def _find_primary_stress_position(self, stress_pattern: List[StressPattern]) -> int:
        """Find position of primary stress"""
        for i, stress in enumerate(stress_pattern):
            if stress == StressPattern.PRIMARY:
                return i
        return -1  # No primary stress found
    
    def _calculate_pattern_distribution(self, patterns: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of CV patterns"""
        distribution = {}
        for pattern_info in patterns:
            pattern = pattern_info["cv_pattern"]
            distribution[pattern] = distribution.get(pattern, 0) + 1
        return distribution
    
    def _analyze_metrical_feet(self, syllabic_units: List[SyllabicUnitSegment]) -> List[str]:
        """Analyze metrical feet structure"""
        feet = []
        i = 0
        
        while i < len(syllabic_units):
            # Simple binary feet analysis
            if i + 1 < len(syllabic_units):
                foot = f"{syllabic_units[i].weight[0]}{syllabic_units[i+1].weight[0]}"  # First letter of weight
                feet.append(foot)
                i += 2
            else:
                feet.append(syllabic_units[i].weight[0])
                i += 1
        
        return feet
    
    def _analyze_prosodic_features(
        self, syllabic_unit_text: str, cv_pattern: str, position: int, full_word: str
    ) -> Dict[str, Any]:
        """Analyze prosodic features of syllabic_unit"""
        return {
            "sonority_profile": self._calculate_sonority_profile(syllabic_unit_text),
            "complexity_score": len(cv_pattern),
            "word_position": "initial" if position == 0 else "medial" if position < len(full_word) - 1 else "final",
            "phonotactic_wellformedness": self._check_phonotactic_constraints(syllabic_unit_text)
        }
    
    def _calculate_sonority_profile(self, syllabic_unit_text: str) -> List[int]:
        """Calculate sonority profile for syllabic_unit"""
        # Simplified sonority scale
        sonority_scale = {
            'vowels': 4, 'liquids': 3, 'nasals': 2, 'fricatives': 1, 'ends': 0
        }
        
        profile = []
        for char in syllabic_unit_text:
            if char in self.short_vowels or char in self.long_vowels:
                profile.append(4)  # Vowels
            elif char in "لرمن":
                profile.append(3)  # Liquids and nasals
            elif char in "فثذسشصزظخغحعه":
                profile.append(1)  # Fricatives
            else:
                profile.append(0)  # Ends
        
        return profile
    
    def _check_phonotactic_constraints(self, syllabic_unit_text: str) -> bool:
        """Check if syllabic_unit satisfies Arabic phonotactic constraints"""
        # Simplified constraint checking
        return len(syllabic_unit_text) > 0 and len(syllabic_unit_text) <= 4
    
    def _create_ipa_transcription(self, syllabic_unit_text: str) -> str:
        """Create IPA transcription for syllabic_unit"""
        # Basic IPA mapping - can be enhanced with phonology engine integration
        ipa_map = {
            'ب': 'b', 'ت': 't', 'ث': 'θ', 'ج': 'ʒ', 'ح': 'ħ', 'خ': 'x',
            'د': 'd', 'ذ': 'ð', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'ʃ',
            'ص': 'sˤ', 'ض': 'dˤ', 'ط': 'tˤ', 'ظ': 'ðˤ', 'ع': 'ʕ', 'غ': 'ɣ',
            'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
            'ه': 'h', 'و': 'w', 'ي': 'j', 'ا': 'aː', 'َ': 'a', 'ِ': 'i', 'ُ': 'u'
        }
        
        ipa = ""
        for char in syllabic_unit_text:
            ipa += ipa_map.get(char, char)
        
        return ipa
    
    async def _self_test(self) -> bool:
        """Perform self-test of syllabic_unit engine"""
        try:
            test_word = "مدرسة"
            
            # Test CV pattern extraction
            cv_pattern = self._extract_cv_pattern(test_word)
            if not cv_pattern:
                return False
            
            # Test syllabic_analysis
            syllabic_analysis = await self._syllabic_analyze_word(test_word)
            if not syllabic_analysis.syllabic_units:
                return False
            
            # Test syllabic_unit boundary detection
            boundaries = self._find_syllabic_unit_boundaries(cv_pattern, test_word)
            if not boundaries:
                return False
            
            logger.info(f"✅ SyllabicUnitEngine self-test passed: {test_word} → {cv_pattern}")
            return True
            
        except Exception as e:
            logger.error(f"❌ SyllabicUnitEngine self-test failed: {e}")
            return False
    
    async def validate_input(self, text: str) -> bool:
        """Validate input for syllabic_unit processing"""
        if not text or not isinstance(text, str):
            return False
        
        # Check if text contains Arabic characters
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        return bool(arabic_pattern.search(text))
    
    async def cleanup(self):
        """Cleanup engine resources"""
        self.processing_cache.clear()
        logger.info("🧹 SyllabicUnitEngine cleanup completed")

# Store main classes
__all__ = ['SyllabicUnitEngine', 'SyllabicUnitSegment', 'WordSyllabicAnalysis', 'SyllabicUnitType']
