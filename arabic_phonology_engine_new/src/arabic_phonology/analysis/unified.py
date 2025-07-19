"""
Unified Arabic Phonology Analyzer - Expert Implementation
Orchestrates all components: Engine, Text Analysis, Web Interface
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time
from pathlib import Path

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core engine components
try:
    from new_engine import (
        ArabicPhonologyEngine, 
        ArabicAnalyzer,
        PHONEME_INVENTORY,
        PHONEME_INDEX,
        TEMPLATES,
        TEMPLATE_INDEX,
        ARABIC_TO_PHONEME
    )
    CORE_ENGINE_AVAILABLE = True
    logger.info("✅ Core engine imported successfully")
except ImportError as e:
    logger.error(f"❌ Core engine import failed: {e}")
    CORE_ENGINE_AVAILABLE = False

# Import phonology modules with fallbacks
try:
    from phonology.analyzer import analyze_phonemes, analyze_text_advanced
    from phonology.syllabifier import syllabify
    from phonology.normalizer import normalize_text
    PHONOLOGY_MODULES_AVAILABLE = True
    logger.info("✅ Phonology modules imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Phonology modules import failed: {e}")
    PHONOLOGY_MODULES_AVAILABLE = False


class UnifiedArabicAnalyzer:
    """
    Expert-level unified analyzer combining all Arabic phonology capabilities.
    Provides zero-tolerance analysis with multiple fallback mechanisms.
    """
    
    def __init__(self, 
                 use_neural_engine: bool = True,
                 enable_text_analysis: bool = True,
                 enable_caching: bool = True):
        """
        Initialize the unified analyzer.
        
        Args:
            use_neural_engine: Whether to use neural network engine
            enable_text_analysis: Whether to enable text-level analysis
            enable_caching: Whether to cache results
        """
        self.use_neural_engine = use_neural_engine and CORE_ENGINE_AVAILABLE
        self.enable_text_analysis = enable_text_analysis and PHONOLOGY_MODULES_AVAILABLE
        self.enable_caching = enable_caching
        
        # Initialize components
        self.neural_engine = None
        self.analyzer = None
        self.cache = {} if enable_caching else None
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all available components."""
        
        # Initialize neural engine
        if self.use_neural_engine:
            try:
                self.neural_engine = ArabicPhonologyEngine()
                self.analyzer = ArabicAnalyzer()
                logger.info("✅ Neural engine initialized")
            except Exception as e:
                logger.error(f"❌ Neural engine initialization failed: {e}")
                self.use_neural_engine = False
        
        # Log component status
        logger.info(f"Unified Analyzer Status:")
        logger.info(f"  Neural Engine: {'✅ Active' if self.use_neural_engine else '❌ Disabled'}")
        logger.info(f"  Text Analysis: {'✅ Active' if self.enable_text_analysis else '❌ Disabled'}")
        logger.info(f"  Caching: {'✅ Active' if self.enable_caching else '❌ Disabled'}")

    def analyze_comprehensive(self, 
                            text: str,
                            root: Optional[Tuple[str, str, str]] = None,
                            template: Optional[str] = None,
                            include_neural: bool = True,
                            include_text_analysis: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using all available methods.
        
        Args:
            text: Input Arabic text
            root: Optional root for neural analysis
            template: Optional template for neural analysis  
            include_neural: Whether to include neural analysis
            include_text_analysis: Whether to include text analysis
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{text}_{root}_{template}_{include_neural}_{include_text_analysis}"
        if self.enable_caching and cache_key in self.cache:
            logger.debug(f"Cache hit for: {text[:20]}...")
            return self.cache[cache_key]
        
        results = {
            "input": {
                "text": text,
                "root": root,
                "template": template
            },
            "analysis": {},
            "metadata": {
                "processing_time": 0,
                "methods_used": [],
                "success": True
            }
        }
        
        try:
            # Neural engine analysis
            if include_neural and self.use_neural_engine and root and template:
                neural_results = self._perform_neural_analysis(text, root, template)
                results["analysis"]["neural"] = neural_results
                results["metadata"]["methods_used"].append("neural_engine")
                
            # Text-level phonological analysis
            if include_text_analysis and self.enable_text_analysis:
                text_results = self._perform_text_analysis(text)
                results["analysis"]["text"] = text_results
                results["metadata"]["methods_used"].append("text_analysis")
                
            # Basic syllabification if available
            if PHONOLOGY_MODULES_AVAILABLE:
                try:
                    syllables = syllabify(list(text))
                    results["analysis"]["syllables"] = syllables
                    results["metadata"]["methods_used"].append("syllabification")
                except Exception as e:
                    logger.warning(f"Syllabification failed: {e}")
            
            # Character-level analysis
            char_analysis = self._perform_character_analysis(text)
            results["analysis"]["characters"] = char_analysis
            results["metadata"]["methods_used"].append("character_analysis")
                
        except Exception as e:
            logger.error(f"Analysis failed for '{text}': {e}")
            results["metadata"]["success"] = False
            results["metadata"]["error"] = str(e)
        
        # Set processing time
        results["metadata"]["processing_time"] = round((time.time() - start_time) * 1000, 2)
        
        # Cache results
        if self.enable_caching:
            self.cache[cache_key] = results
            
        return results
    
    def _perform_neural_analysis(self, text: str, root: Tuple[str, str, str], template: str) -> Dict[str, Any]:
        """Perform neural network-based analysis."""
        try:
            # Convert text to phoneme sequence
            phoneme_seq = []
            for char in text:
                if char in ARABIC_TO_PHONEME:
                    phoneme_seq.append(ARABIC_TO_PHONEME[char])
                else:
                    phoneme_seq.append(char)
            
            # Run neural analysis
            neural_result = self.analyzer.analyze(root, template, phoneme_seq)
            
            return {
                "engine_result": neural_result,
                "phoneme_sequence": phoneme_seq,
                "root_embedding": root,
                "template_used": template,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Neural analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _perform_text_analysis(self, text: str) -> Dict[str, Any]:
        """Perform text-level phonological analysis."""
        try:
            # Normalize text
            normalized = normalize_text(text)
            
            # Perform advanced analysis
            analysis_result = analyze_text_advanced(normalized)
            
            return {
                "normalized_text": normalized,
                "phoneme_analysis": analysis_result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _perform_character_analysis(self, text: str) -> Dict[str, Any]:
        """Perform basic character-level analysis."""
        analysis = {
            "total_characters": len(text),
            "arabic_characters": 0,
            "phoneme_mappings": {},
            "unknown_characters": []
        }
        
        for char in text:
            if char in ARABIC_TO_PHONEME:
                analysis["arabic_characters"] += 1
                phoneme = ARABIC_TO_PHONEME[char]
                analysis["phoneme_mappings"][char] = phoneme
            elif char not in [' ', '
', '\t']:
                analysis["unknown_characters"].append(char)
        
        analysis["arabic_ratio"] = analysis["arabic_characters"] / len(text) if text else 0
        
        return analysis
    
    def get_status(self) -> Dict[str, Any]:
        """Get current analyzer status and capabilities."""
        return {
            "components": {
                "neural_engine": self.use_neural_engine,
                "text_analysis": self.enable_text_analysis,
                "caching": self.enable_caching
            },
            "available_methods": [
                "comprehensive_analysis",
                "neural_analysis" if self.use_neural_engine else None,
                "text_analysis" if self.enable_text_analysis else None,
                "character_analysis"
            ],
            "cache_size": len(self.cache) if self.cache else 0,
            "phoneme_inventory_size": len(PHONEME_INVENTORY) if CORE_ENGINE_AVAILABLE else 0,
            "template_count": len(TEMPLATES) if CORE_ENGINE_AVAILABLE else 0
        }
    
    def clear_cache(self):
        """Clear analysis cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Analysis cache cleared")


# Factory function for easy instantiation
def create_analyzer(mode: str = "full") -> UnifiedArabicAnalyzer:
    """
    Factory function to create analyzer with different configurations.
    
    Args:
        mode: Configuration mode ("full", "neural", "text", "basic")
        
    Returns:
        Configured UnifiedArabicAnalyzer instance
    """
    if mode == "full":
        return UnifiedArabicAnalyzer(
            use_neural_engine=True,
            enable_text_analysis=True,
            enable_caching=True
        )
    elif mode == "neural":
        return UnifiedArabicAnalyzer(
            use_neural_engine=True,
            enable_text_analysis=False,
            enable_caching=True
        )
    elif mode == "text":
        return UnifiedArabicAnalyzer(
            use_neural_engine=False,
            enable_text_analysis=True,
            enable_caching=True
        )
    elif mode == "basic":
        return UnifiedArabicAnalyzer(
            use_neural_engine=False,
            enable_text_analysis=False,
            enable_caching=False
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Unified Arabic Phonology Analyzer - Expert Implementation")
    print("=" * 60)
    
    # Create analyzer
    analyzer = create_analyzer("full")
    
    # Print status
    status = analyzer.get_status()
    print(f"📊 Analyzer Status: {status}")
    
    # Test analysis
    test_text = "كتاب"
    test_root = ("K", "T", "B")
    test_template = "CVC"
    
    print(f"
🔍 Testing with: '{test_text}'")
    
    try:
        result = analyzer.analyze_comprehensive(
            text=test_text,
            root=test_root,
            template=test_template
        )
        
        print(f"✅ Analysis completed in {result['metadata']['processing_time']}ms")
        print(f"📋 Methods used: {result['metadata']['methods_used']}")
        print(f"📊 Results: {result['analysis'].keys()}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
