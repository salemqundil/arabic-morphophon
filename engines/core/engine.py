#!/usr/bin/env python3
"""
Unified Arabic NLP Engine
=========================

A unified interface to all Arabic NLP processing engines.
Provides a single entry point for morphological, phonological,
and syntactic analysis of Arabic text.

Author: Arabic NLP Team,
    Date: July 26, 2025,
    Version: 1.0.0
"""

import logging
    from typing import Dict, List, Any, Optional
    from pathlib import Path

# Global suppressions for clean development
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# noqa: E501,F401,F403,E722,A001,F821,
    class UnifiedArabicEngine:
    """
    Unified interface for Arabic NLP processing.

    Integrates all core engines:
    - Phonological processing
    - Morphological analysis
    - Inflection handling
    - Syllable processing
    - Weight analysis
    """

    def __init__(self, config: Optional[Dict] = None):
    """Initialize the unified engine with configuration"""
    self.config = config or {}
    self.logger = logging.getLogger(__name__)

        # Engine components,
    self.engines = {}
    self.is_initialized = False

        # Initialize if auto_init is enabled,
    if self.config.get('auto_init', True):
    self.initialize()

    def initialize(self) -> bool:
    """Initialize all core engine components"""
        try:
    self.logger.info("🚀 Initializing Unified Arabic Engine...")

            # Initialize core engines,
    self._init_phonological_engine()
    self._init_morphological_engine()
    self._init_inflection_engine()
    self._init_syllable_engine()
    self._init_weight_engine()

    self.is_initialized = True,
    self.logger.info("✅ Unified Arabic Engine initialized successfully")
    return True,
    except Exception as e:
    self.logger.error(f"❌ Failed to initialize engine: {e}")
    return False,
    def _init_phonological_engine(self):
    """Initialize phonological processing engine"""
        try:
            # Import core phonological engine
    from core.nlp.phonological.engine import PhonologicalEngine,
    self.engines['phonological'] = PhonologicalEngine()
    self.logger.debug("✅ Phonological engine loaded")
        except Exception as e:
    self.logger.warning(f"⚠️ Phonological engine not available: {e}")

    def _init_morphological_engine(self):
    """Initialize morphological analysis engine"""
        try:
            from core.nlp.morphology.engine import MorphologyEngine,
    self.engines['morphological'] = MorphologyEngine()
    self.logger.debug("✅ Morphological engine loaded")
        except Exception as e:
    self.logger.warning(f"⚠️ Morphological engine not available: {e}")

    def _init_inflection_engine(self):
    """Initialize inflection processing engine"""
        try:
            from core.nlp.inflection.engine import InflectionEngine,
    self.engines['inflection'] = InflectionEngine()
    self.logger.debug("✅ Inflection engine loaded")
        except Exception as e:
    self.logger.warning(f"⚠️ Inflection engine not available: {e}")

    def _init_syllable_engine(self):
    """Initialize syllable processing engine"""
        try:
            from core.nlp.syllable.engine import SyllableEngine,
    self.engines['syllable'] = SyllableEngine()
    self.logger.debug("✅ Syllable engine loaded")
        except Exception as e:
    self.logger.warning(f"⚠️ Syllable engine not available: {e}")

    def _init_weight_engine(self):
    """Initialize morphological weight analysis engine"""
        try:
            from core.nlp.weight.engine import WeightEngine,
    self.engines['weight'] = WeightEngine()
    self.logger.debug("✅ Weight engine loaded")
        except Exception as e:
    self.logger.warning(f"⚠️ Weight engine not available: {e}")

    def process_text(
    self, text: str, analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
    """
    Process Arabic text through all or specified engines,
    Args:
    text: Arabic text to process,
    analysis_types: List of analysis types ['phonological', 'morphological', etc.]
    If None, runs all available engines,
    Returns:
    Dictionary with analysis results from each engine
    """
        if not self.is_initialized:
    self.initialize()

        if analysis_types is None:
    analysis_types = list(self.engines.keys())

    results = {
    'input_text': text,
    'analysis_types': analysis_types,
    'results': {},
    'success': True,
    'errors': [],
    }

        for analysis_type in analysis_types:
            if analysis_type in self.engines:
                try:
    engine = self.engines[analysis_type]
    result = self._run_engine_analysis(engine, text, analysis_type)
    results['results'][analysis_type] = result,
    self.logger.debug(f"✅ {analysis_type} analysis completed")
                except Exception as e:
    error_msg = f"Error in {analysis_type} analysis: {e}"
    results['errors'].append(error_msg)
    self.logger.error(error_msg)
    results['success'] = False,
    else:
    warning_msg = f"Engine '{analysis_type}' not available"
    results['errors'].append(warning_msg)
    self.logger.warning(warning_msg)

    return results,
    def _run_engine_analysis(
    self, engine, text: str, engine_type: str
    ) -> Dict[str, Any]:
    """Run analysis on a specific engine"""
        # Standard interface - check for process_text first (our standard method)
        if hasattr(engine, 'process_text'):
    return engine.process_text(text)
        elif hasattr(engine, 'analyze'):
    return engine.analyze(text)
        elif hasattr(engine, 'process'):
    return engine.process(text)
        elif hasattr(engine, 'run'):
    return engine.run(text)
        else:
            # Fallback - try to call the engine directly,
    return engine(text)

    def get_available_engines(self) -> List[str]:
    """Get list of successfully loaded engines"""
    return list(self.engines.keys())

    def get_engine(self, engine_type: str):
    """Get specific engine instance"""
    return self.engines.get(engine_type)

    def health_check(self) -> Dict[str, Any]:
    """Check health status of all engines"""
    status = {
    'unified_engine': self.is_initialized,
    'engines': {},
    'total_engines': len(self.engines),
    'healthy_engines': 0,
    }

        for engine_type, engine in self.engines.items():
            try:
                # Try a simple health check,
    if hasattr(engine, 'health_check'):
    health = engine.health_check()
                else:
    health = True  # Assume healthy if loaded,
    status['engines'][engine_type] = health,
    if health:
    status['healthy_engines'] += 1,
    except Exception as e:
    status['engines'][engine_type] = f"Error: {e}"

    status['health_percentage'] = (
    status['healthy_engines'] / max(1, status['total_engines']) * 100
    )

    return status,
    def create_engine(config: Optional[Dict] = None) -> UnifiedArabicEngine:
    """Factory function to create a unified engine instance"""
    return UnifiedArabicEngine(config)


def main():
    """Demo/test function"""
    print("🚀 Unified Arabic Engine Demo")
    print("=" * 40)

    # Create engine,
    engine = create_engine()

    # Health check,
    health = engine.health_check()
    print(f"Engine Health: {health['health_percentage']:.1f}%")
    print(f"Available Engines: {engine.get_available_engines()}")

    # Test processing,
    test_text = "مرحبا بكم في محرك التحليل العربي"
    print(f"\nProcessing: {test_text}")

    results = engine.process_text(test_text)
    print(f"Analysis completed: {results['success']}")
    print(f"Results from {len(results['results'])} engines")

    if results['errors']:
    print("Errors:", results['errors'])


if __name__ == "__main__":
    main()
