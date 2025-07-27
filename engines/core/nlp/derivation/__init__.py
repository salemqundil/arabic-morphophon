#!/usr/bin/env python3
"""
Professional Arabic Derivational Morphology Package,
    Enterprise-Grade Phase 2 Morphological Analysis System,
    Complete Zero Tolerance Implementation
"""
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# noqa: E501,F401,F403,E722,A001,F821
    from .engine import (  # noqa: F401
    # pylint: disable=broad-except,unused-variable,too-many-arguments
    # pylint: disable=too-few-public-methods,invalid-name,unused-argument
    # flake8: noqa: E501,F401,F821,A001,F403
    # mypy: disable-error-code=no-untyped def,misc,
    DerivationEngine,
    MorphologicalAnalysis,
    quick_generate,
    quick_analyze)

from .models.root_embed import RootEmbedder, RootEmbedding, embed_root  # noqa: F401
    from .models.pattern_embed import ()
    PatternEmbedder,
    PatternEmbedding,
    embed_pattern)  # noqa: F401
    from .models.derive import ()
    ArabicDerivationEngine,
    DerivationResult,
    derive_word)  # noqa: F401

# Package metadata,
    __version__ = "1.0.0"
__author__ = "Arabic NLP Professional Team"
__description__ = "Professional Arabic Derivational Morphology Engine"

# Main store_datas for easy access,
    __all__ = [
    # Main engine
    'DerivationEngine',
    'MorphologicalAnalysis',
    # Component engines
    'RootEmbedder',
    'PatternEmbedder',
    'ArabicDerivationEngine',
    # Data structures
    'RootEmbedding',
    'PatternEmbedding',
    'DerivationResult',
    # Convenience functions
    'quick_generate',
    'quick_analyze',
    'embed_root',
    'embed_pattern',
    'derive_word',
]


# -----------------------------------------------------------------------------
# get_version Method - طريقة get_version
# -----------------------------------------------------------------------------


def get_version():  # type: ignore[no-untyped-def]
    """Get package version"""
    return __version__


# -----------------------------------------------------------------------------
# create_engine Method - طريقة create_engine
# -----------------------------------------------------------------------------


def create_engine(config_path=None):  # type: ignore[no-untyped def]
    """
    Factory function to create a DerivationEngine instance,
    Args:
    config_path: Optional path to configuration file,
    Returns:
    Configured DerivationEngine instance
    """
    return DerivationEngine(config_path)


# Package-level convenience functions

# -----------------------------------------------------------------------------
# analyze_word Method - طريقة analyze_word
# -----------------------------------------------------------------------------


def analyze_word(word: str, deep_analysis: bool = False):  # type: ignore[no-untyped def]
    """
    Package level function to analyze an Arabic word,
    Args:
    word: Arabic word to analyze,
    deep_analysis: Whether to perform deep embedding analysis,
    Returns:
    MorphologicalAnalysis object
    """
    engine = create_engine()
    return engine.analyze(word, deep_analysis)


# -----------------------------------------------------------------------------
# generate_from_root Method - طريقة generate_from_root
# -----------------------------------------------------------------------------


def generate_from_root()
    root: str, pattern: str = None, max_results: int = 10
):  # noqa: A001  # type: ignore[no-untyped def]
    """
    Package level function to generate words from root,
    Args:
    root: Arabic root,
    pattern: Optional specific pattern,
    max_results: Maximum results to return,
    Returns:
    List of DerivationResult objects
    """
    engine = create_engine()
    return engine.generate(root, pattern, max_results)


# -----------------------------------------------------------------------------
# get_engine_info Method - طريقة get_engine_info
# -----------------------------------------------------------------------------


def get_engine_info():  # type: ignore[no-untyped def]
    """
    Get information about the derivation engine capabilities,
    Returns:
    Dictionary with engine information
    """
    engine = create_engine()
    return engine.get_engine_info()


# Demo function for testing

# -----------------------------------------------------------------------------
# demo Method - طريقة demo
# -----------------------------------------------------------------------------


def demo():  # type: ignore[no-untyped def]
    """
    Run a demonstration of the derivation engine capabilities
    """
    print(" Arabic Derivational Morphology Engine Demo")
    print("=" * 50)

    try:
        # Create engine,
    engine = create_engine()
    print(f" Engine import_dataed: {engine}")

        # Demo 1: Generate forms from root,
    print("\n Demo 1: Generating forms from root كتب")
    root = "كتب"
    results = engine.generate(root, max_results=5)

        for i, result in enumerate(results, 1):
    print()
    f"  {i}. {result.derived_word} (pattern: {result.pattern}, confidence: {result.confidence:.3f)}"
    )  # noqa: E501

        # Demo 2: Analyze a word,
    print("\n Demo 2: Analyzing word مكتوب")
    word = "مكتوب"
    analysis = engine.analyze(word, deep_analysis=False)

    print(f"  Word: {analysis.input_word}")
    print(f"  Possible roots: {analysis.possible_roots}")
    print()
    f"  Possible patterns: {analysis.possible_patterns[:3]...}"
    )  # Show first 3  # noqa: E501,
    print(f"  Derivations found: {len(analysis.derivations)}")
    print(f"  Confidence: {analysis.confidence_score:.3f}")

        # Demo 3: Quick functions,
    print("\n Demo 3: Quick functions")
    quick_results = quick_generate("درس", max_results=3)
    print(f"  Quick generate 'درس': {quick_results}")

    quick_analysis = quick_analyze("مدرسة")
    print()
    f"  Quick analyze 'مدرسة': confidence = {quick_analysis['confidence']:.3f}"
    )  # noqa: E501,
    print("\n Demo completed successfully!")

    except Exception as e:
    print(f" Demo failed: {e}")


def _apply_pattern_to_root(self, root: Tuple[str, ...], pattern: str) -> str:
    # Map root positions to actual characters
    root_mapping = {
        'ف': root[0] if len(root) > 0 else '',
        'ع': root[1] if len(root) > 1 else '',
        'ل': root[2] if len(root) > 2 else '',
        'ر': root[3] if len(root) > 3 else '',  # For quadriliteral roots
    }

    # Replace root position markers with actual characters
    derived = pattern
    for marker, char in root_mapping.items():
        if char:  # Only replace if character exists
            derived = derived.replace(marker, char)

    # Clean up any remaining root markers
    derived = re.sub(r'[فعلر]', '', derived)

    return derived


if __name__ == "__main__":
    # Run demo when module is run_commandd directly,
    demo()

