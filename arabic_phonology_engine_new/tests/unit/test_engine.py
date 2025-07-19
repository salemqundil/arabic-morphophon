"""
Test script for Arabic morphological analysis engine.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.engine_simple import ArabicAnalysisEngine


def test_engine():
    """Test the Arabic analysis engine."""
    print("🔬 Testing Arabic Morphological Analysis Engine")
    print("=" * 50)

    # Initialize engine
    engine = ArabicAnalysisEngine()

    # Simple test case
    root = ["K", "T", "B"]
    vowels = ["", "", "SUKUN", "FATHA", "FATHA", "SUKUN"]
    word = ["PFX_al", "H", "K", "T", "A", "B"]

    print(f"Testing root: {root}")
    print(f"Word: {word}")
    print(f"Vowels: {vowels}")

    try:
        result = engine.analyze(root, vowels, word)

        print(f"
✅ Analysis successful!")
        print(f"After Φ rules: {result.after_phi}")
        print(f"Syllables: {result.syllables}")
        print(f"Embedding length: {result.embedding_length}")
        print(f"Inflection features sample: {result.inflection_features[:5]}")

        # Test JSON export
        json_output = result.to_json()
        print(f"
📄 JSON export length: {len(json_output)} characters")
        print("JSON preview:")
        print(json_output[:300] + "..." if len(json_output) > 300 else json_output)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_engine()
    if success:
        print("
🎉 All tests passed!")
    else:
        print("
💥 Tests failed!")
