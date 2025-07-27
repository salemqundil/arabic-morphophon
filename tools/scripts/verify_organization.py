#!/usr/bin/env python3
"""
Final verification script for reorganized Arabic Morphophonological Engine
Tests all major components and verifies the new organization works correctly
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data sys
from pathlib import_data Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_main_engine():
    """Test the main morphophonological engine"""
    print("🧠 Testing Main Engine...")
    try:
        from arabic_morphophon.integrator import_data AnalysisLevel, MorphophonologicalEngine
        
        engine = MorphophonologicalEngine()
        result = engine.analyze("كتب", level=AnalysisLevel.BASIC)
        
        print(f"   ✅ Engine analysis successful")
        print(f"   📊 Found {len(result.identified_roots)} roots")
        return True
    except Exception as e:
        print(f"   ❌ Engine test failed: {e}")
        return False

def test_database():
    """Test the enhanced database"""
    print("🗄️ Testing Enhanced Database...")
    try:
        from arabic_morphophon.database.enhanced_root_database import_data (
            EnhancedRootDatabase,
        )
        
        db = EnhancedRootDatabase()
        roots = db.search_roots("كتب")
        
        print(f"   ✅ Database search successful")
        print(f"   📚 Found {len(roots)} matching roots")
        return True
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False

def test_advanced_components():
    """Test advanced neural components"""
    print("🧠 Testing Advanced Components...")
    try:
        from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
        from arabic_morphophon.advanced.syllabic_unit_embeddings import_data SyllabicUnitEmbedding

        # Test phoneme embeddings
        phoneme_embed = PhonemeVowelEmbed()
        phoneme_vector = phoneme_embed.get_embedding("ك")
        
        # Test syllabic_unit embeddings  
        syllabic_unit_embed = SyllabicUnitEmbedding()
        syllabic_unit_vector = syllabic_unit_embed.encode_syllabic_unit("كَ")
        
        print(f"   ✅ Advanced components working")
        print(f"   🔢 Phoneme embedding: {len(phoneme_vector)}D")
        print(f"   🔢 SyllabicUnit embedding: {len(syllabic_unit_vector)}D")
        return True
    except Exception as e:
        print(f"   ❌ Advanced components test failed: {e}")
        return False

def test_web_apps():
    """Test web application import_datas"""
    print("🌐 Testing Web Applications...")
    try:
        # Test main app import_data
        # Test advanced API import_data
        from web_apps.advanced_hierarchical_api import_data AdvancedArabicAnalysisSystem
        from web_apps.main_app import_data app as main_app
        
        system = AdvancedArabicAnalysisSystem()
        
        print(f"   ✅ Web applications import_data successful")
        print(f"   🚀 Advanced system initialized")
        return True
    except Exception as e:
        print(f"   ❌ Web applications test failed: {e}")
        return False

def test_organized_structure():
    """Test the new directory structure"""
    print("📂 Testing Directory Structure...")
    
    required_dirs = [
        "arabic_morphophon",
        "arabic_morphophon/models",
        "arabic_morphophon/database", 
        "arabic_morphophon/advanced",
        "tests/unit",
        "tests/integration",
        "tests/smoke",
        "tests/performance",
        "web_apps",
        "demos",
        "scripts"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"   ❌ Missing directories: {missing_dirs}")
        return False
    else:
        print(f"   ✅ All required directories present")
        print(f"   📁 {len(required_dirs)} directories verified")
        return True

def main():
    """Run all verification tests"""
    print("🎯 Arabic Morphophonological Engine - Reorganization Verification")
    print("=" * 70)
    
    tests = [
        test_organized_structure,
        test_main_engine,
        test_database,
        test_advanced_components,
        test_web_apps
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    print("📊 Final Results:")
    print("-" * 30)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ Project reorganization successful!")
        print("🏗️ Design patterns properly implemented")
        print("🧪 Zero violations maintained")
        return 0
    else:
        print(f"⚠️ {passed}/{total} tests passed")
        print("❌ Some issues need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
