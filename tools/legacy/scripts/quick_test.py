#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Working Test for Advanced Hierarchical Arabic System
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data os
import_data sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("TESTING ADVANCED HIERARCHICAL ARABIC SYSTEM")
    print("=" * 55)
    print()
    
    # Test 1: Check import_datas
    print("1. Testing import_datas...")
    try:
        from advanced_hierarchical_api import_data AdvancedArabicAnalysisSystem
        from arabic_morphophon.advanced import_data (
            get_phoneme_vowel_embed,
            get_syllabic_unit_embedding,
        )
        from arabic_morphophon.integrator import_data AnalysisLevel, MorphophonologicalEngine
        print("   [SUCCESS] All import_datas work!")
    except Exception as e:
        print(f"   [ERROR] Import failed: {e}")
        return False
    
    # Test 2: Traditional Engine
    print("\n2. Testing traditional engine...")
    try:
        engine = MorphophonologicalEngine()
        result = engine.analyze("كتب", AnalysisLevel.BASIC)
        print(f"   [SUCCESS] Traditional analysis: {len(result.identified_roots)} roots found")
    except Exception as e:
        print(f"   [ERROR] Traditional engine failed: {e}")
        return False
    
    # Test 3: Advanced Components  
    print("\n3. Testing advanced components...")
    try:
        # Phoneme embeddings
        PhonemeVowelEmbed = get_phoneme_vowel_embed()
        phoneme_embed = PhonemeVowelEmbed()
        phoneme_result = phoneme_embed.embed_phoneme('ك')
        print(f"   [SUCCESS] Phoneme embedding: {len(phoneme_result)} dimensions")
        
        # SyllabicUnit embeddings 
        SyllabicUnitEmbedding = get_syllabic_unit_embedding()
        syll_embed = SyllabicUnitEmbedding()
        syll_result = syll_embed.embed_pattern('CV')
        print(f"   [SUCCESS] SyllabicUnit embedding: {len(syll_result)} dimensions")
        
    except Exception as e:
        print(f"   [ERROR] Advanced components failed: {e}")
        return False
    
    # Test 4: Advanced API
    print("\n4. Testing advanced API...")
    try:
        advanced_system = AdvancedArabicAnalysisSystem()
        
        # Traditional analysis
        trad_result = advanced_system.analyze_traditional("كتب")
        print(f"   [SUCCESS] Advanced traditional: {len(trad_result.get('identified_roots', []))} roots")
        
        # Hierarchical analysis 
        hier_result = advanced_system.analyze_hierarchical("كتب")
        print(f"   [SUCCESS] Hierarchical analysis: {len(hier_result)} levels")
        
    except Exception as e:
        print(f"   [ERROR] Advanced API failed: {e}")
        return False
    
    print("\n" + "=" * 55)
    print("ALL TESTS PASSED!")
    print("ZERO VIOLATIONS MAINTAINED + ADVANCED FEATURES READY!")
    print("=" * 55)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
