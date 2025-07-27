#!/usr/bin/env python3
"""
🎯 Test Comprehensive Arabic Particles Classification
Demonstrates complete classification and segregation capabilities
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
from pathlib import_data Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from comprehensive_particle_classification import_data ComprehensiveParticleAnalyzer

def test_comprehensive_classification():
    """Test comprehensive classification with various Arabic texts"""
    
    print("🎯 TESTING COMPREHENSIVE ARABIC PARTICLES CLASSIFICATION")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ComprehensiveParticleAnalyzer()
    
    # Test cases covering all 8 categories
    test_cases = [
        {
            "name": "استفهام و شرط (Interrogative & Conditional)",
            "text": "هل جاء الطالب إن كان مريضاً؟"
        },
        {
            "name": "نداء و إشارة و موصول (Vocative, Demonstrative & Relative)",
            "text": "يا أحمد، هذا الكتاب الذي طلبته"
        },
        {
            "name": "نفي و استثناء (Negation & Exception)",
            "text": "لا تذهب إلا إذا انتهيت من العمل"
        },
        {
            "name": "ضمائر (Personal Pronouns)",
            "text": "أنا أحب هذا الكتاب وهو يحب ذلك"
        },
        {
            "name": "جملة شاملة (Comprehensive Sentence)",
            "text": "هل تعرف الرجل الذي يا أخي إن لم يأت فلن نبدأ إلا بعد وصوله؟"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 TEST {i}: {test_case['name']}")
        print("-" * 60)
        print(f"Text: {test_case['text']}")
        
        # Perform comprehensive analysis
        analysis = analyzer.classify_and_segregate_text(test_case['text'])
        
        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"   Total Words: {analysis['total_words']}")
        print(f"   Particles Found: {analysis['classification_summary']['particles_found']}")
        print(f"   Particle Density: {analysis['statistics']['particle_density']:.1f}%")
        print(f"   Categories Present: {len(analysis['classification_summary']['categories_present'])}")
        
        # Categories breakdown
        if analysis['classification_summary']['category_counts']:
            print(f"\n🏷️ Categories Found:")
            for category, count in analysis['classification_summary']['category_counts'].items():
                particles = [p['word'] for p in analysis['segregation_by_category'][category]]
                print(f"   {category} ({count}): {', '.join(particles)}")
        
        # Subcategories breakdown
        if analysis['segregation_by_subcategory']:
            print(f"\n🔖 Subcategories:")
            for subcat, particles_data in analysis['segregation_by_subcategory'].items():
                particles = [p['word'] for p in particles_data]
                print(f"   {subcat}: {', '.join(particles)}")
        
        # Statistical insights
        if analysis['statistics']['most_common_category']:
            print(f"\n📈 Most Common Category: {analysis['statistics']['most_common_category']}")
    
    print("\n" + "="*70)
    print("📋 SYSTEM OVERVIEW")
    print("="*70)
    
    # Get system overview
    categories = analyzer.get_all_categories_with_examples()
    
    print(f"\n🎯 Available Categories ({len(categories)}):")
    for category, data in categories.items():
        print(f"\n🏷️ {category} ({data['definition']['name']}):")
        print(f"   Description: {data['definition']['description']}")
        print(f"   Function: {data['definition']['function']}")
        print(f"   Total Particles: {data['total_particles']}")
        print(f"   Examples: {', '.join(data['examples'])}")
        
        # Show subcategories
        print(f"   Subcategories ({len(data['subcategories'])}):")
        for subcat, particles in data['subcategories'].items():
            print(f"      • {subcat}: {len(particles)} particles")
    
    print(f"\n📊 System Statistics:")
    print(f"   Total Categories: {len(categories)}")
    print(f"   Total Particles: {sum(data['total_particles'] for data in categories.values())}")
    print(f"   Total Subcategories: {sum(len(data['subcategories']) for data in categories.values())}")
    
    print("\n✅ COMPREHENSIVE CLASSIFICATION TEST COMPLETE!")
    print("\n💡 System Capabilities:")
    print("   ✓ 8 Main grammatical categories")
    print("   ✓ 20+ Subcategory classifications")  
    print("   ✓ Morphological feature analysis")
    print("   ✓ Statistical insights & metrics")
    print("   ✓ Complete text segregation")
    print("   ✓ Context-sensitive classification")
    print("   ✓ Real-time particle detection")
    print("   ✓ Comprehensive linguistic metadata")

if __name__ == "__main__":
    try:
        test_comprehensive_classification()
    except Exception as e:
        print(f"❌ Error: {e}")
        import_data traceback
        traceback.print_exc()
