#!/usr/bin/env python3
"""
🎯 Arabic Particles Engine Demonstration
Shows how to call and use the GrammaticalParticlesEngine
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
from pathlib import_data Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from engines.nlp.particles.engine import_data GrammaticalParticlesEngine

def demonstrate_particles_engine():
    """Demonstrate how to call and use the Particles Engine"""
    
    print("🚀 Arabic Grammatical Particles Engine Demo")
    print("=" * 50)
    
    # 1. Initialize the Particles Engine
    print("\n1️⃣ Initializing Particles Engine...")
    particles_engine = GrammaticalParticlesEngine()
    
    # 2. Test single particle analysis
    print("\n2️⃣ Single Particle Analysis:")
    test_particles = [
        "إن",      # Conditional particle
        "هل",      # Interrogative particle  
        "لا",      # Negation particle
        "يا",      # Vocative particle
        "الذي",    # Relative particle
        "هذا",     # Demonstrative particle
        "في",      # Preposition particle
        "لكن"      # Adversative particle
    ]
    
    for particle in test_particles:
        print(f"\n📝 Analyzing: {particle}")
        result = particles_engine.analyze(particle)
        
        print(f"   Category: {result['category']}")
        print(f"   Phonemes: {' '.join(result['phonemes'])}")
        print(f"   SyllabicUnits: {result['syllabic_units']}")
        print(f"   Processing Time: {result['analysis_metadata']['processing_time_ms']}ms")
        
        if result['morphological_features']:
            print(f"   Features: {result['morphological_features']}")
    
    # 3. Batch analysis
    print("\n3️⃣ Batch Analysis:")
    batch_particles = ["إن", "لكن", "هل", "لا", "يا"]
    batch_results = particles_engine.batch_analyze(batch_particles)
    
    print(f"   Processed {len(batch_results)} particles in batch")
    for i, result in enumerate(batch_results):
        print(f"   {i+1}. {result['particle']} → {result['category']}")
    
    # 4. Get engine statistics
    print("\n4️⃣ Engine Statistics:")
    stats = particles_engine.get_statistics()
    print(f"   Total Analyses: {stats['engine_info']['total_analyses']}")
    print(f"   Classification Rate: {stats['success_rates']['classification_rate']:.1f}%")
    print(f"   Phoneme Extraction Rate: {stats['success_rates']['phoneme_extraction_rate']:.1f}%")
    
    # 5. Get supported categories
    print("\n5️⃣ Supported Categories:")
    categories = particles_engine.get_supported_categories()
    for category, description in categories.items():
        print(f"   {category}: {description}")
    
    # 6. Find particles by category
    print("\n6️⃣ Particles by Category:")
    test_category = "حروف_الاستفهام"  # Interrogative particles
    particles_in_category = particles_engine.find_particles_by_category(test_category)
    print(f"   {test_category}: {particles_in_category}")
    
    # 7. Validate engine
    print("\n7️⃣ Engine Validation:")
    validation = particles_engine.validate_engine()
    print(f"   Status: {validation['validation_status']}")
    print(f"   Success Rate: {validation['success_rate']}")
    print(f"   Engine Ready: {validation['engine_ready']}")
    
    print("\n✅ Particles Engine Demo Complete!")
    return particles_engine

def integrate_with_main_platform(particles_engine):
    """Show how to integrate particles engine with main platform"""
    
    print("\n🔗 Integration with Main Platform:")
    print("=" * 40)
    
    # Example integration function
    def analyze_text_with_particles(text: str):
        """Analyze text and extract particles"""
        words = text.split()
        particle_analyses = []
        
        for word in words:
            # Check if word is a particle
            result = particles_engine.analyze(word)
            if result['analysis_metadata'].get('is_recognized_particle', False):
                particle_analyses.append({
                    'word': word,
                    'category': result['category'],
                    'phonemes': result['phonemes'],
                    'syllabic_units': result['syllabic_units']
                })
        
        return particle_analyses
    
    # Test with sample Arabic text
    sample_text = "هل كتب الطالب الواجب في المكتبة؟"
    print(f"\n📖 Sample Text: {sample_text}")
    
    particles_found = analyze_text_with_particles(sample_text)
    print(f"\n🔍 Particles Found: {len(particles_found)}")
    
    for particle in particles_found:
        print(f"   • {particle['word']} ({particle['category']})")
        print(f"     Phonemes: {' '.join(particle['phonemes'])}")
        print(f"     SyllabicUnits: {particle['syllabic_units']}")

if __name__ == "__main__":
    try:
        # Run the demonstration
        engine = demonstrate_particles_engine()
        
        # Show integration example
        integrate_with_main_platform(engine)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import_data traceback
        traceback.print_exc()
