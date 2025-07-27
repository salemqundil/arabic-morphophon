#!/usr/bin/env python3
"""
🎯 Integration Guide: How to Call Particles Engine in Your Platform
Complete integration examples for the Arabic Grammatical Particles Engine
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
from pathlib import_data Path
from typing import_data Any, Dict, List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from engines.nlp.particles.engine import_data GrammaticalParticlesEngine

class ArabicPlatformWithParticles:
    """
    Enhanced Arabic morphophonological platform with particles engine integration
    """
    
    def __init__(self):
        """Initialize platform with all engines including particles"""
        print("🚀 Initializing Enhanced Arabic Platform...")
        
        # Initialize particles engine
        self.particles_engine = GrammaticalParticlesEngine()
        
        # Mock other engines (replace with your actual engines)
        self.phonology_engine = None  # Your phonology engine
        self.syllabic_unit_engine = None   # Your syllabic_unit engine  
        self.root_engine = None       # Your root engine
        self.inflection_engine = None # Your inflection engine
        
        print("✅ Platform initialized with particles engine")
    
    def analyze_complete_text(self, text: str) -> Dict[str, Any]:
        """
        Complete text analysis including particles identification
        
        Args:
            text: Arabic text to analyze
            
        Returns:
            Comprehensive analysis including particles
        """
        words = text.split()
        analysis_result = {
            'original_text': text,
            'words': [],
            'particles': [],
            'non_particles': [],
            'statistics': {}
        }
        
        for word in words:
            # Clean word (remove punctuation)
            clean_word = word.strip('؟!،.')
            
            # Analyze with particles engine
            particle_result = self.particles_engine.analyze(clean_word)
            
            word_analysis = {
                'word': word,
                'clean_word': clean_word,
                'is_particle': particle_result['analysis_metadata'].get('is_recognized_particle', False),
                'particle_category': particle_result['category'],
                'phonemes': particle_result['phonemes'],
                'syllabic_units': particle_result['syllabic_units'],
                'morphological_features': particle_result['morphological_features']
            }
            
            analysis_result['words'].append(word_analysis)
            
            # Separate particles from non-particles
            if word_analysis['is_particle']:
                analysis_result['particles'].append(word_analysis)
            else:
                analysis_result['non_particles'].append(word_analysis)
        
        # Calculate statistics
        analysis_result['statistics'] = {
            'total_words': len(words),
            'particle_count': len(analysis_result['particles']),
            'non_particle_count': len(analysis_result['non_particles']),
            'particle_percentage': (len(analysis_result['particles']) / len(words) * 100) if words else 0
        }
        
        return analysis_result
    
    def extract_particles_only(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract only grammatical particles from text
        
        Args:
            text: Arabic text
            
        Returns:
            List of particle analyses
        """
        words = text.split()
        particles = []
        
        for word in words:
            clean_word = word.strip('؟!،.')
            result = self.particles_engine.analyze(clean_word)
            
            if result['analysis_metadata'].get('is_recognized_particle', False):
                particles.append({
                    'particle': clean_word,
                    'category': result['category'],
                    'description': result.get('category_description', ''),
                    'phonemes': result['phonemes'],
                    'syllabic_units': result['syllabic_units'],
                    'processing_time': result['analysis_metadata']['processing_time_ms']
                })
        
        return particles
    
    def analyze_particles_by_category(self, text: str) -> Dict[str, List[str]]:
        """
        Group particles by their grammatical categories
        
        Args:
            text: Arabic text
            
        Returns:
            Dictionary with categories as keys and particles as values
        """
        particles = self.extract_particles_only(text)
        categorized = {}
        
        for particle in particles:
            category = particle['category']
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(particle['particle'])
        
        return categorized
    
    def get_particle_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get detailed statistics about particles in text
        
        Args:
            text: Arabic text
            
        Returns:
            Detailed particle statistics
        """
        analysis = self.analyze_complete_text(text)
        categories = self.analyze_particles_by_category(text)
        
        return {
            'text_length': len(text),
            'word_count': analysis['statistics']['total_words'],
            'particle_count': analysis['statistics']['particle_count'],
            'particle_percentage': analysis['statistics']['particle_percentage'],
            'categories_found': list(categories.keys()),
            'category_distribution': {cat: len(particles) for cat, particles in categories.items()},
            'most_common_category': max(categories.keys(), key=lambda k: len(categories[k])) if categories else None,
            'particles_by_category': categories
        }

def demonstrate_integration():
    """Demonstrate platform integration with particles engine"""
    
    print("\n" + "="*60)
    print("🔧 PARTICLES ENGINE INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Initialize enhanced platform
    platform = ArabicPlatformWithParticles()
    
    # Test texts
    test_texts = [
        "هل كتب الطالب الواجب في المكتبة؟",
        "إن الله غفور رحيم",
        "يا أحمد، هل أنت مستعد للامتحان؟",
        "الذي يدرس بجد ينجح في الحياة",
        "لا تنس أن تحضر الكتاب غداً"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        print("-" * 50)
        
        # 1. Complete analysis
        complete_analysis = platform.analyze_complete_text(text)
        print(f"   Words: {complete_analysis['statistics']['total_words']}")
        print(f"   Particles: {complete_analysis['statistics']['particle_count']}")
        print(f"   Particle %: {complete_analysis['statistics']['particle_percentage']:.1f}%")
        
        # 2. Extract particles only
        particles = platform.extract_particles_only(text)
        print(f"   Found Particles: {[p['particle'] for p in particles]}")
        
        # 3. Categorize particles
        categories = platform.analyze_particles_by_category(text)
        for category, particle_list in categories.items():
            print(f"   {category}: {particle_list}")
        
        # 4. Get statistics
        stats = platform.get_particle_statistics(text)
        if stats['most_common_category']:
            print(f"   Most Common Category: {stats['most_common_category']}")

def api_integration_example():
    """Show how to integrate particles engine into Flask API"""
    
    print("\n" + "="*60)
    print("🌐 FLASK API INTEGRATION EXAMPLE")
    print("="*60)
    
    # Simulate Flask route integration
    def flask_route_example():
        """
        Example Flask route that uses particles engine
        Add this to your production_platform_enhanced.py
        """
        code_example = '''
@app.route('/api/analyze-particles', methods=['POST'])
def analyze_particles():
    """API endpoint for particles analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Initialize particles engine (or use global instance)
        particles_engine = GrammaticalParticlesEngine()
        
        # Analyze particles in text
        words = text.split()
        results = []
        
        for word in words:
            clean_word = word.strip('؟!،.')
            particle_analysis = particles_engine.analyze(clean_word)
            
            if particle_analysis['analysis_metadata'].get('is_recognized_particle', False):
                results.append({
                    'word': word,
                    'category': particle_analysis['category'],
                    'phonemes': particle_analysis['phonemes'],
                    'syllabic_units': particle_analysis['syllabic_units']
                })
        
        return jsonify({
            'status': 'success',
            'original_text': text,
            'particles_found': len(results),
            'particles': results,
            'statistics': {
                'total_words': len(words),
                'particle_percentage': len(results) / len(words) * 100 if words else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        '''
        
        print("🔧 Add this route to your Flask app:")
        print(code_example)
    
    flask_route_example()

if __name__ == "__main__":
    try:
        # Run integration demonstration
        demonstrate_integration()
        
        # Show API integration
        api_integration_example()
        
        print("\n✅ Integration demonstration complete!")
        print("\n💡 Next Steps:")
        print("   1. Add particles engine to your main platform")
        print("   2. Integrate the Flask API route")
        print("   3. Test with your Arabic texts")
        print("   4. Monitor performance with engine statistics")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import_data traceback
        traceback.print_exc()
