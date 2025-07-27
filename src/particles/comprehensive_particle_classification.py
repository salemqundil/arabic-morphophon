#!/usr/bin/env python3
"""
🎯 Comprehensive Arabic Particles Classification & Segregation System
Complete analysis and categorization of all Arabic grammatical particles
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data sys
from collections import_data defaultdict
from pathlib import_data Path
from typing import_data Any, Dict, List, Set

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from engines.nlp.particles.engine import_data GrammaticalParticlesEngine

class ComprehensiveParticleAnalyzer:
    """
    Advanced Arabic Particles Classification and Segregation System
    Provides comprehensive analysis of all grammatical particle categories
    """
    
    def __init__(self):
        """Initialize the comprehensive analyzer"""
        self.particles_engine = GrammaticalParticlesEngine()
        
        # Comprehensive category definitions with subcategories
        self.category_definitions = {
            "شرط": {
                "name": "Conditional Particles",
                "description": "حروف الشرط - Particles that introduce conditional clauses",
                "subcategories": {
                    "شرط_جازم": "Conditional particles that affect mood (جازم)",
                    "شرط_غير_جازم": "Non-modal conditional particles",
                    "شرط_زمني": "Temporal conditional particles"
                },
                "function": "Introduce conditional or hypothetical statements"
            },
            "استفهام": {
                "name": "Interrogative Particles", 
                "description": "حروف الاستفهام - Question formation particles",
                "subcategories": {
                    "استفهام_نعم_لا": "Yes/No question particles",
                    "استفهام_مفتوح": "Open-ended question particles",
                    "استفهام_تقريري": "Rhetorical question particles"
                },
                "function": "Form questions and interrogative expressions"
            },
            "استثناء": {
                "name": "Exception Particles",
                "description": "حروف الاستثناء - Exception and exclusion particles", 
                "subcategories": {
                    "استثناء_مفرغ": "Complete exception particles",
                    "استثناء_ناقص": "Incomplete exception particles",
                    "استثناء_متصل": "Connected exception particles"
                },
                "function": "Express exceptions and exclusions"
            },
            "نفي": {
                "name": "Negation Particles",
                "description": "حروف النفي - Negation and denial particles",
                "subcategories": {
                    "نفي_مطلق": "Absolute negation particles",
                    "نفي_مقيد": "Conditional negation particles", 
                    "نفي_استقبال": "Future negation particles"
                },
                "function": "Negate verbs, nouns, and sentences"
            },
            "إشارة": {
                "name": "Demonstrative Particles",
                "description": "أسماء الإشارة - Demonstrative pronouns and particles",
                "subcategories": {
                    "إشارة_قريب": "Near demonstratives (this/these)",
                    "إشارة_بعيد": "Far demonstratives (that/those)",
                    "إشارة_مكان": "Locative demonstratives (here/there)"
                },
                "function": "Point to or indicate specific referents"
            },
            "نداء": {
                "name": "Vocative Particles",
                "description": "حروف النداء - Calling and addressing particles",
                "subcategories": {
                    "نداء_قريب": "Near vocatives",
                    "نداء_بعيد": "Distant vocatives",
                    "نداء_تعجب": "Exclamatory vocatives"
                },
                "function": "Call attention or address someone/something"
            },
            "موصول": {
                "name": "Relative Particles", 
                "description": "الأسماء الموصولة - Relative pronouns",
                "subcategories": {
                    "موصول_عاقل": "Rational being relatives",
                    "موصول_غير_عاقل": "Non-rational relatives",
                    "موصول_مشترك": "Common relatives"
                },
                "function": "Connect relative clauses to main clauses"
            },
            "ضمير": {
                "name": "Personal Pronouns",
                "description": "الضمائر المنفصلة - Detached personal pronouns",
                "subcategories": {
                    "ضمير_رفع": "Nominative pronouns",
                    "ضمير_نصب": "Accusative pronouns", 
                    "ضمير_جر": "Genitive pronouns"
                },
                "function": "Replace or refer to nouns and noun phrases"
            }
        }
        
        # Extended particle mappings with detailed classification
        self.extended_particles = {
            # Conditional Particles (شرط)
            "إن": {"category": "شرط", "subcategory": "شرط_جازم", "mood_effect": True},
            "إذا": {"category": "شرط", "subcategory": "شرط_غير_جازم", "mood_effect": False},
            "كلما": {"category": "شرط", "subcategory": "شرط_زمني", "mood_effect": False},
            "لو": {"category": "شرط", "subcategory": "شرط_غير_جازم", "mood_effect": False},
            "لولا": {"category": "شرط", "subcategory": "شرط_غير_جازم", "mood_effect": False},
            "أن": {"category": "شرط", "subcategory": "شرط_غير_جازم", "mood_effect": False},
            "كي": {"category": "شرط", "subcategory": "شرط_غير_جازم", "mood_effect": False},
            "لكي": {"category": "شرط", "subcategory": "شرط_غير_جازم", "mood_effect": False},
            
            # Interrogative Particles (استفهام) 
            "هل": {"category": "استفهام", "subcategory": "استفهام_نعم_لا", "answer_type": "yes_no"},
            "أ": {"category": "استفهام", "subcategory": "استفهام_نعم_لا", "answer_type": "yes_no"},
            "من": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "person"},
            "ما": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "thing"},
            "ماذا": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "thing"},
            "أين": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "place"},
            "متى": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "time"},
            "كيف": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "manner"},
            "لماذا": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "reason"},
            "كم": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "quantity"},
            "أي": {"category": "استفهام", "subcategory": "استفهام_مفتوح", "answer_type": "choice"},
            
            # Exception Particles (استثناء)
            "إلا": {"category": "استثناء", "subcategory": "استثناء_متصل", "exception_type": "connected"},
            "غير": {"category": "استثناء", "subcategory": "استثناء_متصل", "exception_type": "connected"},
            "سوى": {"category": "استثناء", "subcategory": "استثناء_متصل", "exception_type": "connected"},
            "خلا": {"category": "استثناء", "subcategory": "استثناء_منقطع", "exception_type": "disconnected"},
            "عدا": {"category": "استثناء", "subcategory": "استثناء_منقطع", "exception_type": "disconnected"},
            "حاشا": {"category": "استثناء", "subcategory": "استثناء_منقطع", "exception_type": "disconnected"},
            
            # Negation Particles (نفي)
            "لا": {"category": "نفي", "subcategory": "نفي_مطلق", "tense_scope": "present"},
            "لن": {"category": "نفي", "subcategory": "نفي_استقبال", "tense_scope": "future"},
            "لم": {"category": "نفي", "subcategory": "نفي_ماضي", "tense_scope": "past"},
            "ما": {"category": "نفي", "subcategory": "نفي_مطلق", "tense_scope": "general"},
            "ليس": {"category": "نفي", "subcategory": "نفي_مطلق", "tense_scope": "present"},
            "ليت": {"category": "نفي", "subcategory": "نفي_تمني", "tense_scope": "conditional"},
            
            # Demonstrative Particles (إشارة)
            "هذا": {"category": "إشارة", "subcategory": "إشارة_قريب", "distance": "near", "gender": "masculine"},
            "هذه": {"category": "إشارة", "subcategory": "إشارة_قريب", "distance": "near", "gender": "feminine"},
            "ذلك": {"category": "إشارة", "subcategory": "إشارة_بعيد", "distance": "far", "gender": "masculine"},
            "تلك": {"category": "إشارة", "subcategory": "إشارة_بعيد", "distance": "far", "gender": "feminine"},
            "أولئك": {"category": "إشارة", "subcategory": "إشارة_بعيد", "distance": "far", "number": "plural"},
            "هؤلاء": {"category": "إشارة", "subcategory": "إشارة_قريب", "distance": "near", "number": "plural"},
            "هنا": {"category": "إشارة", "subcategory": "إشارة_مكان", "location_type": "here"},
            "هناك": {"category": "إشارة", "subcategory": "إشارة_مكان", "location_type": "there"},
            "هنالك": {"category": "إشارة", "subcategory": "إشارة_مكان", "location_type": "far_there"},
            
            # Vocative Particles (نداء)
            "يا": {"category": "نداء", "subcategory": "نداء_قريب", "distance": "neutral"},
            "أيا": {"category": "نداء", "subcategory": "نداء_بعيد", "distance": "far"},
            "هيا": {"category": "نداء", "subcategory": "نداء_بعيد", "distance": "far"},
            "أي": {"category": "نداء", "subcategory": "نداء_قريب", "distance": "near"},
            "وا": {"category": "نداء", "subcategory": "نداء_تعجب", "emotional_tone": "exclamatory"},
            
            # Relative Particles (موصول)
            "الذي": {"category": "موصول", "subcategory": "موصول_عاقل", "gender": "masculine", "number": "singular"},
            "التي": {"category": "موصول", "subcategory": "موصول_عاقل", "gender": "feminine", "number": "singular"},
            "الذين": {"category": "موصول", "subcategory": "موصول_عاقل", "gender": "masculine", "number": "plural"},
            "اللذان": {"category": "موصول", "subcategory": "موصول_عاقل", "gender": "masculine", "number": "dual"},
            "اللتان": {"category": "موصول", "subcategory": "موصول_عاقل", "gender": "feminine", "number": "dual"},
            "اللواتي": {"category": "موصول", "subcategory": "موصول_عاقل", "gender": "feminine", "number": "plural"},
            "ما": {"category": "موصول", "subcategory": "موصول_غير_عاقل", "rationality": "non_rational"},
            
            # Personal Pronouns (ضمير)
            "أنا": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 1, "number": "singular"},
            "أنت": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 2, "number": "singular", "gender": "masculine"},
            "أنتِ": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 2, "number": "singular", "gender": "feminine"},
            "أنتم": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 2, "number": "plural", "gender": "masculine"},
            "أنتن": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 2, "number": "plural", "gender": "feminine"},
            "هو": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 3, "number": "singular", "gender": "masculine"},
            "هي": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 3, "number": "singular", "gender": "feminine"},
            "هم": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 3, "number": "plural", "gender": "masculine"},
            "هن": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 3, "number": "plural", "gender": "feminine"},
            "نحن": {"category": "ضمير", "subcategory": "ضمير_رفع", "person": 1, "number": "plural"},
            
            # Accusative Pronouns
            "إياي": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 1, "number": "singular"},
            "إياك": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 2, "number": "singular", "gender": "masculine"},
            "إياكِ": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 2, "number": "singular", "gender": "feminine"},
            "إياه": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 3, "number": "singular", "gender": "masculine"},
            "إياها": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 3, "number": "singular", "gender": "feminine"},
            "إيانا": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 1, "number": "plural"},
            "إياكم": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 2, "number": "plural", "gender": "masculine"},
            "إياكن": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 2, "number": "plural", "gender": "feminine"},
            "إياهم": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 3, "number": "plural", "gender": "masculine"},
            "إياهن": {"category": "ضمير", "subcategory": "ضمير_نصب", "person": 3, "number": "plural", "gender": "feminine"}
        }
    
    def classify_and_segregate_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive classification and segregation of particles in text
        
        Args:
            text: Arabic text to analyze
            
        Returns:
            Complete classification analysis
        """
        words = text.split()
        analysis = {
            "original_text": text,
            "total_words": len(words),
            "classification_summary": {},
            "segregation_by_category": defaultdict(list),
            "segregation_by_subcategory": defaultdict(list),
            "detailed_analysis": [],
            "statistics": {}
        }
        
        # Analyze each word
        for word in words:
            clean_word = word.strip('؟!،.')
            word_analysis = self.analyze_word_comprehensive(clean_word)
            
            if word_analysis["is_particle"]:
                analysis["detailed_analysis"].append(word_analysis)
                
                # Segregate by main category
                category = word_analysis["category"]
                analysis["segregation_by_category"][category].append(word_analysis)
                
                # Segregate by subcategory
                subcategory = word_analysis.get("subcategory", "غير محدد")
                analysis["segregation_by_subcategory"][subcategory].append(word_analysis)
        
        # Generate classification summary
        analysis["classification_summary"] = self.generate_classification_summary(analysis["detailed_analysis"])
        
        # Generate statistics
        analysis["statistics"] = self.generate_statistics(analysis)
        
        return analysis
    
    def analyze_word_comprehensive(self, word: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single word
        
        Args:
            word: Arabic word to analyze
            
        Returns:
            Detailed word analysis
        """
        # Basic particle analysis
        particle_result = self.particles_engine.analyze(word)
        
        # Enhanced analysis with extended data
        enhanced_analysis = {
            "word": word,
            "is_particle": particle_result['analysis_metadata'].get('is_recognized_particle', False),
            "category": particle_result['category'],
            "phonemes": particle_result['phonemes'],
            "syllabic_units": particle_result['syllabic_units'],
            "morphological_features": particle_result['morphological_features']
        }
        
        # Add extended classification if particle is recognized
        if word in self.extended_particles:
            extended_data = self.extended_particles[word]
            enhanced_analysis.update({
                "subcategory": extended_data.get("subcategory", "غير محدد"),
                "detailed_features": extended_data,
                "category_description": self.category_definitions.get(
                    enhanced_analysis["category"], {}
                ).get("description", ""),
                "function": self.category_definitions.get(
                    enhanced_analysis["category"], {}
                ).get("function", "")
            })
        
        return enhanced_analysis
    
    def generate_classification_summary(self, detailed_analysis: List[Dict]) -> Dict[str, Any]:
        """Generate summary of particle classifications"""
        summary = {
            "particles_found": len(detailed_analysis),
            "categories_present": set(),
            "subcategories_present": set(),
            "category_counts": defaultdict(int),
            "subcategory_counts": defaultdict(int)
        }
        
        for analysis in detailed_analysis:
            category = analysis["category"]
            subcategory = analysis.get("subcategory", "غير محدد")
            
            summary["categories_present"].add(category)
            summary["subcategories_present"].add(subcategory)
            summary["category_counts"][category] += 1
            summary["subcategory_counts"][subcategory] += 1
        
        # Convert sets to lists for JSON serialization
        summary["categories_present"] = list(summary["categories_present"])
        summary["subcategories_present"] = list(summary["subcategories_present"])
        summary["category_counts"] = dict(summary["category_counts"])
        summary["subcategory_counts"] = dict(summary["subcategory_counts"])
        
        return summary
    
    def generate_statistics(self, analysis: Dict) -> Dict[str, Any]:
        """Generate comprehensive statistics"""
        total_words = analysis["total_words"]
        particles_found = analysis["classification_summary"]["particles_found"]
        
        return {
            "particle_density": (particles_found / total_words * 100) if total_words > 0 else 0,
            "most_common_category": max(
                analysis["classification_summary"]["category_counts"].items(),
                key=lambda x: x[1]
            )[0] if analysis["classification_summary"]["category_counts"] else None,
            "category_diversity": len(analysis["classification_summary"]["categories_present"]),
            "subcategory_diversity": len(analysis["classification_summary"]["subcategories_present"]),
            "average_phonemes_per_particle": sum(
                len(p["phonemes"]) for p in analysis["detailed_analysis"]
            ) / particles_found if particles_found > 0 else 0
        }
    
    def get_all_categories_with_examples(self) -> Dict[str, Any]:
        """Get all categories with their particles and examples"""
        categories_data = {}
        
        for category, definition in self.category_definitions.items():
            particles_in_category = [
                particle for particle, data in self.extended_particles.items()
                if data["category"] == category
            ]
            
            # Group by subcategory
            subcategories = defaultdict(list)
            for particle in particles_in_category:
                subcategory = self.extended_particles[particle].get("subcategory", "غير محدد")
                subcategories[subcategory].append(particle)
            
            categories_data[category] = {
                "definition": definition,
                "total_particles": len(particles_in_category),
                "subcategories": dict(subcategories),
                "examples": particles_in_category[:5]  # First 5 as examples
            }
        
        return categories_data

def demonstrate_comprehensive_classification():
    """Demonstrate comprehensive classification and segregation"""
    
    print("🎯 COMPREHENSIVE ARABIC PARTICLES CLASSIFICATION & SEGREGATION")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ComprehensiveParticleAnalyzer()
    
    # Test texts with various particle types
    test_texts = [
        "هل كتب الطالب الواجب في المكتبة؟",  # Interrogative
        "إن الله غفور رحيم والذي يتوب يغفر له",  # Conditional + Relative
        "يا أحمد، هذا كتابك الذي نسيته أمس",  # Vocative + Demonstrative + Relative
        "لا تنس أن تحضر الكتاب إلا إذا كنت مريضاً",  # Negation + Conditional + Exception
        "ما عدا أولئك الذين لم يأتوا، فإن هؤلاء حضروا"  # Exception + Demonstrative + Relative + Negation
    ]
    
    print("\n📊 CATEGORY DEFINITIONS:")
    print("-" * 40)
    categories = analyzer.get_all_categories_with_examples()
    for category, data in categories.items():
        print(f"\n🏷️ {category} ({data['definition']['name']}):")
        print(f"   Description: {data['definition']['description']}")
        print(f"   Function: {data['definition']['function']}")
        print(f"   Total Particles: {data['total_particles']}")
        print(f"   Examples: {', '.join(data['examples'])}")
        
        print(f"   Subcategories:")
        for subcat, particles in data['subcategories'].items():
            print(f"      • {subcat}: {', '.join(particles[:3])}...")
    
    print("\n" + "="*70)
    print("📝 TEXT ANALYSIS EXAMPLES:")
    print("="*70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Analysis {i}: {text}")
        print("-" * 60)
        
        analysis = analyzer.classify_and_segregate_text(text)
        
        print(f"📊 Summary:")
        print(f"   Total Words: {analysis['total_words']}")
        print(f"   Particles Found: {analysis['classification_summary']['particles_found']}")
        print(f"   Particle Density: {analysis['statistics']['particle_density']:.1f}%")
        print(f"   Category Diversity: {analysis['statistics']['category_diversity']}")
        
        print(f"\n🏷️ Categories Found:")
        for category, count in analysis['classification_summary']['category_counts'].items():
            particles = [p['word'] for p in analysis['segregation_by_category'][category]]
            print(f"   {category} ({count}): {', '.join(particles)}")
        
        print(f"\n🔖 Subcategories Found:")
        for subcat, particles_data in analysis['segregation_by_subcategory'].items():
            particles = [p['word'] for p in particles_data]
            print(f"   {subcat}: {', '.join(particles)}")
        
        if analysis['statistics']['most_common_category']:
            print(f"\n📈 Most Common Category: {analysis['statistics']['most_common_category']}")

if __name__ == "__main__":
    try:
        demonstrate_comprehensive_classification()
        
        print("\n✅ Comprehensive Classification Complete!")
        print("\n💡 Available Features:")
        print("   • Complete particle categorization (8 main categories)")
        print("   • Subcategory classification (24+ subcategories)")
        print("   • Morphological feature analysis")
        print("   • Statistical analysis and diversity metrics")
        print("   • Text segregation by category/subcategory")
        print("   • Comprehensive linguistic metadata")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import_data traceback
        traceback.print_exc()
