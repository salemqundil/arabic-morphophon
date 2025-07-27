#!/usr/bin/env python3
"""
🎯 Final Demo: Complete Arabic Particles Classification & Segregation
Showcases all 8 categories with comprehensive analysis capabilities
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data sys
from pathlib import_data Path
from typing import_data Dict, List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from comprehensive_particle_classification import_data ComprehensiveParticleAnalyzer

def create_category_demonstration():
    """Create demonstration text covering all 8 particle categories"""
    
    # Texts designed to showcase each category
    category_examples = {
        "شرط": [
            "إن درست جيداً فستنجح",  # Conditional particle
            "إذا جاء الأستاذ سنبدأ الدرس",
            "كلما قرأت تعلمت شيئاً جديداً"
        ],
        "استفهام": [
            "هل فهمت الدرس؟",  # Yes/No question
            "من جاء إلى المدرسة اليوم؟",  # Who question
            "ماذا تريد أن تدرس؟",  # What question
            "أين ذهب الطلاب؟",  # Where question
            "متى سيبدأ الامتحان؟",  # When question
            "كيف تحل هذه المسألة؟"  # How question
        ],
        "استثناء": [
            "جاء كل الطلاب إلا أحمد",  # Exception
            "لا أحب شيئاً غير القراءة",
            "كل شيء جميل سوى هذا"
        ],
        "نفي": [
            "لا تتكلم في الصف",  # Negation
            "لن أنسى هذا الدرس أبداً",  # Future negation
            "لم يحضر الطالب أمس",  # Past negation
            "ليس الطقس جميلاً اليوم"  # Present negation
        ],
        "إشارة": [
            "هذا كتاب مفيد جداً",  # Near demonstrative (masculine)
            "هذه مدرسة ممتازة",  # Near demonstrative (feminine)
            "ذلك البيت بعيد عنا",  # Far demonstrative (masculine)
            "تلك السيارة سريعة",  # Far demonstrative (feminine)
            "هؤلاء الطلاب مجتهدون",  # Near plural
            "أولئك الأساتذة خبراء",  # Far plural
            "هنا مكان جميل للدراسة",  # Here
            "هناك مكتبة كبيرة"  # There
        ],
        "نداء": [
            "يا أحمد تعال هنا",  # Common vocative
            "أي طالب اجتهد في دروسه",  # Near vocative
            "أيا من سمع فليجب",  # Far vocative
            "وا حسرتاه على الوقت الضائع"  # Exclamatory vocative
        ],
        "موصول": [
            "الطالب الذي درس نجح",  # Masculine rational
            "الطالبة التي اجتهدت تفوقت",  # Feminine rational
            "الطلاب الذين حضروا فهموا",  # Masculine plural rational
            "الطالبات اللواتي شاركن أبدعن",  # Feminine plural rational
            "الكتاب الذي قرأته مفيد",  # Thing/object
            "ما تعلمته اليوم مهم"  # Non-rational relative
        ],
        "ضمير": [
            "أنا أحب العلم والمعرفة",  # First person singular
            "أنت طالب مجتهد ونشيط",  # Second person singular masculine
            "أنتِ طالبة متفوقة ومبدعة",  # Second person singular feminine
            "هو أستاذ خبير ومتمكن",  # Third person singular masculine
            "هي معلمة متميزة ومبدعة",  # Third person singular feminine
            "نحن طلاب في هذه المدرسة",  # First person plural
            "أنتم تدرسون بجد واجتهاد",  # Second person plural masculine
            "هم يحبون القراءة والكتابة",  # Third person plural masculine
            "هن معلمات في المدرسة"  # Third person plural feminine
        ]
    }
    
    return category_examples

def demonstrate_complete_system():
    """Demonstrate complete particle classification and segregation system"""
    
    print("🎯 COMPLETE ARABIC PARTICLES CLASSIFICATION & SEGREGATION SYSTEM")
    print("=" * 80)
    print("📚 Comprehensive Analysis of All 8 Grammatical Categories")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ComprehensiveParticleAnalyzer()
    
    # Get category examples
    category_examples = create_category_demonstration()
    
    print("\n📋 CATEGORY-BY-CATEGORY ANALYSIS:")
    print("-" * 50)
    
    total_particles_found = 0
    total_texts_analyzed = 0
    category_statistics = {}
    
    for category, texts in category_examples.items():
        print(f"\n🏷️ CATEGORY: {category}")
        print(f"   ({analyzer.category_definitions[category]['name']})")
        print(f"   {analyzer.category_definitions[category]['description']}")
        print("-" * 40)
        
        category_particles = []
        category_subcategories = set()
        
        for i, text in enumerate(texts, 1):
            analysis = analyzer.classify_and_segregate_text(text)
            total_texts_analyzed += 1
            
            particles_in_text = analysis['classification_summary']['particles_found']
            total_particles_found += particles_in_text
            
            print(f"\n   Text {i}: {text}")
            print(f"   📊 Particles: {particles_in_text} | Density: {analysis['statistics']['particle_density']:.1f}%")
            
            if particles_in_text > 0:
                text_particles = []
                for particle_data in analysis['detailed_analysis']:
                    if particle_data['category'] == category:
                        text_particles.append(particle_data['word'])
                        category_particles.append(particle_data['word'])
                        if 'subcategory' in particle_data:
                            category_subcategories.add(particle_data['subcategory'])
                
                if text_particles:
                    print(f"   🎯 Found: {', '.join(text_particles)}")
        
        # Category summary
        unique_particles = list(set(category_particles))
        category_statistics[category] = {
            'total_occurrences': len(category_particles),
            'unique_particles': len(unique_particles),
            'subcategories': len(category_subcategories),
            'particles_list': unique_particles,
            'subcategories_list': list(category_subcategories)
        }
        
        print(f"\n   📈 Category Summary:")
        print(f"      Total Occurrences: {len(category_particles)}")
        print(f"      Unique Particles: {len(unique_particles)} ({', '.join(unique_particles)})")
        print(f"      Subcategories: {len(category_subcategories)}")
        if category_subcategories:
            print(f"      Types: {', '.join(category_subcategories)}")
    
    # Comprehensive analysis of a complex sentence
    print(f"\n{'='*80}")
    print("🔍 COMPREHENSIVE SENTENCE ANALYSIS")
    print("="*80)
    
    complex_sentence = ("هل تعرف يا صديقي الطالب الذي إن لم يدرس فلن ينجح إلا إذا "
                       "اجتهد، وهذا ما أخبرتك إياه أنا أمس هناك في المكتبة؟")
    
    print(f"Complex Sentence:")
    print(f"{complex_sentence}")
    print("-" * 60)
    
    comprehensive_analysis = analyzer.classify_and_segregate_text(complex_sentence)
    
    print(f"📊 Overall Analysis:")
    print(f"   Total Words: {comprehensive_analysis['total_words']}")
    print(f"   Particles Found: {comprehensive_analysis['classification_summary']['particles_found']}")
    print(f"   Particle Density: {comprehensive_analysis['statistics']['particle_density']:.1f}%")
    print(f"   Categories Present: {len(comprehensive_analysis['classification_summary']['categories_present'])}")
    print(f"   Subcategories Present: {len(comprehensive_analysis['classification_summary']['subcategories_present'])}")
    
    print(f"\n🏷️ Complete Category Breakdown:")
    for category, count in comprehensive_analysis['classification_summary']['category_counts'].items():
        particles = [p['word'] for p in comprehensive_analysis['segregation_by_category'][category]]
        category_name = analyzer.category_definitions[category]['name']
        print(f"   {category} ({category_name}) - {count} particles: {', '.join(particles)}")
    
    print(f"\n🔖 Complete Subcategory Breakdown:")
    for subcat, particles_data in comprehensive_analysis['segregation_by_subcategory'].items():
        particles = [p['word'] for p in particles_data]
        print(f"   {subcat}: {', '.join(particles)}")
    
    print(f"\n📈 Detailed Particle Analysis:")
    for particle_data in comprehensive_analysis['detailed_analysis']:
        word = particle_data['word']
        category = particle_data['category']
        subcategory = particle_data.get('subcategory', 'غير محدد')
        phonemes = len(particle_data['phonemes'])
        syllabic_units = len(particle_data['syllabic_units'])
        
        print(f"   '{word}' → {category} ({subcategory}) | {phonemes} phonemes, {syllabic_units} syllabic_units")
    
    # System statistics
    print(f"\n{'='*80}")
    print("📊 COMPLETE SYSTEM STATISTICS")
    print("="*80)
    
    print(f"🎯 Analysis Summary:")
    print(f"   Total Texts Analyzed: {total_texts_analyzed}")
    print(f"   Total Particles Found: {total_particles_found}")
    print(f"   Average Particles per Text: {total_particles_found/total_texts_analyzed:.1f}")
    
    print(f"\n🏷️ Category Distribution:")
    for category, stats in category_statistics.items():
        percentage = (stats['total_occurrences'] / total_particles_found * 100) if total_particles_found > 0 else 0
        print(f"   {category}: {stats['total_occurrences']} occurrences ({percentage:.1f}%) | {stats['unique_particles']} unique")
    
    print(f"\n📋 System Capabilities:")
    all_categories = analyzer.get_all_categories_with_examples()
    total_system_particles = sum(data['total_particles'] for data in all_categories.values())
    total_subcategories = sum(len(data['subcategories']) for data in all_categories.values())
    
    print(f"   ✅ Categories Supported: {len(all_categories)}")
    print(f"   ✅ Total Particles in Database: {total_system_particles}")
    print(f"   ✅ Total Subcategories: {total_subcategories}")
    print(f"   ✅ Morphological Analysis: Phonemes + SyllabicUnits")
    print(f"   ✅ Statistical Metrics: Density, Distribution, Diversity")
    print(f"   ✅ Segregation: By Category and Subcategory")
    print(f"   ✅ Context-Sensitive Classification")
    print(f"   ✅ Real-time Processing")
    
    print(f"\n💡 Classification Categories:")
    for category, data in all_categories.items():
        print(f"   🏷️ {category} ({data['definition']['name']}):")
        print(f"      Function: {data['definition']['function']}")
        print(f"      Particles: {data['total_particles']} | Subcategories: {len(data['subcategories'])}")
    
    print(f"\n✅ COMPLETE CLASSIFICATION & SEGREGATION SYSTEM DEMONSTRATION FINISHED!")
    print("🎯 All 8 Arabic grammatical particle categories successfully analyzed!")

if __name__ == "__main__":
    try:
        demonstrate_complete_system()
    except Exception as e:
        print(f"❌ Error: {e}")
        import_data traceback
        traceback.print_exc()
