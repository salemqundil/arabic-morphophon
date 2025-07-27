# Simple test script for GrammaticalParticlesEngine

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data sys
import_data os
from pathlib import_data Path

# Add the project root to Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from engines.nlp.particles.engine import_data GrammaticalParticlesEngine

def test_particles_engine():
    """Simple test function for particles engine"""
    print("🧪 TESTING GRAMMATICAL PARTICLES ENGINE")
    print("=" * 50)
    
    try:
        # Initialize engine
        print("1. Initializing engine...")
        engine = GrammaticalParticlesEngine()
        print("✅ Engine initialized successfully")
        
        # Test basic functionality
        test_particles = [
            ("إن", "شرط"),
            ("هل", "استفهام"), 
            ("لا", "نفي"),
            ("هذا", "إشارة"),
            ("يا", "نداء"),
            ("الذي", "موصول"),
            ("أنا", "ضمير"),
            ("إلا", "استثناء")
        ]
        
        print("\n2. Testing particle classification...")
        success_count = 0
        
        for particle, expected_category in test_particles:
            try:
                result = engine.analyze(particle)
                actual_category = result["category"]
                
                if actual_category == expected_category:
                    print(f"✅ {particle} → {actual_category}")
                    success_count += 1
                else:
                    print(f"❌ {particle} → {actual_category} (expected: {expected_category})")
                    
                # Check that we have phonemes and syllabic_units
                if len(result["phonemes"]) > 0 and len(result["syllabic_units"]) > 0:
                    print(f"   📞 Phonemes: {result['phonemes']}")
                    print(f"   📝 SyllabicUnits: {result['syllabic_units']}")
                
            except Exception as e:
                print(f"❌ Error analyzing {particle}: {e}")
        
        print(f"\n3. Testing unknown particles...")
        unknown_result = engine.analyze("كتاب")
        if unknown_result["category"] == "غير معروف":
            print("✅ Unknown particle handling works")
            success_count += 1
        else:
            print("❌ Unknown particle handling failed")
        
        print(f"\n4. Testing batch analysis...")
        batch_results = engine.batch_analyze(["إن", "هل", "لا"])
        if len(batch_results) == 3:
            print("✅ Batch analysis works")
            success_count += 1
        else:
            print("❌ Batch analysis failed")
        
        print(f"\n5. Testing engine statistics...")
        stats = engine.get_statistics()
        if "engine_info" in stats and "performance" in stats:
            print("✅ Statistics generation works")
            print(f"   📊 Total analyses: {stats['engine_info']['total_analyses']}")
            success_count += 1
        else:
            print("❌ Statistics generation failed")
        
        print(f"\n6. Testing engine validation...")
        validation = engine.validate_engine()
        if validation["engine_ready"]:
            print("✅ Engine validation passed")
            success_count += 1
        else:
            print("❌ Engine validation failed")
        
        # Final results
        total_tests = len(test_particles) + 4  # +4 for additional tests
        print("\n" + "=" * 50)
        print(f"📊 TEST RESULTS:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   📝 Total: {total_tests}")
        print(f"   📈 Success Rate: {(success_count/total_tests)*100:.1f}%")
        
        if success_count >= total_tests - 1:  # Allow 1 failure
            print("\n🎉 GRAMMATICAL PARTICLES ENGINE - READY FOR PRODUCTION!")
            return True
        else:
            print("\n❌ SOME TESTS FAILED - NEEDS ATTENTION")
            return False
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False

if __name__ == "__main__":
    test_particles_engine()
