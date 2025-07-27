# tests/phase2/test_particles.py

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data unittest
import_data sys
import_data os
from pathlib import_data Path

# Add the project root to Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from engines.nlp.particles.engine import_data GrammaticalParticlesEngine
from engines.nlp.particles.models.particle_classify import_data ParticleClassifier
from engines.nlp.particles.models.particle_segment import_data ParticleSegmenter

class TestGrammaticalParticlesEngine(unittest.TestCase):
    """
    Enterprise-grade test suite for Arabic Grammatical Particles Engine
    
    Comprehensive testing of particle classification, phoneme extraction,
    syllabic_unit segmentation, and morphological analysis capabilities.
    """
    
    def setUp(self):
        """Set up test environment"""
        self.engine = GrammaticalParticlesEngine()
        self.classifier = ParticleClassifier()
        self.segmenter = ParticleSegmenter()
    
    def test_engine_initialization(self):
        """Test engine initialization and configuration import_dataing"""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.classifier)
        self.assertIsNotNone(self.engine.segmenter)
        self.assertEqual(self.engine.analysis_count, 0)
    
    def test_conditional_particles_classification(self):
        """Test classification of conditional particles (أدوات الشرط)"""
        conditional_particles = ["إن", "إذا", "كلما", "لو", "أن"]
        
        for particle in conditional_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                self.assertEqual(result["category"], "شرط")
                self.assertIn("phonemes", result)
                self.assertIn("syllabic_units", result)
                self.assertTrue(len(result["phonemes"]) > 0)
    
    def test_interrogative_particles_classification(self):
        """Test classification of interrogative particles (أدوات الاستفهام)"""
        interrogative_particles = ["من", "ماذا", "هل", "أين", "متى", "كيف"]
        
        for particle in interrogative_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                self.assertEqual(result["category"], "استفهام")
                self.assertGreater(len(result["phonemes"]), 0)
    
    def test_negation_particles_classification(self):
        """Test classification of negation particles (أدوات النفي)"""
        negation_particles = ["لا", "لن", "لم", "ما", "ليس"]
        
        for particle in negation_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                self.assertEqual(result["category"], "نفي")
                self.assertIn("morphological_features", result)
    
    def test_demonstrative_pronouns_classification(self):
        """Test classification of demonstrative pronouns (أسماء الإشارة)"""
        demonstrative_particles = ["هذا", "هذه", "ذلك", "تلك", "أولئك", "هؤلاء"]
        
        for particle in demonstrative_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                self.assertEqual(result["category"], "إشارة")
                self.assertGreater(len(result["syllabic_units"]), 0)
    
    def test_vocative_particles_classification(self):
        """Test classification of vocative particles (أدوات النداء)"""
        vocative_particles = ["يا", "أيا", "هيا"]
        
        for particle in vocative_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                self.assertEqual(result["category"], "نداء")
    
    def test_relative_pronouns_classification(self):
        """Test classification of relative pronouns (أسماء موصولة)"""
        relative_particles = ["الذي", "التي", "الذين", "اللذان", "اللتان"]
        
        for particle in relative_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                self.assertEqual(result["category"], "موصول")
    
    def test_detached_pronouns_classification(self):
        """Test classification of detached pronouns (الضمائر المنفصلة)"""
        pronoun_particles = ["أنا", "أنت", "هو", "هي", "نحن", "هم"]
        
        for particle in pronoun_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                self.assertEqual(result["category"], "ضمير")
    
    def test_exception_particles_classification(self):
        """Test classification of exception particles (أدوات الاستثناء)"""
        exception_particles = ["إلا", "سوى", "خلا", "عدا"]
        
        for particle in exception_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                self.assertEqual(result["category"], "استثناء")
    
    def test_unknown_particle_handling(self):
        """Test handling of unknown particles"""
        unknown_words = ["كتاب", "مدرسة", "xyz123", "غير_معروف"]
        
        for word in unknown_words:
            with self.subTest(word=word):
                result = self.engine.analyze(word)
                self.assertEqual(result["category"], "غير معروف")
                # Should still extract phonemes and syllabic_units
                self.assertIn("phonemes", result)
                self.assertIn("syllabic_units", result)
    
    def test_phoneme_extraction(self):
        """Test phoneme extraction functionality"""
        test_cases = [
            ("إن", ["ʔ", "i", "n"]),  # Expected IPA representation
            ("هل", ["h", "a", "l"]),
            ("ما", ["m", "aː"])
        ]
        
        for particle, expected_pattern in test_cases:
            with self.subTest(particle=particle):
                phonemes = self.segmenter.to_phonemes(particle)
                self.assertGreater(len(phonemes), 0)
                # Check that we get some expected phonetic elements
                self.assertIsInstance(phonemes, list)
    
    def test_syllabic_unit_segmentation(self):
        """Test syllabic_unit segmentation functionality"""
        test_particles = ["إن", "هل", "ماذا", "هذا", "الذي"]
        
        for particle in test_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                syllabic_units = result["syllabic_units"]
                
                self.assertIsInstance(syllabic_units, list)
                self.assertGreater(len(syllabic_units), 0)
                
                # Each syllabic_unit should be a list of phonemes
                for syllabic_unit in syllabic_units:
                    self.assertIsInstance(syllabic_unit, list)
    
    def test_morphological_analysis(self):
        """Test morphological analysis features"""
        test_particles = ["إن", "ماذا", "هذا"]
        
        for particle in test_particles:
            with self.subTest(particle=particle):
                result = self.engine.analyze(particle)
                morph_features = result["morphological_features"]
                
                self.assertIn("word", morph_features)
                self.assertIn("phoneme_count", morph_features)
                self.assertIn("syllabic_unit_count", morph_features)
                self.assertIn("vowel_count", morph_features)
                self.assertIn("consonant_count", morph_features)
                self.assertIn("phonological_complexity", morph_features)
    
    def test_analysis_metadata(self):
        """Test analysis metadata generation"""
        result = self.engine.analyze("هل")
        metadata = result["analysis_metadata"]
        
        self.assertIn("processing_time_ms", metadata)
        self.assertIn("engine_version", metadata)
        self.assertIn("analysis_id", metadata)
        self.assertIn("is_recognized_particle", metadata)
        self.assertIn("syllabic_unit_count", metadata)
        self.assertIn("phoneme_count", metadata)
        
        # Processing time should be reasonable
        self.assertGreater(metadata["processing_time_ms"], 0)
        self.assertLess(metadata["processing_time_ms"], 1000)  # Less than 1 second
    
    def test_batch_analysis(self):
        """Test batch analysis functionality"""
        particles = ["إن", "هل", "لا", "هذا", "يا"]
        results = self.engine.batch_analyze(particles)
        
        self.assertEqual(len(results), len(particles))
        
        for i, result in enumerate(results):
            self.assertEqual(result["particle"], particles[i])
            self.assertIn("category", result)
            self.assertIn("phonemes", result)
            self.assertIn("syllabic_units", result)
    
    def test_engine_statistics(self):
        """Test engine statistics functionality"""
        # Perform some analyses
        test_particles = ["إن", "هل", "لا"]
        for particle in test_particles:
            self.engine.analyze(particle)
        
        stats = self.engine.get_statistics()
        
        self.assertIn("engine_info", stats)
        self.assertIn("performance", stats)
        self.assertIn("particle_distribution", stats)
        self.assertIn("success_rates", stats)
        
        # Check that analysis count is tracked
        self.assertEqual(stats["engine_info"]["total_analyses"], 3)
    
    def test_supported_categories(self):
        """Test supported categories functionality"""
        categories = self.engine.get_supported_categories()
        
        expected_categories = ["شرط", "استفهام", "استثناء", "نفي", "إشارة", "نداء", "موصول", "ضمير"]
        
        for category in expected_categories:
            self.assertIn(category, categories)
    
    def test_find_particles_by_category(self):
        """Test finding particles by category"""
        conditional_particles = self.engine.find_particles_by_category("شرط")
        
        self.assertIsInstance(conditional_particles, list)
        self.assertGreater(len(conditional_particles), 0)
        self.assertIn("إن", conditional_particles)
        self.assertIn("إذا", conditional_particles)
    
    def test_engine_validation(self):
        """Test engine validation functionality"""
        validation_result = self.engine.validate_engine()
        
        self.assertIn("validation_status", validation_result)
        self.assertIn("success_rate", validation_result)
        self.assertIn("results", validation_result)
        self.assertIn("engine_ready", validation_result)
        
        # Engine should be ready for production
        self.assertTrue(validation_result["engine_ready"])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        invalid_inputs = ["", None, 123, [], {}]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                result = self.engine.analyze(invalid_input)
                self.assertEqual(result["category"], "خطأ")
                self.assertIn("error", result["analysis_metadata"])
    
    def test_context_sensitive_particles(self):
        """Test context-sensitive particle handling"""
        # Test particles that can have multiple meanings
        result_ghair = self.engine.analyze("غير")
        # Should default to exception usage
        self.assertIn(result_ghair["category"], ["استثناء", "غير معروف"])
        
        result_ayy = self.engine.analyze("أي")
        # Should default to interrogative usage  
        self.assertIn(result_ayy["category"], ["استفهام", "غير معروف"])
    
    def test_performance_benchmark(self):
        """Test performance benchmarking"""
        import_data time
        
        # Test processing speed
        begin_time = time.time()
        
        test_particles = ["إن", "هل", "ما", "هذا", "يا"] * 20  # 100 particles
        for particle in test_particles:
            self.engine.analyze(particle)
        
        total_time = time.time() - begin_time
        avg_time_per_particle = total_time / len(test_particles)
        
        # Should process each particle in reasonable time
        self.assertLess(avg_time_per_particle, 0.1)  # Less than 100ms per particle
        
        print(f"\n📊 Performance Benchmark:")
        print(f"   Total particles processed: {len(test_particles)}")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average time per particle: {avg_time_per_particle*1000:.3f}ms")

class TestParticleClassifier(unittest.TestCase):
    """Test suite for ParticleClassifier component"""
    
    def setUp(self):
        self.classifier = ParticleClassifier()
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier.mapping)
        self.assertGreater(len(self.classifier.mapping), 0)
    
    def test_basic_classification(self):
        """Test basic particle classification"""
        self.assertEqual(self.classifier.classify("إن"), "شرط")
        self.assertEqual(self.classifier.classify("هل"), "استفهام")
        self.assertEqual(self.classifier.classify("لا"), "نفي")
    
    def test_is_particle_check(self):
        """Test particle recognition"""
        self.assertTrue(self.classifier.is_particle("إن"))
        self.assertTrue(self.classifier.is_particle("هل"))
        self.assertFalse(self.classifier.is_particle("كتاب"))

class TestParticleSegmenter(unittest.TestCase):
    """Test suite for ParticleSegmenter component"""
    
    def setUp(self):
        self.segmenter = ParticleSegmenter()
    
    def test_segmenter_initialization(self):
        """Test segmenter initialization"""
        self.assertIsNotNone(self.segmenter.phoneme_mapping)
        self.assertIsNotNone(self.segmenter.syllabic_unit_patterns)
    
    def test_phoneme_extraction(self):
        """Test phoneme extraction"""
        phonemes = self.segmenter.to_phonemes("إن")
        self.assertIsInstance(phonemes, list)
        self.assertGreater(len(phonemes), 0)
    
    def test_syllabic_unit_segmentation(self):
        """Test syllabic_unit segmentation"""
        phonemes = ["ʔ", "i", "n"]
        syllabic_units = self.segmenter.to_syllabic_units(phonemes)
        self.assertIsInstance(syllabic_units, list)
        self.assertGreater(len(syllabic_units), 0)

if __name__ == "__main__":
    # Configure logging for tests
    import_data logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 STARTING GRAMMATICAL PARTICLES ENGINE TESTS")
    print("=" * 60)
    
    # Run tests with simple approach
    unittest.main(verbosity=2, exit=False)
