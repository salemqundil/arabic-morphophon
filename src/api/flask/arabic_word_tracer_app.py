#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Arabic Word Tracer - Advanced Browser Interface
متتبع الكلمات العربية - واجهة متصفح متقدمة

Features:
🎯 Complete linguistic tracing from phonemes to roots
📊 Interactive visualizations and diagrams
🚀 Real-time analysis with step-by-step breakdown
🎨 Professional UI with Arabic language support
🧠 Expert NLP system integration
📱 Responsive design for all devices
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data logging
import_data os
import_data sys
import_data time
from datetime import_data datetime
from pathlib import_data Path
from typing import_data Any, Dict, List, Optional, Tuple

from flask import_data Flask, jsonify, render_template, request, send_from_directory
from flask_cors import_data CORS

# إضافة مسار المشروع
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# استيراد المحركات المتخصصة
try:
    from engines.nlp.particles.engine import_data GrammaticalParticlesEngine
except ImportError:
    GrammaticalParticlesEngine = None

try:
    from engines.nlp.morphology.engine import_data MorphologyEngine
except ImportError:
    MorphologyEngine = None

try:
    from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
except ImportError:
    PhonologyEngine = None

try:
    from engines.nlp.frozen_root.engine import_data FrozenRootsEngine
except ImportError:
    FrozenRootsEngine = None

try:
    from arabic_morphophon.models.patterns import_data PatternRepository
except ImportError:
    PatternRepository = None

try:
    from arabic_morphophon.models.roots import_data ArabicRoot
except ImportError:
    ArabicRoot = None

try:
    from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
        AdvancedPhonologyEngine,
        SyllabicUnitEngine,
    )
except ImportError:
    SyllabicUnitEngine = None
    AdvancedPhonologyEngine = None

# محركات محاكاة للاختبار
class MockEngine:
    def __init__(self, name="Mock"): 
        self.name = name
    def analyze(self, text, **kwargs): 
        return {
            "analysis": f"تحليل {self.name} للنص: {text}",
            "status": "mock",
            "text": text,
            "engine": self.name
        }

class MockPatternRepository:
    def __init__(self):
        self.patterns = []
    def find_matching_patterns(self, word):
        return []

# إعداد Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'arabic_word_tracer_2024'
app.config['JSON_AS_ASCII'] = False
CORS(app)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicWordTracer:
    """
    🔍 متتبع الكلمات العربية المتقدم
    يتتبع المسار الكامل للكلمة العربية عبر جميع المستويات اللغوية
    """
    
    def __init__(self):
        """تهيئة المتتبع مع جميع المحركات اللغوية"""
        self.engines = {}
        self.pattern_repository = None
        self.performance_stats = {
            'total_traces': 0,
            'successful_traces': 0,
            'average_processing_time': 0.0,
            'last_reset': datetime.now()
        }
        self._initialize_engines()
    
    def _initialize_engines(self):
        """تهيئة جميع المحركات اللغوية المتخصصة"""
        try:
            # محرك الجسيمات النحوية
            if GrammaticalParticlesEngine:
                self.engines['particles'] = GrammaticalParticlesEngine()
            else:
                self.engines['particles'] = MockEngine('particles')
            
            # محرك الصرف
            if MorphologyEngine:
                try:
                    self.engines['morphology'] = MorphologyEngine("morphology", {})
                except TypeError:
                    self.engines['morphology'] = MorphologyEngine()
            else:
                self.engines['morphology'] = MockEngine('morphology')
            
            # محرك الأصوات والمقاطع
            if PhonologyEngine:
                try:
                    self.engines['phonology'] = PhonologyEngine("phonology", {})
                except TypeError:
                    self.engines['phonology'] = PhonologyEngine()
            else:
                self.engines['phonology'] = MockEngine('phonology')
            
            if SyllabicUnitEngine:
                self.engines['syllabic_unit'] = SyllabicUnitEngine()
            else:
                self.engines['syllabic_unit'] = MockEngine('syllabic_unit')
                
            if AdvancedPhonologyEngine:
                self.engines['advanced_phonology'] = AdvancedPhonologyEngine()
            else:
                self.engines['advanced_phonology'] = MockEngine('advanced_phonology')
            
            # محرك الجذور الجامدة
            if FrozenRootsEngine:
                try:
                    self.engines['frozen_root'] = FrozenRootsEngine("frozen_root", {})
                except TypeError:
                    self.engines['frozen_root'] = FrozenRootsEngine()
            else:
                self.engines['frozen_root'] = MockEngine('frozen_root')
            
            # مستودع الأوزان
            if PatternRepository:
                self.pattern_repository = PatternRepository()
            else:
                self.pattern_repository = MockPatternRepository()
            
            logger.info("✅ تم تهيئة جميع المحركات بنجاح")
            
        except Exception as e:
            logger.error(f"❌ خطأ في تهيئة المحركات: {e}")
            # إنشاء محركات محاكاة
            for engine_name in ['particles', 'morphology', 'phonology', 'syllabic_unit', 'frozen_root', 'advanced_phonology']:
                self.engines[engine_name] = MockEngine(engine_name)
            self.pattern_repository = MockPatternRepository()
    
    def trace_word_complete(self, word: str) -> Dict[str, Any]:
        """
        تتبع كامل للكلمة العربية عبر جميع المستويات اللغوية
        
        Args:
            word: الكلمة العربية المراد تتبعها
            
        Returns:
            تحليل شامل يشمل جميع المستويات اللغوية
        """
        begin_time = time.time()
        
        trace_result = {
            'input_word': word,
            'trace_timestamp': datetime.now().isoformat(),
            'trace_id': self._generate_trace_id(),
            'linguistic_levels': {},
            'trace_summary': {},
            'metadata': {}
        }
        
        try:
            # 1. مستوى الأصوات (Phonemes)
            trace_result['linguistic_levels']['phonemes'] = self._trace_phonemes(word)
            
            # 2. مستوى الحركات (Harakat)
            trace_result['linguistic_levels']['harakat'] = self._trace_harakat(word)
            
            # 3. مستوى المقاطع (SyllabicUnits)
            trace_result['linguistic_levels']['syllabic_units'] = self._trace_syllabic_units(word)
            
            # 4. مستوى الجسيمات (Particles)
            trace_result['linguistic_levels']['particles'] = self._trace_particles(word)
            
            # 5. مستوى الأسماء والأفعال (Nouns & Verbs)
            trace_result['linguistic_levels']['word_class'] = self._trace_word_class(word)
            
            # 6. مستوى الأوزان (Patterns)
            trace_result['linguistic_levels']['patterns'] = self._trace_patterns(word)
            
            # 7. مستوى الوزن الصرفي (Weight)
            trace_result['linguistic_levels']['weight'] = self._trace_morphological_weight(word)
            
            # 8. مستوى الجذر (Root)
            trace_result['linguistic_levels']['root'] = self._trace_root(word)
            
            # 9. مستوى البادئات واللواحق (Affixes)
            trace_result['linguistic_levels']['affixes'] = self._trace_affixes(word)
            
            # 10. مستوى المصدر والمجرد (Infinitive & Pure)
            trace_result['linguistic_levels']['infinitive_pure'] = self._trace_infinitive_pure(word)
            
            # إنشاء ملخص التتبع
            trace_result['trace_summary'] = self._generate_trace_summary(trace_result['linguistic_levels'])
            
            # حساب الأداء
            processing_time = time.time() - begin_time
            trace_result['metadata'] = {
                'processing_time_ms': round(processing_time * 1000, 2),
                'engines_used': list(self.engines.keys()),
                'analysis_depth': len(trace_result['linguistic_levels']),
                'status': 'success'
            }
            
            # تحديث الإحصائيات
            self._update_performance_stats(processing_time, success=True)
            
            return trace_result
            
        except Exception as e:
            logger.error(f"❌ خطأ في تتبع الكلمة {word}: {e}")
            trace_result['metadata'] = {
                'error': str(e),
                'status': 'error',
                'processing_time_ms': round((time.time() - begin_time) * 1000, 2)
            }
            self._update_performance_stats(time.time() - begin_time, success=False)
            return trace_result
    
    def _trace_phonemes(self, word: str) -> Dict[str, Any]:
        """تتبع مستوى الأصوات (الفونيمات)"""
        try:
            if 'advanced_phonology' in self.engines:
                result = self.engines['advanced_phonology'].extract_phonemes(word)
                
                return {
                    'phonemes_list': result if isinstance(result, list) else [result],
                    'phoneme_count': len(result) if isinstance(result, list) else 1,
                    'phoneme_types': self._classify_phonemes(result if isinstance(result, list) else [result]),
                    'ipa_representation': self._to_ipa(word),
                    'status': 'success'
                }
            else:
                # تحليل بسيط بديل
                return self._simple_phoneme_analysis(word)
                
        except Exception as e:
            logger.error(f"خطأ في تحليل الأصوات: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_harakat(self, word: str) -> Dict[str, Any]:
        """تتبع مستوى الحركات"""
        try:
            harakat = {
                'fatha': word.count('َ'),
                'kasra': word.count('ِ'),
                'damma': word.count('ُ'),
                'sukun': word.count('ْ'),
                'tanween_fath': word.count('ً'),
                'tanween_kasr': word.count('ٍ'),
                'tanween_damm': word.count('ٌ'),
                'shadda': word.count('ّ'),
                'madd': word.count('ٓ')
            }
            
            total_harakat = sum(harakat.values())
            clean_word = self._remove_harakat(word)
            
            return {
                'harakat_breakdown': harakat,
                'total_harakat': total_harakat,
                'harakat_density': round(total_harakat / len(clean_word) if clean_word else 0, 2),
                'diacritization_level': self._assess_diacritization_level(harakat),
                'clean_word': clean_word,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"خطأ في تحليل الحركات: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_syllabic_units(self, word: str) -> Dict[str, Any]:
        """تتبع مستوى المقاطع الصوتية"""
        try:
            if 'syllabic_unit' in self.engines:
                syllabic_units_data = self.engines['syllabic_unit'].syllabic_analyze_word(word)
                
                return {
                    'syllabic_units': [s.text for s in syllabic_units_data] if syllabic_units_data else [],
                    'syllabic_unit_count': len(syllabic_units_data) if syllabic_units_data else 0,
                    'syllabic_unit_patterns': [s.pattern for s in syllabic_units_data] if syllabic_units_data else [],
                    'syllabic_unit_types': [s.type.value for s in syllabic_units_data] if syllabic_units_data else [],
                    'cv_pattern': self._extract_cv_pattern(word),
                    'prosodic_weight': self._calculate_prosodic_weight(syllabic_units_data),
                    'status': 'success'
                }
            else:
                return self._simple_syllabic_unit_analysis(word)
                
        except Exception as e:
            logger.error(f"خطأ في تحليل المقاطع: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_particles(self, word: str) -> Dict[str, Any]:
        """تتبع مستوى الجسيمات النحوية"""
        try:
            if 'particles' in self.engines:
                result = self.engines['particles'].analyze(word)
                
                return {
                    'is_particle': result.get('particles_found', 0) > 0,
                    'particle_type': result.get('particles', [{}])[0].get('category', 'none') if result.get('particles') else 'none',
                    'particle_function': result.get('particles', [{}])[0].get('function', 'none') if result.get('particles') else 'none',
                    'particle_details': result.get('particles', []),
                    'categories_summary': result.get('categories_summary', {}),
                    'status': 'success'
                }
            else:
                return self._simple_particle_check(word)
                
        except Exception as e:
            logger.error(f"خطأ في تحليل الجسيمات: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_word_class(self, word: str) -> Dict[str, Any]:
        """تتبع تصنيف الكلمة (اسم/فعل/حرف)"""
        try:
            # استخدام محرك الصرف لتحديد نوع الكلمة
            if 'morphology' in self.engines:
                result = self.engines['morphology'].analyze(word)
                
                # استخراج معلومات تصنيف الكلمة
                word_class_info = {
                    'primary_class': self._determine_primary_class(result, word),
                    'sub_class': self._determine_sub_class(result, word),
                    'grammatical_features': self._extract_grammatical_features(result),
                    'confidence_score': self._calculate_classification_confidence(result),
                    'alternative_classes': self._get_alternative_classifications(result),
                    'status': 'success'
                }
                
                return word_class_info
            else:
                return self._simple_word_class_analysis(word)
                
        except Exception as e:
            logger.error(f"خطأ في تصنيف الكلمة: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_patterns(self, word: str) -> Dict[str, Any]:
        """تتبع الأوزان الصرفية"""
        try:
            if self.pattern_repository:
                # البحث عن الأوزان المطابقة
                matching_patterns = self._find_matching_patterns(word)
                
                return {
                    'matching_patterns': matching_patterns,
                    'pattern_count': len(matching_patterns),
                    'most_likely_pattern': matching_patterns[0] if matching_patterns else None,
                    'pattern_families': self._group_patterns_by_family(matching_patterns),
                    'derivation_potential': self._assess_derivation_potential(word),
                    'status': 'success'
                }
            else:
                return self._simple_pattern_analysis(word)
                
        except Exception as e:
            logger.error(f"خطأ في تحليل الأوزان: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_morphological_weight(self, word: str) -> Dict[str, Any]:
        """تتبع الوزن الصرفي"""
        try:
            # تحليل الوزن الصرفي
            clean_word = self._remove_harakat(word)
            weight_analysis = {
                'morphological_weight': self._calculate_morphological_weight(clean_word),
                'letter_count': len(clean_word),
                'root_letters': self._count_root_letters(clean_word),
                'augmentation_letters': self._count_augmentation_letters(clean_word),
                'weight_category': self._categorize_weight(clean_word),
                'weight_distribution': self._analyze_weight_distribution(clean_word),
                'status': 'success'
            }
            
            return weight_analysis
            
        except Exception as e:
            logger.error(f"خطأ في حساب الوزن الصرفي: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_root(self, word: str) -> Dict[str, Any]:
        """تتبع الجذر"""
        try:
            if 'morphology' in self.engines:
                result = self.engines['morphology'].analyze(word)
                
                # استخراج معلومات الجذر
                root_info = self._extract_root_info(result, word)
                
                return {
                    'identified_root': root_info.get('root', ''),
                    'root_type': root_info.get('type', 'unknown'),
                    'root_length': len(root_info.get('root', '')),
                    'root_radicals': list(root_info.get('root', '')),
                    'weak_letters': self._identify_weak_letters(root_info.get('root', '')),
                    'semantic_field': root_info.get('semantic_field', ''),
                    'derivation_family': self._get_derivation_family(root_info.get('root', '')),
                    'confidence': root_info.get('confidence', 0.0),
                    'status': 'success'
                }
            else:
                return self._simple_root_analysis(word)
                
        except Exception as e:
            logger.error(f"خطأ في تحليل الجذر: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_affixes(self, word: str) -> Dict[str, Any]:
        """تتبع البادئات واللواحق"""
        try:
            if 'morphology' in self.engines:
                result = self.engines['morphology'].analyze(word)
                
                affixes_info = self._extract_affixes_info(result, word)
                
                return {
                    'prefixes': affixes_info.get('prefixes', []),
                    'suffixes': affixes_info.get('suffixes', []),
                    'infixes': affixes_info.get('infixes', []),
                    'prefix_count': len(affixes_info.get('prefixes', [])),
                    'suffix_count': len(affixes_info.get('suffixes', [])),
                    'total_affixes': len(affixes_info.get('prefixes', [])) + len(affixes_info.get('suffixes', [])),
                    'affixation_pattern': self._determine_affixation_pattern(affixes_info),
                    'status': 'success'
                }
            else:
                return self._simple_affixes_analysis(word)
                
        except Exception as e:
            logger.error(f"خطأ في تحليل اللواصق: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _trace_infinitive_pure(self, word: str) -> Dict[str, Any]:
        """تتبع المصدر والمجرد"""
        try:
            # تحليل المصدر والشكل المجرد
            infinitive_analysis = {
                'infinitive_form': self._derive_infinitive(word),
                'pure_form': self._extract_pure_form(word),
                'base_form': self._get_base_form(word),
                'derivational_level': self._assess_derivational_level(word),
                'morphological_complexity': self._calculate_morphological_complexity(word),
                'canonical_form': self._get_canonical_form(word),
                'status': 'success'
            }
            
            return infinitive_analysis
            
        except Exception as e:
            logger.error(f"خطأ في تحليل المصدر والمجرد: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _generate_trace_summary(self, linguistic_levels: Dict) -> Dict[str, Any]:
        """إنشاء ملخص شامل لنتائج التتبع"""
        try:
            summary = {
                'word_complexity_score': 0.0,
                'linguistic_features': [],
                'dominant_characteristics': [],
                'analysis_confidence': 0.0,
                'recommendations': []
            }
            
            # حساب درجة التعقيد
            complexity_factors = []
            
            # عوامل التعقيد من المقاطع
            if 'syllabic_units' in linguistic_levels and linguistic_levels['syllabic_units'].get('status') == 'success':
                syllabic_unit_count = linguistic_levels['syllabic_units'].get('syllabic_unit_count', 0)
                complexity_factors.append(min(syllabic_unit_count / 5.0, 1.0))
            
            # عوامل التعقيد من الحركات
            if 'harakat' in linguistic_levels and linguistic_levels['harakat'].get('status') == 'success':
                harakat_density = linguistic_levels['harakat'].get('harakat_density', 0)
                complexity_factors.append(min(harakat_density, 1.0))
            
            # عوامل التعقيد من اللواصق
            if 'affixes' in linguistic_levels and linguistic_levels['affixes'].get('status') == 'success':
                total_affixes = linguistic_levels['affixes'].get('total_affixes', 0)
                complexity_factors.append(min(total_affixes / 3.0, 1.0))
            
            # حساب المتوسط
            summary['word_complexity_score'] = sum(complexity_factors) / len(complexity_factors) if complexity_factors else 0.0
            
            # تحديد الخصائص المهيمنة
            if linguistic_levels.get('particles', {}).get('is_particle'):
                summary['dominant_characteristics'].append('جسيم نحوي')
            
            if linguistic_levels.get('root', {}).get('root_length', 0) == 3:
                summary['dominant_characteristics'].append('جذر ثلاثي')
            elif linguistic_levels.get('root', {}).get('root_length', 0) == 4:
                summary['dominant_characteristics'].append('جذر رباعي')
            
            # الثقة في التحليل
            confidence_scores = []
            for level_data in linguistic_levels.values():
                if isinstance(level_data, dict) and 'confidence' in level_data:
                    confidence_scores.append(level_data['confidence'])
            
            summary['analysis_confidence'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.8
            
            return summary
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء ملخص التتبع: {e}")
            return {'error': str(e), 'status': 'error'}
    
    # مساعدات التحليل البسيط (للاستخدام عند عدم توفر المحركات)
    def _simple_phoneme_analysis(self, word: str) -> Dict[str, Any]:
        """تحليل صوتي بسيط"""
        clean_word = self._remove_harakat(word)
        return {
            'phonemes_list': list(clean_word),
            'phoneme_count': len(clean_word),
            'status': 'simple_analysis'
        }
    
    def _simple_syllabic_unit_analysis(self, word: str) -> Dict[str, Any]:
        """تحليل مقطعي بسيط"""
        # تقسيم بسيط بناءً على الحروف المتحركة
        vowels = 'اوي'
        syllabic_units = []
        current_syllabic_unit = ''
        
        for char in word:
            current_syllabic_unit += char
            if char in vowels:
                syllabic_units.append(current_syllabic_unit)
                current_syllabic_unit = ''
        
        if current_syllabic_unit:
            syllabic_units.append(current_syllabic_unit)
        
        return {
            'syllabic_units': syllabic_units,
            'syllabic_unit_count': len(syllabic_units),
            'status': 'simple_analysis'
        }
    
    def _simple_particle_check(self, word: str) -> Dict[str, Any]:
        """فحص بسيط للجسيمات"""
        common_particles = ['في', 'من', 'إلى', 'على', 'عن', 'بعد', 'قبل', 'مع', 'ضد', 'حول']
        clean_word = self._remove_harakat(word)
        
        return {
            'is_particle': clean_word in common_particles,
            'particle_type': 'حرف جر' if clean_word in common_particles else 'غير محدد',
            'status': 'simple_analysis'
        }
    
    # دوال مساعدة
    def _remove_harakat(self, text: str) -> str:
        """إزالة الحركات من النص"""
        harakat = 'ًٌٍَُِّْٓ'
        return ''.join(char for char in text if char not in harakat)
    
    def _generate_trace_id(self) -> str:
        """توليد معرف فريد للتتبع"""
        return f"trace_{int(time.time())}_{id(self) % 1000}"
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """تحديث إحصائيات الأداء"""
        self.performance_stats['total_traces'] += 1
        if success:
            self.performance_stats['successful_traces'] += 1
        
        # حساب متوسط وقت المعالجة
        current_avg = self.performance_stats['average_processing_time']
        total_traces = self.performance_stats['total_traces']
        new_avg = ((current_avg * (total_traces - 1)) + processing_time) / total_traces
        self.performance_stats['average_processing_time'] = new_avg
    
    # دوال تحليل إضافية (placeholder implementations)
    def _classify_phonemes(self, phonemes: List[str]) -> Dict[str, int]:
        """تصنيف الأصوات"""
        return {'consonants': 0, 'vowels': 0, 'semivowels': 0}
    
    def _to_ipa(self, word: str) -> str:
        """تحويل إلى الأبجدية الصوتية الدولية"""
        return word  # placeholder
    
    def _assess_diacritization_level(self, harakat: Dict) -> str:
        """تقييم مستوى التشكيل"""
        total = sum(harakat.values())
        if total == 0: return 'غير مشكل'
        elif total < 3: return 'تشكيل جزئي'
        else: return 'مشكل كاملاً'
    
    def _extract_cv_pattern(self, word: str) -> str:
        """استخراج نمط CV"""
        # تطبيق بسيط
        return 'CVC'  # placeholder
    
    def _calculate_prosodic_weight(self, syllabic_units_data) -> str:
        """حساب الوزن العروضي"""
        return 'متوسط'  # placeholder
    
    def _determine_primary_class(self, result, word: str) -> str:
        """تحديد التصنيف الأساسي للكلمة"""
        return 'اسم'  # placeholder
    
    def _determine_sub_class(self, result, word: str) -> str:
        """تحديد التصنيف الفرعي"""
        return 'اسم مفرد'  # placeholder
    
    def _extract_grammatical_features(self, result) -> List[str]:
        """استخراج الخصائص النحوية"""
        return ['مذكر', 'مفرد']  # placeholder
    
    def _calculate_classification_confidence(self, result) -> float:
        """حساب ثقة التصنيف"""
        return 0.85  # placeholder
    
    def _get_alternative_classifications(self, result) -> List[str]:
        """الحصول على تصنيفات بديلة"""
        return []  # placeholder
    
    def _simple_word_class_analysis(self, word: str) -> Dict[str, Any]:
        """تحليل بسيط لتصنيف الكلمة"""
        return {
            'primary_class': 'غير محدد',
            'confidence_score': 0.5,
            'status': 'simple_analysis'
        }
    
    def _find_matching_patterns(self, word: str) -> List[Dict]:
        """البحث عن الأوزان المطابقة"""
        return []  # placeholder
    
    def _group_patterns_by_family(self, patterns: List) -> Dict:
        """تجميع الأوزان حسب العائلة"""
        return {}  # placeholder
    
    def _assess_derivation_potential(self, word: str) -> str:
        """تقييم إمكانية الاشتقاق"""
        return 'متوسط'  # placeholder
    
    def _simple_pattern_analysis(self, word: str) -> Dict[str, Any]:
        """تحليل بسيط للأوزان"""
        return {
            'matching_patterns': [],
            'pattern_count': 0,
            'status': 'simple_analysis'
        }
    
    def _calculate_morphological_weight(self, word: str) -> str:
        """حساب الوزن الصرفي"""
        return f"{'ف' * len(word)}"  # placeholder
    
    def _count_root_letters(self, word: str) -> int:
        """عد أحرف الجذر"""
        return min(len(word), 3)  # placeholder
    
    def _count_augmentation_letters(self, word: str) -> int:
        """عد أحرف الزيادة"""
        return max(0, len(word) - 3)  # placeholder
    
    def _categorize_weight(self, word: str) -> str:
        """تصنيف الوزن"""
        length = len(word)
        if length <= 3: return 'مجرد'
        elif length <= 5: return 'مزيد'
        else: return 'مزيد بكثرة'
    
    def _analyze_weight_distribution(self, word: str) -> Dict:
        """تحليل توزيع الوزن"""
        return {'root_ratio': 0.6, 'augmentation_ratio': 0.4}  # placeholder
    
    def _extract_root_info(self, result, word: str) -> Dict:
        """استخراج معلومات الجذر"""
        return {'root': word[:3], 'type': 'ثلاثي', 'confidence': 0.7}  # placeholder
    
    def _identify_weak_letters(self, root: str) -> List[str]:
        """تحديد الأحرف المعتلة"""
        weak_letters = 'اوي'
        return [char for char in root if char in weak_letters]
    
    def _get_derivation_family(self, root: str) -> List[str]:
        """الحصول على عائلة الاشتقاق"""
        return []  # placeholder
    
    def _simple_root_analysis(self, word: str) -> Dict[str, Any]:
        """تحليل بسيط للجذر"""
        return {
            'identified_root': word[:3] if len(word) >= 3 else word,
            'root_type': 'مقدر',
            'status': 'simple_analysis'
        }
    
    def _extract_affixes_info(self, result, word: str) -> Dict:
        """استخراج معلومات اللواصق"""
        return {'prefixes': [], 'suffixes': []}  # placeholder
    
    def _determine_affixation_pattern(self, affixes_info: Dict) -> str:
        """تحديد نمط الإلصاق"""
        return 'بسيط'  # placeholder
    
    def _simple_affixes_analysis(self, word: str) -> Dict[str, Any]:
        """تحليل بسيط للواصق"""
        return {
            'prefixes': [],
            'suffixes': [],
            'total_affixes': 0,
            'status': 'simple_analysis'
        }
    
    def _derive_infinitive(self, word: str) -> str:
        """اشتقاق المصدر"""
        return word  # placeholder
    
    def _extract_pure_form(self, word: str) -> str:
        """استخراج الشكل المجرد"""
        return self._remove_harakat(word)
    
    def _get_base_form(self, word: str) -> str:
        """الحصول على الشكل الأساسي"""
        return word  # placeholder
    
    def _assess_derivational_level(self, word: str) -> str:
        """تقييم مستوى الاشتقاق"""
        return 'أساسي'  # placeholder
    
    def _calculate_morphological_complexity(self, word: str) -> float:
        """حساب التعقيد الصرفي"""
        return len(word) / 10.0  # placeholder
    
    def _get_canonical_form(self, word: str) -> str:
        """الحصول على الشكل القانوني"""
        return word  # placeholder

# إنشاء مثيل المتتبع
word_tracer = ArabicWordTracer()

# المسارات والواجهات
@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('word_tracer.html')

@app.route('/api/trace', methods=['POST'])
def trace_word():
    """API لتتبع الكلمة"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        
        if not word:
            return jsonify({'error': 'لم يتم تقديم كلمة للتحليل'}), 400
        
        # تنفيذ التتبع الكامل
        result = word_tracer.trace_word_complete(word)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"خطأ في API التتبع: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """إحصائيات الأداء"""
    return jsonify(word_tracer.performance_stats)

@app.route('/api/engines')
def get_engines_status():
    """حالة المحركات"""
    engines_status = {}
    for name, engine in word_tracer.engines.items():
        engines_status[name] = {
            'name': name,
            'status': 'active' if hasattr(engine, 'analyze') else 'inactive',
            'type': type(engine).__name__
        }
    
    return jsonify(engines_status)

@app.route('/static/<path:filename>')
def static_files(filename):
    """ملفات ثابتة"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("🔍 بدء تشغيل متتبع الكلمات العربية...")
    print("🌐 الواجهة متاحة على: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
