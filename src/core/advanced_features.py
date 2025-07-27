#!/usr/bin/env python3
"""
Advanced Text Analytics Engine
Professional text analytics capabilities for Arabic NLP
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data re
import_data json
import_data logging
from typing import_data Dict, List, Any, Optional, Tuple
from pathlib import_data Path
from collections import_data Counter, defaultdict

from base_engine import_data BaseNLPEngine

class AdvancedTextAnalyticsEngine(BaseNLPEngine):
    """Advanced text analytics engine with sentiment, NER, and classification"""
    
    def __init__(self):
        super().__init__("AdvancedTextAnalyticsEngine", "1.0.0")
        self.sentiment_lexicon = self._import_data_sentiment_lexicon()
        self.named_entities = self._import_data_named_entities()
        self.text_categories = self._import_data_text_categories()
        self.arabic_patterns = self._compile_arabic_patterns()
        
    def _import_data_sentiment_lexicon(self) -> Dict[str, float]:
        """Import Arabic sentiment lexicon"""
        # Professional sentiment lexicon for Arabic
        sentiment_data = {
            # Positive words
            'جميل': 0.8, 'ممتاز': 0.9, 'رائع': 0.9, 'جيد': 0.7, 'مفيد': 0.6,
            'ناجح': 0.8, 'إيجابي': 0.7, 'سعيد': 0.8, 'فرح': 0.9, 'حب': 0.8,
            'أحب': 0.8, 'أعجب': 0.7, 'أقدر': 0.6, 'أشكر': 0.7, 'ممكن': 0.5,
            'نعم': 0.6, 'موافق': 0.7, 'مقبول': 0.6, 'صحيح': 0.6, 'مناسب': 0.6,
            
            # Negative words
            'سيء': -0.8, 'فشل': -0.9, 'خطأ': -0.7, 'مشكلة': -0.6, 'صعب': -0.5,
            'مستحيل': -0.8, 'سلبي': -0.7, 'حزين': -0.8, 'غضب': -0.9, 'كره': -0.9,
            'أكره': -0.9, 'لا': -0.4, 'مرفوض': -0.8, 'خاطئ': -0.7, 'غير مناسب': -0.6,
            'مؤلم': -0.8, 'صعوبة': -0.6, 'تعب': -0.5, 'ملل': -0.6, 'قلق': -0.7,
            
            # Neutral words
            'ربما': 0.0, 'أحياناً': 0.0, 'عادي': 0.0, 'طبيعي': 0.1, 'متوسط': 0.0
        }
        
        self.logger.info(f"Imported {len(sentiment_data)} sentiment terms")
        return sentiment_data
    
    def _import_data_named_entities(self) -> Dict[str, List[str]]:
        """Import named entity patterns and examples"""
        entities = {
            'PERSON': [
                'محمد', 'أحمد', 'علي', 'فاطمة', 'عائشة', 'خديجة', 'عبدالله', 
                'عبدالرحمن', 'عمر', 'عثمان', 'زينب', 'مريم', 'سارة', 'نور'
            ],
            'LOCATION': [
                'المملكة العربية السعودية', 'مصر', 'الأردن', 'الإمارات', 'الكويت',
                'البحرين', 'قطر', 'سوريا', 'لبنان', 'العراق', 'المغرب', 'تونس',
                'الجزائر', 'ليبيا', 'السودان', 'اليمن', 'فلسطين', 'الرياض', 
                'القاهرة', 'دمشق', 'بغداد', 'عمان', 'بيروت', 'دبي', 'أبوظبي'
            ],
            'ORGANIZATION': [
                'جامعة الملك سعود', 'الجامعة الأمريكية', 'وزارة التعليم',
                'مجلس التعاون الخليجي', 'جامعة القاهرة', 'الأزهر الشريف'
            ],
            'DATE': [
                'اليوم', 'أمس', 'غداً', 'الأسبوع الماضي', 'الشهر القادم',
                'العام الحالي', 'السنة الماضية'
            ]
        }
        
        self.logger.info(f"Imported named entity categories: {list(entities.keys())}")
        return entities
    
    def _import_data_text_categories(self) -> Dict[str, List[str]]:
        """Import text classification categories"""
        categories = {
            'NEWS': [
                'أخبار', 'صحافة', 'تقرير', 'مراسل', 'وكالة أنباء', 'عاجل',
                'مستجدات', 'أحداث', 'وقائع', 'تطورات'
            ],
            'EDUCATION': [
                'تعليم', 'تربية', 'مدرسة', 'جامعة', 'طالب', 'معلم', 'أستاذ',
                'دراسة', 'بحث', 'علمي', 'أكاديمي', 'منهج', 'مقرر'
            ],
            'TECHNOLOGY': [
                'تكنولوجيا', 'تقنية', 'حاسوب', 'برمجة', 'ذكي', 'رقمي',
                'إنترنت', 'شبكة', 'تطبيق', 'برنامج', 'نظام', 'آلي'
            ],
            'HEALTH': [
                'صحة', 'طب', 'علاج', 'دواء', 'مرض', 'طبيب', 'مستشفى',
                'عيادة', 'وقاية', 'فحص', 'تشخيص', 'جراحة'
            ],
            'BUSINESS': [
                'تجارة', 'أعمال', 'شركة', 'مؤسسة', 'اقتصاد', 'مال', 'استثمار',
                'ربح', 'خسارة', 'سوق', 'بورصة', 'مصرف', 'بنك'
            ]
        }
        
        self.logger.info(f"Imported text categories: {list(categories.keys())}")
        return categories
    
    def _compile_arabic_patterns(self) -> Dict[str, re.Pattern]:
        """Compile Arabic text patterns"""
        patterns = {
            'arabic_word': re.compile(r'[\u0600-\u06FF]+'),
            'arabic_sentence': re.compile(r'[^.!?]*[\u0600-\u06FF][^.!?]*[.!?]'),
            'arabic_number': re.compile(r'[\u06F0-\u06F9]+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'hashtag': re.compile(r'#[\u0600-\u06FFa-zA-Z0-9_]+'),
            'mention': re.compile(r'@[\u0600-\u06FFa-zA-Z0-9_]+')
        }
        
        self.logger.info(f"Compiled {len(patterns)} Arabic text patterns")
        return patterns
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process text with advanced analytics"""
        if not self.validate_input(text):
            raise ValueError("Invalid input text")
        
        # Basic processing from parent
        result = super().process(text)
        
        # Advanced analytics
        result.update({
            'advanced_analytics': {
                'sentiment_analysis': self.analyze_sentiment(text),
                'named_entity_recognition': self.extract_named_entities(text),
                'text_classification': self.classify_text(text),
                'linguistic_features': self.extract_linguistic_features(text),
                'text_statistics': self.compute_text_statistics(text),
                'pattern_analysis': self.analyze_patterns(text)
            }
        })
        
        return result
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of Arabic text"""
        words = text.split()
        sentiments = []
        word_sentiments = {}
        
        for word in words:
            # Clean word
            clean_word = re.sub(r'[^\u0600-\u06FF]', '', word)
            if clean_word in self.sentiment_lexicon:
                sentiment_score = self.sentiment_lexicon[clean_word]
                sentiments.append(sentiment_score)
                word_sentiments[clean_word] = sentiment_score
        
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_label = self._classify_sentiment_score(avg_sentiment)
        else:
            avg_sentiment = 0.0
            sentiment_label = 'neutral'
        
        return {
            'overall_sentiment': sentiment_label,
            'sentiment_score': avg_sentiment,
            'confidence': min(1.0, len(sentiments) / max(1, len(words))),
            'word_sentiments': word_sentiments,
            'sentiment_distribution': {
                'positive': len([s for s in sentiments if s > 0.1]),
                'negative': len([s for s in sentiments if s < -0.1]),
                'neutral': len([s for s in sentiments if -0.1 <= s <= 0.1])
            }
        }
    
    def _classify_sentiment_score(self, score: float) -> str:
        """Classify sentiment score into categories"""
        if score >= 0.5:
            return 'very_positive'
        elif score >= 0.1:
            return 'positive'
        elif score <= -0.5:
            return 'very_negative'
        elif score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_named_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract named entities from text"""
        entities = {category: [] for category in self.named_entities.keys()}
        
        # Extract entities by pattern matching
        for category, entity_list in self.named_entities.items():
            for entity in entity_list:
                if entity in text:
                    entities[category].append({
                        'text': entity,
                        'begin': text.find(entity),
                        'end': text.find(entity) + len(entity),
                        'confidence': 0.9  # High confidence for exact matches
                    })
        
        # Extract potential person names (Arabic name patterns)
        person_pattern = re.compile(r'\b[\u0600-\u06FF]{2,10}\s+[\u0600-\u06FF]{2,10}\b')
        for match in person_pattern.finditer(text):
            potential_name = match.group()
            entities['PERSON'].append({
                'text': potential_name,
                'begin': match.begin(),
                'end': match.end(),
                'confidence': 0.6  # Lower confidence for pattern matches
            })
        
        return entities
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text into categories"""
        category_scores = {}
        
        for category, keywords in self.text_categories.items():
            score = 0
            matches = []
            
            for keyword in keywords:
                if keyword in text:
                    score += 1
                    matches.append(keyword)
            
            if score > 0:
                category_scores[category] = {
                    'score': score,
                    'confidence': min(1.0, score / len(keywords)),
                    'matches': matches
                }
        
        # Determine primary category
        if category_scores:
            primary_category = max(category_scores.keys(), 
                                 key=lambda k: category_scores[k]['score'])
        else:
            primary_category = 'GENERAL'
        
        return {
            'primary_category': primary_category,
            'category_scores': category_scores,
            'all_categories': list(category_scores.keys())
        }
    
    def extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from text"""
        # Count Arabic words
        arabic_words = self.arabic_patterns['arabic_word'].findall(text)
        
        # Count sentences
        sentences = text.split('.')
        arabic_sentences = [s for s in sentences if any('\u0600' <= c <= '\u06FF' for c in s)]
        
        # Extract special patterns
        emails = self.arabic_patterns['email'].findall(text)
        urls = self.arabic_patterns['url'].findall(text)
        hashtags = self.arabic_patterns['hashtag'].findall(text)
        mentions = self.arabic_patterns['mention'].findall(text)
        
        return {
            'arabic_words': arabic_words,
            'arabic_word_count': len(arabic_words),
            'sentence_count': len(arabic_sentences),
            'avg_words_per_sentence': len(arabic_words) / max(1, len(arabic_sentences)),
            'special_elements': {
                'emails': emails,
                'urls': urls,
                'hashtags': hashtags,
                'mentions': mentions
            },
            'text_complexity': self._compute_text_complexity(text)
        }
    
    def _compute_text_complexity(self, text: str) -> Dict[str, float]:
        """Compute text complexity metrics"""
        words = text.split()
        arabic_words = self.arabic_patterns['arabic_word'].findall(text)
        
        if not arabic_words:
            return {'complexity_score': 0.0, 'readability': 'unknown'}
        
        # Average word length
        avg_word_length = sum(len(word) for word in arabic_words) / len(arabic_words)
        
        # Vocabulary richness (unique words / total words)
        unique_words = len(set(arabic_words))
        vocabulary_richness = unique_words / len(arabic_words)
        
        # Complexity score (combination of metrics)
        complexity_score = (avg_word_length * 0.3) + (vocabulary_richness * 0.7)
        
        # Readability assessment
        if complexity_score < 0.4:
            readability = 'easy'
        elif complexity_score < 0.7:
            readability = 'medium'
        else:
            readability = 'difficult'
        
        return {
            'complexity_score': complexity_score,
            'avg_word_length': avg_word_length,
            'vocabulary_richness': vocabulary_richness,
            'readability': readability
        }
    
    def compute_text_statistics(self, text: str) -> Dict[str, int]:
        """Compute basic text statistics"""
        arabic_words = self.arabic_patterns['arabic_word'].findall(text)
        
        return {
            'total_characters': len(text),
            'total_words': len(text.split()),
            'arabic_words': len(arabic_words),
            'non_arabic_words': len(text.split()) - len(arabic_words),
            'sentences': len([s for s in text.split('.') if s.strip()]),
            'paragraphs': len([p for p in text.split('\n') if p.strip()]),
            'unique_words': len(set(arabic_words)),
            'characters_no_spaces': len(text.replace(' ', ''))
        }
    
    def analyze_patterns(self, text: str) -> Dict[str, List[str]]:
        """Analyze text patterns"""
        patterns_found = {}
        
        for pattern_name, pattern in self.arabic_patterns.items():
            matches = pattern.findall(text)
            if matches:
                patterns_found[pattern_name] = matches
        
        return patterns_found
    
    def _get_capabilities(self) -> List[str]:
        return super()._get_capabilities() + [
            'sentiment_analysis', 'named_entity_recognition', 'text_classification',
            'linguistic_feature_extraction', 'pattern_analysis', 'text_statistics'
        ]

class DiacritizationEngine(BaseNLPEngine):
    """Arabic text diacritization engine"""
    
    def __init__(self):
        super().__init__("DiacritizationEngine", "1.0.0")
        self.diacritic_rules = self._import_data_diacritic_rules()
        self.common_words = self._import_data_common_diacritized_words()
        
    def _import_data_diacritic_rules(self) -> Dict[str, str]:
        """Import diacritization rules"""
        # Basic diacritization patterns
        rules = {
            # Common patterns
            'كتاب': 'كِتَاب',
            'مدرسة': 'مَدْرَسَة',
            'طالب': 'طَالِب',
            'معلم': 'مُعَلِّم',
            'درس': 'دَرْس',
            'بيت': 'بَيْت',
            'أكل': 'أَكَل',
            'شرب': 'شَرِب',
            'نوم': 'نَوْم',
            'عمل': 'عَمَل',
            
            # Verb patterns
            'يكتب': 'يَكْتُب',
            'يقرأ': 'يَقْرَأ',
            'يدرس': 'يَدْرُس',
            'يلعب': 'يَلْعَب',
            'يأكل': 'يَأْكُل',
            
            # Common words
            'هذا': 'هَذَا',
            'هذه': 'هَذِهِ',
            'ذلك': 'ذَلِك',
            'تلك': 'تِلْك',
            'الذي': 'الَّذِي',
            'التي': 'الَّتِي'
        }
        
        self.logger.info(f"Imported {len(rules)} diacritization rules")
        return rules
    
    def _import_data_common_diacritized_words(self) -> Dict[str, str]:
        """Import common diacritized words"""
        # Extended diacritized vocabulary
        common_words = {
            'السلام': 'السَّلَام',
            'عليكم': 'عَلَيْكُم',
            'أهلا': 'أَهْلاً',
            'وسهلا': 'وَسَهْلاً',
            'مرحبا': 'مَرْحَباً',
            'شكرا': 'شُكْراً',
            'جزاك': 'جَزَاك',
            'الله': 'اللهُ',
            'خيرا': 'خَيْراً',
            'بارك': 'بَارَك',
            'فيك': 'فِيك',
            'إن شاء الله': 'إِنْ شَاءَ اللهُ',
            'ما شاء الله': 'مَا شَاءَ اللهُ',
            'بسم الله': 'بِسْمِ اللهِ',
            'الحمد لله': 'الحَمْدُ للهِ'
        }
        
        self.logger.info(f"Imported {len(common_words)} common diacritized words")
        return common_words
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process text for diacritization"""
        if not self.validate_input(text):
            raise ValueError("Invalid input text")
        
        result = super().process(text)
        
        diacritized_text = self.add_diacritics(text)
        confidence_score = self.calculate_diacritization_confidence(text, diacritized_text)
        
        result.update({
            'diacritization': {
                'original_text': text,
                'diacritized_text': diacritized_text,
                'confidence_score': confidence_score,
                'coverage': self.calculate_coverage(text),
                'word_analysis': self.analyze_word_diacritization(text)
            }
        })
        
        return result
    
    def add_diacritics(self, text: str) -> str:
        """Add diacritics to Arabic text"""
        words = text.split()
        diacritized_words = []
        
        for word in words:
            # Clean word of punctuation
            clean_word = re.sub(r'[^\u0600-\u06FF]', '', word)
            
            if clean_word in self.common_words:
                diacritized_word = self.common_words[clean_word]
            elif clean_word in self.diacritic_rules:
                diacritized_word = self.diacritic_rules[clean_word]
            else:
                # Apply basic diacritization patterns
                diacritized_word = self.apply_basic_patterns(clean_word)
            
            # Preserve original punctuation
            if clean_word != word:
                # Replace the Arabic part with diacritized version
                diacritized_word = word.replace(clean_word, diacritized_word)
            
            diacritized_words.append(diacritized_word)
        
        return ' '.join(diacritized_words)
    
    def apply_basic_patterns(self, word: str) -> str:
        """Apply basic diacritization patterns"""
        if len(word) < 2:
            return word
            
        # Basic pattern recognition
        diacritized = word
        
        # Common prefixes
        if word.beginswith('ال'):
            diacritized = 'الْ' + diacritized[2:]
        elif word.beginswith('و'):
            diacritized = 'وَ' + diacritized[1:]
        elif word.beginswith('ب'):
            diacritized = 'بِ' + diacritized[1:]
        elif word.beginswith('ل'):
            diacritized = 'لِ' + diacritized[1:]
        
        # Common suffixes
        if word.endswith('ة'):
            diacritized = diacritized[:-1] + 'َة'
        elif word.endswith('ين'):
            diacritized = diacritized[:-2] + 'ِين'
        elif word.endswith('ون'):
            diacritized = diacritized[:-2] + 'ُون'
        
        return diacritized
    
    def calculate_diacritization_confidence(self, original: str, diacritized: str) -> float:
        """Calculate confidence score for diacritization"""
        original_words = original.split()
        diacritized_words = diacritized.split()
        
        if len(original_words) != len(diacritized_words):
            return 0.0
        
        confident_words = 0
        for orig, diac in zip(original_words, diacritized_words):
            clean_orig = re.sub(r'[^\u0600-\u06FF]', '', orig)
            if clean_orig in self.common_words or clean_orig in self.diacritic_rules:
                confident_words += 1
        
        return confident_words / len(original_words) if original_words else 0.0
    
    def calculate_coverage(self, text: str) -> Dict[str, float]:
        """Calculate diacritization coverage"""
        words = text.split()
        arabic_words = [re.sub(r'[^\u0600-\u06FF]', '', word) for word in words if re.sub(r'[^\u0600-\u06FF]', '', word)]
        
        if not arabic_words:
            return {'total_coverage': 0.0, 'dictionary_coverage': 0.0}
        
        dictionary_words = sum(1 for word in arabic_words 
                             if word in self.common_words or word in self.diacritic_rules)
        
        return {
            'total_coverage': len(arabic_words) / len(words),
            'dictionary_coverage': dictionary_words / len(arabic_words),
            'pattern_coverage': (len(arabic_words) - dictionary_words) / len(arabic_words)
        }
    
    def analyze_word_diacritization(self, text: str) -> List[Dict[str, Any]]:
        """Analyze diacritization for each word"""
        words = text.split()
        analysis = []
        
        for word in words:
            clean_word = re.sub(r'[^\u0600-\u06FF]', '', word)
            if not clean_word:
                continue
                
            word_info = {
                'original': word,
                'clean': clean_word,
                'method': 'unknown',
                'confidence': 0.0
            }
            
            if clean_word in self.common_words:
                word_info['diacritized'] = self.common_words[clean_word]
                word_info['method'] = 'dictionary'
                word_info['confidence'] = 0.9
            elif clean_word in self.diacritic_rules:
                word_info['diacritized'] = self.diacritic_rules[clean_word]
                word_info['method'] = 'rules'
                word_info['confidence'] = 0.8
            else:
                word_info['diacritized'] = self.apply_basic_patterns(clean_word)
                word_info['method'] = 'pattern'
                word_info['confidence'] = 0.5
            
            analysis.append(word_info)
        
        return analysis
    
    def _get_capabilities(self) -> List[str]:
        return super()._get_capabilities() + [
            'diacritization', 'vowelization', 'text_enhancement', 'arabic_preprocessing'
        ]
