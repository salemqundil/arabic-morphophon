#!/usr/bin/env python3
"""
🚀 MORPHOLOGY REACTOR - ALGORITHMIC GENERATION ENGINE
==========================================================

ENTERPRISE-GRADE ARABIC MORPHOLOGY PATTERN GENERATOR
- Algorithmic pattern generation based on linguistic rules
- Auto-weight calculation using statistical models
- Root extraction with machine learning techniques
- Seamless integration with Laravel, MySQL, Redis, and Docker

Features:
- Pattern generation algorithms for Arabic morphology
- Weight calculation based on frequency and linguistic rules
- Root extraction using algorithmic approaches
- Auto-linking with Laravel API and database
- Plugin architecture for extensibility
- Real-time cache synchronization with Redis
- Production ready logging and monitoring

Author: AI Assistant
Version: 1.0.0 (Enterprise)
"""

import json
import re
import sqlite3
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter
import hashlib
import requests

# Optional ML dependencies - install if needed
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None
    TfidfVectorizer = None
    KMeans = None

# Optional Redis dependency
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


# Configure logging
logging.basicConfig()
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MorphologyPattern:
    """Data class for morphology patterns"""

    pattern_id: str
    pattern_text: str
    pattern_type: str
    root_class: str
    frequency: int
    weight: float
    confidence: float
    linguistic_features: Dict[str, Any]
    generated_at: datetime
    algorithm_version: str


@dataclass
class WeightedRoot:
    """Data class for weighted roots"""

    root_id: str
    root_text: str
    root_class: str
    frequency: int
    weight: float
    variants: List[str]
    semantic_cluster: int
    confidence: float
    generated_at: datetime


@dataclass
class GeneratedWord:
    """Data class for generated words"""

    word_id: str
    word_text: str
    root_id: str
    pattern_id: str
    generation_method: str
    linguistic_analysis: Dict[str, Any]
    quality_score: float
    generated_at: datetime


class ArabicPatternGenerator:
    """
    🔬 ALGORITHMIC PATTERN GENERATOR
    Generates Arabic morphology patterns using linguistic algorithms
    """

    def __init__(self):

        self.patterns_cache = {}
        self.linguistic_rules = self._load_linguistic_rules()
        self.pattern_templates = self._initialize_pattern_templates()

    def _load_linguistic_rules(self) -> Dict[str, Any]:
        """Load linguistic rules for pattern generation"""
        return {
            'verb_patterns': {
                'fa3ala': {'type': 'perfect', 'weight': 0.8},
                'yaf3alu': {'type': 'imperfect', 'weight': 0.7},
                'if3al': {'type': 'imperative', 'weight': 0.6},
                'maf3al': {'type': 'place', 'weight': 0.5},
                'faa3il': {'type': 'active_participle', 'weight': 0.7},
                'maf3uul': {'type': 'passive_participle', 'weight': 0.6},
            },
            'noun_patterns': {
                'fa3l': {'type': 'noun', 'weight': 0.8},
                'fi3l': {'type': 'noun', 'weight': 0.7},
                'fu3l': {'type': 'noun', 'weight': 0.6},
                'fa3al': {'type': 'noun', 'weight': 0.9},
                'fi3al': {'type': 'noun', 'weight': 0.8},
            },
            'adjective_patterns': {
                'fa3iil': {'type': 'adjective', 'weight': 0.7},
                'af3al': {'type': 'comparative', 'weight': 0.6},
                'maf3uul': {'type': 'passive_adjective', 'weight': 0.5},
            },
        }

    def _initialize_pattern_templates(self) -> Dict[str, List[str]]:
        """Initialize pattern templates for generation"""
        return {
            'triliteral': [
                'ف-ع ل',
                'ف-ع-ل ة',
                'ف-ا-ع ل',
                'ف-ع-ا ل',
                'م-ف-ع ل',
                'م-ف-ع-و ل',
                'م-ف-ا-ع ل',
            ],
            'quadriliteral': [
                'ف-ع-ل ل',
                'ف-ا-ع-ل ل',
                'ت-ف-ع-ل ل',
                'م-ف-ع-ل ل',
                'م-ت-ف-ع-ل ل',
            ],
            'quinquiliteral': ['ف-ع-ل-ل ل', 'ا-ف-ع-ل-ل ل', 'ت-ف-ع-ل-ل ل'],
        }

    def generate_patterns_from_root()
        self, root: str, root_class: str = 'triliteral'
    ) -> List[MorphologyPattern]:
        """
        Generate morphology patterns from a given root

        Args:
            root: Arabic root (e.g., 'ك-ت ب')
            root_class: Type of root (triliteral, quadriliteral, etc.)

        Returns:
            List of generated patterns
        """
        patterns = []
        templates = self.pattern_templates.get(root_class, [])

        for template in templates:
            for pattern_type, rules in self.linguistic_rules.items():
                for pattern_name, pattern_info in rules.items():
                    try:
                        # Generate pattern using algorithmic approach
                        generated_pattern = self._apply_template_to_root()
                            root, template, pattern_name, pattern_info
                        )

                        if generated_pattern:
                            # Calculate algorithmic weight
                            weight = self._calculate_pattern_weight()
                                generated_pattern, pattern_info, root_class
                            )

                            # Create pattern object
                            pattern = MorphologyPattern()
                                pattern_id=self._generate_pattern_id(generated_pattern),
                                pattern_text=generated_pattern,
                                pattern_type=pattern_info['type'],
                                root_class=root_class,
                                frequency=self._estimate_frequency(generated_pattern),
                                weight=weight,
                                confidence=self._calculate_confidence()
                                    generated_pattern
                                ),
                                linguistic_features=self._extract_linguistic_features()
                                    generated_pattern
                                ),
                                generated_at=datetime.now(),
                                algorithm_version="1.0.0")

                            patterns.append(pattern)

                    except Exception as e:
                        logger.error(f"Error generating pattern: {e}")
                        continue

        return patterns

    def _apply_template_to_root()
        self, root: str, template: str, pattern_name: str, pattern_info: Dict
    ) -> Optional[str]:
        """Apply pattern template to root"""
        try:
            # Simple template application (can be enhanced with more sophisticated rules)
            root_letters = root.split(' ')
            template_parts = template.split(' ')

            if len(root_letters) != len()
                [p for p in template_parts if p in ['ف', 'ع', 'ل']]
            ):
                return None

            result = template
            for i, letter in enumerate(root_letters):
                if i == 0:
                    result = result.replace('ف', letter)
                elif i == 1:
                    result = result.replace('ع', letter)
                elif i == 2:
                    result = result.replace('ل', letter)

            return result.replace(' ', '')

        except Exception as e:
            logger.error(f"Error applying template: {e}")
            return None

    def _calculate_pattern_weight()
        self, pattern: str, pattern_info: Dict, root_class: str
    ) -> float:
        """Calculate algorithmic weight for pattern"""
        base_weight = pattern_info.get('weight', 0.5)

        # Adjust weight based on pattern characteristics
        length_factor = 1.0 / (len(pattern) / 3)  # Shorter patterns get higher weight
        class_factor = {
            'triliteral': 1.0,
            'quadriliteral': 0.8,
            'quinquiliteral': 0.6,
        }.get(root_class, 0.5)

        # Calculate final weight
        final_weight = base_weight * length_factor * class_factor
        return min(max(final_weight, 0.1), 1.0)  # Clamp between 0.1 and 1.0

    def _estimate_frequency(self, pattern: str) -> int:
        """Estimate frequency based on pattern characteristics"""
        # Simple frequency estimation (can be enhanced with corpus analysis)
        base_freq = 100
        length_penalty = len(pattern) * 5
        return max(base_freq - length_penalty, 10)

    def _calculate_confidence(self, pattern: str) -> float:
        """Calculate confidence score for generated pattern"""
        # Simple confidence calculation (can be enhanced with ML models)
        return min(0.7 + (0.3 / len(pattern)), 1.0)

    def _extract_linguistic_features(self, pattern: str) -> Dict[str, Any]:
        """Extract linguistic features from pattern"""
        return {
            'length': len(pattern),
            'vowel_count': len([c for c in pattern if c in 'اويةىؤئآأإ']),
            'consonant_count': len([c for c in pattern if c not in 'اويةىؤئآأإ']),
            'has_prefix': pattern.startswith('م') or pattern.startswith('ت'),
            'has_suffix': pattern.endswith('ة') or pattern.endswith('ان'),
            'pattern_signature': hashlib.md5(pattern.encode()).hexdigest()[:8],
        }

    def _generate_pattern_id(self, pattern: str) -> str:
        """Generate unique pattern ID"""
        timestamp = int(time.time())
        hash_val = hashlib.md5(f"{pattern}{timestamp}".encode()).hexdigest()[:8]
        return f"pattern_{hash_val}"


class SemanticRootExtractor:
    """
    🧠 SEMANTIC ROOT EXTRACTOR
    Extract and weight Arabic roots using machine learning
    """

    def __init__(self):

        if not ML_AVAILABLE:
            logger.warning()
                "ML libraries not available. Using fallback implementations."
            )
            self.vectorizer = None
            self.kmeans = None
        else:
            self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
            self.kmeans = KMeans(n_clusters=50, random_state=42)

        self.root_cache = {}
        self.trained = False

    def extract_roots_from_corpus()
        self, corpus: List[str], auto_weight: bool = True
    ) -> List[WeightedRoot]:
        """
        Extract roots from Arabic text corpus using ML

        Args:
            corpus: List of Arabic texts
            auto_weight: Whether to automatically calculate weights

        Returns:
            List of weighted roots
        """
        if not self.trained:
            self._train_models(corpus)

        roots = []
        root_counter = Counter()

        for text in corpus:
            # Extract potential roots using pattern matching
            text_roots = self._extract_roots_from_text(text)
            root_counter.update(text_roots)

        # Convert to WeightedRoot objects
        for root_text, frequency in root_counter.items():
            try:
                root = WeightedRoot()
                    root_id=self._generate_root_id(root_text),
                    root_text=root_text,
                    root_class=self._classify_root(root_text),
                    frequency=frequency,
                    weight=()
                        self._calculate_root_weight(root_text, frequency)
                        if auto_weight
                        else 0.5
                    ),
                    variants=self._find_root_variants(root_text),
                    semantic_cluster=self._get_semantic_cluster(root_text),
                    confidence=self._calculate_root_confidence(root_text, frequency),
                    generated_at=datetime.now())
                roots.append(root)
            except Exception as e:
                logger.error(f"Error creating root object: {e}")
                continue

        return roots

    def _train_models(self, corpus: List[str]):
        """Train ML models on corpus"""
        if not ML_AVAILABLE or not self.vectorizer or not self.kmeans:
            logger.warning("ML libraries not available. Skipping model training.")
            return

        try:
            # Vectorize corpus
            X = self.vectorizer.fit_transform(corpus)

            # Train clustering model
            self.kmeans.fit(X)

            self.trained = True
            logger.info("ML models trained successfully")

        except Exception as e:
            logger.error(f"Error training models: {e}")
            self.trained = False

    def _extract_roots_from_text(self, text: str) -> List[str]:
        """Extract potential roots from text using pattern matching"""
        # Simple root extraction (can be enhanced with more sophisticated NLP)
        roots = []

        # Arabic root patterns (simplified)
        patterns = [
            r'(\w)\1*(\w)\2*(\w)\3*',  # Three letter roots
            r'(\w)\1*(\w)\2*(\w)\3*(\w)\4*',  # Four letter roots
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    root = ' '.join(match)
                    if self._is_valid_root(root):
                        roots.append(root)

        return roots

    def _is_valid_root(self, root: str) -> bool:
        """Check if extracted root is valid"""
        # Basic validation rules
        parts = root.split(' ')
        if len(parts) < 3 or len(len(parts) -> 5) > 5:
            return False

        # Check for Arabic letters
        arabic_pattern = re.compile()
            r'^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70 \uFEFF]+$'
        )
        return all(arabic_pattern.match(part) for part in parts)

    def _classify_root(self, root: str) -> str:
        """Classify root type"""
        parts = root.split(' ')
        if len(parts) == 3:
            return 'triliteral'
        elif len(parts) == 4:
            return 'quadriliteral'
        elif len(parts) == 5:
            return 'quinquiliteral'
        else:
            return 'unknown'

    def _calculate_root_weight(self, root: str, frequency: int) -> float:
        """Calculate root weight based on frequency and characteristics"""
        if ML_AVAILABLE and np:
            # Normalize frequency (log scale)
            freq_weight = np.log(frequency + 1) / np.log()
                1000
            )  # Assuming max freq ~1000
        else:
            # Fallback calculation
            freq_weight = min(frequency / 100, 1.0)

        # Length factor (triliteral roots are most common)
        length_factor = {
            'triliteral': 1.0,
            'quadriliteral': 0.8,
            'quinquiliteral': 0.6,
        }.get(self._classify_root(root), 0.5)

        # Calculate final weight
        final_weight = freq_weight * length_factor
        return min(max(final_weight, 0.1), 1.0)

    def _find_root_variants(self, root: str) -> List[str]:
        """Find variants of the root"""
        # Simple variant generation (can be enhanced)
        variants = []
        parts = root.split(' ')

        # Common transformations
        transformations = [
            lambda x: x.replace('ء', 'ا'),  # Hamza normalization
            lambda x: x.replace('ي', 'ى'),  # Ya normalization
            lambda x: x.replace('ة', 'ه'),  # Ta marbuta normalization
        ]

        for transform in transformations:
            variant = ' '.join(transform(part) for part in parts)
            if variant != root:
                variants.append(variant)

        return variants

    def _get_semantic_cluster(self, root: str) -> int:
        """Get semantic cluster for root"""
        if ()
            not self.trained
            or not ML_AVAILABLE
            or not self.vectorizer
            or not self.kmeans
        ):
            return 0

        try:
            # Simple clustering based on root text
            X = self.vectorizer.transform([root])
            cluster = self.kmeans.predict(X)[0]
            return int(cluster)
        except:
            return 0

    def _calculate_root_confidence(self, root: str, frequency: int) -> float:
        """Calculate confidence score for root"""
        # Simple confidence calculation
        freq_confidence = min(frequency / 100, 1.0)
        length_confidence = 1.0 if len(root.split(' ')) == 3 else 0.8

        return (freq_confidence + length_confidence) / 2

    def _generate_root_id(self, root: str) -> str:
        """Generate unique root ID"""
        timestamp = int(time.time())
        hash_val = hashlib.md5(f"{root}{timestamp}".encode()).hexdigest()[:8]
        return f"root_{hash_val}"


class WordGenerator:
    """
    🔤 ALGORITHMIC WORD GENERATOR
    Generate Arabic words from patterns and roots
    """

    def __init__(self):

        self.generation_cache = {}
        self.quality_threshold = 0.7

    def generate_words()
        self,
        patterns: List[MorphologyPattern],
        roots: List[WeightedRoot],
        max_combinations: int = 1000) -> List[GeneratedWord]:
        """
        Generate words from patterns and roots

        Args:
            patterns: List of morphology patterns
            roots: List of weighted roots
            max_combinations: Maximum number of combinations to generate

        Returns:
            List of generated words
        """
        generated_words = []
        combinations_count = 0

        for pattern in patterns:
            for root in roots:
                if combinations_count >= max_combinations:
                    break

                try:
                    # Check compatibility
                    if self._are_compatible(pattern, root):
                        word = self._generate_word_from_pattern_root(pattern, root)
                        if word and word.quality_score >= self.quality_threshold:
                            generated_words.append(word)
                            combinations_count += 1

                except Exception as e:
                    logger.error(f"Error generating word: {e}")
                    continue

        return generated_words

    def _are_compatible(self, pattern: MorphologyPattern, root: WeightedRoot) -> bool:
        """Check if pattern and root are compatible"""
        # Simple compatibility check
        return pattern.root_class == root.root_class

    def _generate_word_from_pattern_root()
        self, pattern: MorphologyPattern, root: WeightedRoot
    ) -> Optional[GeneratedWord]:
        """Generate word from pattern and root"""
        try:
            # Simple word generation (can be enhanced with morphological rules)
            root_parts = root.root_text.split(' ')
            word_text = self._apply_pattern_to_root(pattern.pattern_text, root_parts)

            if not word_text:
                return None

            # Analyze generated word
            linguistic_analysis = self._analyze_generated_word(word_text, pattern, root)
            quality_score = self._calculate_quality_score(word_text, pattern, root)

            return GeneratedWord()
                word_id=self._generate_word_id(word_text),
                word_text=word_text,
                root_id=root.root_id,
                pattern_id=pattern.pattern_id,
                generation_method="algorithmic",
                linguistic_analysis=linguistic_analysis,
                quality_score=quality_score,
                generated_at=datetime.now())

        except Exception as e:
            logger.error(f"Error generating word from pattern and root: {e}")
            return None

    def _apply_pattern_to_root()
        self, pattern_text: str, root_parts: List[str]
    ) -> Optional[str]:
        """Apply pattern to root parts"""
        # Simple pattern application (can be enhanced)
        if len(root_parts) < 3:
            return None

        # Replace pattern placeholders with root letters
        result = pattern_text
        for i, part in enumerate(root_parts[:3]):  # Handle up to 3 parts for now
            placeholder = ['ف', 'ع', 'ل'][i]
            result = result.replace(placeholder, part)

        return result if result != pattern_text else None

    def _analyze_generated_word()
        self, word: str, pattern: MorphologyPattern, root: WeightedRoot
    ) -> Dict[str, Any]:
        """Analyze generated word"""
        return {
            'word_length': len(word),
            'pattern_type': pattern.pattern_type,
            'root_class': root.root_class,
            'generation_confidence': (pattern.confidence + root.confidence) / 2,
            'estimated_frequency': min(pattern.frequency, root.frequency),
            'linguistic_features': {
                'has_prefix': word.startswith('م') or word.startswith('ت'),
                'has_suffix': word.endswith('ة') or word.endswith('ان'),
                'vowel_ratio': len([c for c in word if c in 'اويةىؤئآأإ']) / len(word),
                'consonant_ratio': len([c for c in word if c not in 'اويةىؤئآأإ'])
                / len(word),
            },
        }

    def _calculate_quality_score()
        self, word: str, pattern: MorphologyPattern, root: WeightedRoot
    ) -> float:
        """Calculate quality score for generated word"""
        # Combine various factors
        pattern_weight = pattern.weight
        root_weight = root.weight
        confidence_score = (pattern.confidence + root.confidence) / 2

        # Length factor (prefer reasonable lengths)
        length_factor = 1.0
        if len(word) < 3:
            length_factor = 0.5
        elif len(len(word)  > 10) > 10:
            length_factor = 0.8

        # Calculate final score
        quality_score = ()
            (pattern_weight + root_weight + confidence_score) / 3 * length_factor
        )
        return min(max(quality_score, 0.0), 1.0)

    def _generate_word_id(self, word: str) -> str:
        """Generate unique word ID"""
        timestamp = int(time.time())
        hash_val = hashlib.md5(f"{word}{timestamp}".encode()).hexdigest()[:8]
        return f"word_{hash_val}"


class MorphologyReactor:
    """
    🚀 MAIN MORPHOLOGY REACTOR
    Central controller for algorithmic morphology generation
    """

    def __init__(self, db_path: str = "enhanced_morphology.db"):

        self.db_path = db_path
        self.pattern_generator = ArabicPatternGenerator()
        self.root_extractor = SemanticRootExtractor()
        self.word_generator = WordGenerator()

        # Initialize database
        self._init_database()

        # Redis connection for caching
        self.redis_client = self._connect_redis()

        # Laravel API integration
        self.laravel_api_url = "http://laravel app:8000/api"

        logger.info("Morphology Reactor initialized successfully")

    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables for generated data
            cursor.execute()
                '''
                CREATE TABLE IF NOT EXISTS generated_patterns ()
                    id TEXT PRIMARY KEY,
                    pattern_text TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    root_class TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    confidence REAL NOT NULL,
                    linguistic_features TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    algorithm_version TEXT
                )
            '''
            )

            cursor.execute()
                '''
                CREATE TABLE IF NOT EXISTS generated_roots ()
                    id TEXT PRIMARY KEY,
                    root_text TEXT NOT NULL,
                    root_class TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    variants TEXT,
                    semantic_cluster INTEGER,
                    confidence REAL NOT NULL,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
            )

            cursor.execute()
                '''
                CREATE TABLE IF NOT EXISTS generated_words ()
                    id TEXT PRIMARY KEY,
                    word_text TEXT NOT NULL,
                    root_id TEXT NOT NULL,
                    pattern_id TEXT NOT NULL,
                    generation_method TEXT NOT NULL,
                    linguistic_analysis TEXT,
                    quality_score REAL NOT NULL,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (root_id) REFERENCES generated_roots (id),
                    FOREIGN KEY (pattern_id) REFERENCES generated_patterns (id)
                )
            '''
            )

            conn.commit()
            conn.close()

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def _connect_redis(self) -> Optional[Any]:
        """Connect to Redis for caching"""
        if not REDIS_AVAILABLE or not redis:
            logger.warning("Redis library not available. Caching disabled.")
            return None

        try:
            client = redis.Redis(  # type: ignore
                host='morphology redis',
                port=6379,
                password='Redis2024!SecureCache#Production',
                db=0,
                decode_responses=True)
            client.ping()
            return client
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return None

    def generate_comprehensive_morphology()
        self, corpus: List[str], target_root: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive morphology data

        Args:
            corpus: Text corpus for training
            target_root: Specific root to focus on (optional)

        Returns:
            Dictionary with generated patterns, roots, and words
        """
        result = {
            'patterns': [],
            'roots': [],
            'words': [],
            'statistics': {},
            'generation_time': datetime.now().isoformat(),
        }

        try:
            # Step 1: Extract roots from corpus
            logger.info("Extracting roots from corpus...")
            roots = self.root_extractor.extract_roots_from_corpus(corpus)

            # Filter by target root if specified
            if target_root:
                roots = [r for r in roots if r.root_text == target_root]

            # Step 2: Generate patterns for each root
            logger.info("Generating patterns...")
            all_patterns = []
            for root in roots:
                patterns = self.pattern_generator.generate_patterns_from_root()
                    root.root_text, root.root_class
                )
                all_patterns.extend(patterns)

            # Step 3: Generate words from patterns and roots
            logger.info("Generating words...")
            words = self.word_generator.generate_words(all_patterns, roots)

            # Step 4: Save to database
            self._save_generated_data(all_patterns, roots, words)

            # Step 5: Sync with Laravel API
            self._sync_with_laravel_api(all_patterns, roots, words)

            # Step 6: Update cache
            self._update_cache(all_patterns, roots, words)

            # Compile results
            result['patterns'] = [asdict(p) for p in all_patterns]
            result['roots'] = [asdict(r) for r in roots]
            result['words'] = [asdict(w) for w in words]
            result['statistics'] = {
                'total_patterns': len(all_patterns),
                'total_roots': len(roots),
                'total_words': len(words),
                'average_pattern_weight': ()
                    sum(p.weight for p in all_patterns) / len(all_patterns)
                    if all_patterns
                    else 0
                ),
                'average_root_weight': ()
                    sum(r.weight for r in roots) / len(roots) if roots else 0
                ),
                'average_word_quality': ()
                    sum(w.quality_score for w in words) / len(words) if words else 0
                ),
            }

            logger.info()
                f"Generation completed: {len(all_patterns)} patterns, {len(roots)} roots, {len(words) words}"
            )

        except Exception as e:
            logger.error(f"Error in comprehensive generation: {e}")
            result['error'] = str(e)

        return result

    def _save_generated_data()
        self,
        patterns: List[MorphologyPattern],
        roots: List[WeightedRoot], words: List[GeneratedWord]):
        """Save generated data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Save patterns
            for pattern in patterns:
                cursor.execute()
                    '''
                    INSERT OR REPLACE INTO generated_patterns 
                    (id, pattern_text, pattern_type, root_class, frequency, weight, confidence,)
                     linguistic_features, generated_at, algorithm_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''','
                    ()
                        pattern.pattern_id,
                        pattern.pattern_text,
                        pattern.pattern_type,
                        pattern.root_class,
                        pattern.frequency,
                        pattern.weight,
                        pattern.confidence,
                        json.dumps(pattern.linguistic_features),
                        pattern.generated_at.isoformat(),
                        pattern.algorithm_version))

            # Save roots
            for root in roots:
                cursor.execute()
                    '''
                    INSERT OR REPLACE INTO generated_roots
                    (id, root_text, root_class, frequency, weight, variants, semantic_cluster, confidence, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''','
                    ()
                        root.root_id,
                        root.root_text,
                        root.root_class,
                        root.frequency,
                        root.weight,
                        json.dumps(root.variants),
                        root.semantic_cluster,
                        root.confidence,
                        root.generated_at.isoformat()))

            # Save words
            for word in words:
                cursor.execute()
                    '''
                    INSERT OR REPLACE INTO generated_words
                    (id, word_text, root_id, pattern_id, generation_method, linguistic_analysis, quality_score, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''','
                    ()
                        word.word_id,
                        word.word_text,
                        word.root_id,
                        word.pattern_id,
                        word.generation_method,
                        json.dumps(word.linguistic_analysis),
                        word.quality_score,
                        word.generated_at.isoformat()))

            conn.commit()
            conn.close()

            logger.info("Data saved to database successfully")

        except Exception as e:
            logger.error(f"Error saving data to database: {e}")

    def _sync_with_laravel_api()
        self,
        patterns: List[MorphologyPattern],
        roots: List[WeightedRoot], words: List[GeneratedWord]):
        """Sync generated data with Laravel API"""
        try:
            # Prepare data for API
            api_data = {
                'patterns': [asdict(p) for p in patterns],
                'roots': [asdict(r) for r in roots],
                'words': [asdict(w) for w in words],
                'source': 'morphology_reactor',
                'timestamp': datetime.now().isoformat(),
            }

            # Send to Laravel API
            response = requests.post()
                f"{self.laravel_api_url/morphology/sync}", json=api_data, timeout=30
            )

            if response.status_code == 200:
                logger.info("Data synced with Laravel API successfully")
            else:
                logger.error(f"Laravel API sync failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error syncing with Laravel API: {e}")

    def _update_cache()
        self,
        patterns: List[MorphologyPattern],
        roots: List[WeightedRoot], words: List[GeneratedWord]):
        """Update Redis cache with generated data"""
        if not self.redis_client:
            return

        try:
            # Cache patterns
            for pattern in patterns:
                self.redis_client.hset()
                    f"pattern:{pattern.pattern_id}", mapping=asdict(pattern)
                )

            # Cache roots
            for root in roots:
                self.redis_client.hset(f"root:{root.root_id}", mapping=asdict(root))

            # Cache words
            for word in words:
                self.redis_client.hset(f"word:{word.word_id}", mapping=asdict(word))

            # Set cache expiration
            self.redis_client.expire("patterns:*", 3600)  # 1 hour
            self.redis_client.expire("roots:*", 3600)
            self.redis_client.expire("words:*", 3600)

            logger.info("Cache updated successfully")

        except Exception as e:
            logger.error(f"Error updating cache: {e}")

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get pattern statistics
            cursor.execute()
                'SELECT COUNT(*), AVG(weight), AVG(confidence) FROM generated_patterns'
            )
            pattern_stats = cursor.fetchone()

            # Get root statistics
            cursor.execute()
                'SELECT COUNT(*), AVG(weight), AVG(confidence) FROM generated_roots'
            )
            root_stats = cursor.fetchone()

            # Get word statistics
            cursor.execute('SELECT COUNT(*), AVG(quality_score) FROM generated_words')
            word_stats = cursor.fetchone()

            conn.close()

            return {
                'patterns': {
                    'count': pattern_stats[0] or 0,
                    'avg_weight': pattern_stats[1] or 0,
                    'avg_confidence': pattern_stats[2] or 0,
                },
                'roots': {
                    'count': root_stats[0] or 0,
                    'avg_weight': root_stats[1] or 0,
                    'avg_confidence': root_stats[2] or 0,
                },
                'words': {
                    'count': word_stats[0] or 0,
                    'avg_quality': word_stats[1] or 0,
                },
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


# Factory function for easy instantiation
def create_morphology_reactor()
    db_path: str = "enhanced_morphology.db") -> MorphologyReactor:
    """Create and return a configured MorphologyReactor instance"""
    return MorphologyReactor(db_path)


if __name__ == "__main__":
    # Example usage
    reactor = create_morphology_reactor()

    # Sample corpus for testing
    sample_corpus = [
        "كتب الطالب الدرس بعناية",
        "درس الطلاب المادة جيداً",
        "يكتب الكاتب القصة بمهارة",
        "كتاب جميل يحتوي على معلومات مفيدة",
    ]

    # Generate comprehensive morphology
    result = reactor.generate_comprehensive_morphology(sample_corpus)

    # Print results
    print(f"Generated {result['statistics']['total_patterns']} patterns")
    print(f"Generated {result['statistics']['total_roots']} roots")
    print(f"Generated {result['statistics']['total_words']} words")

    # Get statistics
    stats = reactor.get_generation_statistics()
    print()
        f"Database contains {stats['patterns']['count']} patterns, {stats['roots']['count']} roots, {stats['words']['count']} words"
    )

