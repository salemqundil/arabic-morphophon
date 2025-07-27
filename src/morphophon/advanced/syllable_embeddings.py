"""
SyllabicUnit Embeddings for Arabic Morphophonological Analysis
تضمينات المقاطع للتحليل الصرفي الصوتي العربي

Implements Level 2 syllabic_unit embeddings from the hierarchical architecture:
- SyllabicUnit pattern embeddings (≤64 patterns): 16D specialized for Arabic syllabic_units
- CV pattern recognition and encoding
- Prosodic weight calculations
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data math
import_data re
from typing import_data Dict, List, Optional, Tuple, Union

try:
    import_data torch
    import_data torch.nn as nn
    import_data torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore

# Arabic cv patterns (CV notation)
ARABIC_SYLLABIC_UNIT_PATTERNS = {
    # Basic patterns
    'CV': 1,     # كَ (ka)
    'CVC': 2,    # كَتْ (kat) 
    'CVV': 3,    # كا (kaa)
    'CVVC': 4,   # كاتْ (kaat)
    'CVCC': 5,   # كَتْبْ (katb)
    'CVVCC': 6,  # كاتبْ (kaatb)
    
    # Extended patterns
    'V': 7,      # أَ (a)
    'VC': 8,     # أَتْ (at)
    'VV': 9,     # آ (aa)
    'VVC': 10,   # آتْ (aat)
    'VCC': 11,   # أَتْبْ (atb)
    'VVCC': 12,  # آتبْ (aatb)
    
    # Complex patterns
    'CVCV': 13,    # كَتَبَ (kataba)
    'CVCVC': 14,   # كَتَبْتُ (katabtu)
    'CVVCV': 15,   # كاتَبَ (kaataba)
    'CVVCVC': 16,  # كاتَبْتُ (kaatabtu)
    
    # Gemination patterns
    'CVCC': 17,    # كَبَّ (kabb) - with shadda
    'CVVCV': 18,   # كابَ (kaaba)
    
    '<PAD>': 0,    # Padding
    '<UNK>': 63    # Unknown pattern
}

# Prosodic weights for syllabic types
PROSODIC_WEIGHTS = {
    'light': ['CV', 'V'],           # Light syllabic_units (1 mora)
    'heavy': ['CVC', 'CVV', 'VC', 'VV', 'VVC'],  # Heavy syllabic_units (2 morae)
    'superheavy': ['CVCC', 'CVVC', 'VCC', 'VVCC'],  # Superheavy (3+ morae)
}

# Stress assignment rules for Arabic
STRESS_RULES = {
    'ultimate': ['CVCC', 'CVVC'],      # Final syllabic stress
    'penultimate': ['CVC', 'CVV'],     # Penultimate stress  
    'antepenultimate': ['CV']          # Antepenultimate stress
}

class SyllabicUnitEmbedding:
    """
    SyllabicUnit Embedding Layer (Level 2)
    طبقة تضمين المقاطع (المستوى الثاني)
    
    Implements specialized embeddings for Arabic cv patterns
    with prosodic weight and stress pattern awareness.
    """
    
    def __init__(self, max_patterns: int = 64, d: int = 16, 
                 enable_neural: bool = True, enable_lstm: bool = False):
        """
        Initialize syllabic_unit embeddings
        
        Args:
            max_patterns: Maximum number of cv patterns (default 64)
            d: Embedding dimension (default 16)
            enable_neural: Whether to use neural networks
            enable_lstm: Whether to use LSTM for long sequences
        """
        self.max_patterns = max_patterns
        self.d = d
        self.enable_neural = enable_neural and TORCH_AVAILABLE
        self.enable_lstm = enable_lstm and self.enable_neural
        
        if self.enable_neural:
            self._init_neural_components()
        else:
            self._init_fallback_components()
            
    def _init_neural_components(self):
        """Initialize PyTorch neural components"""
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise RuntimeError("PyTorch not available for neural components")
            
        # SyllabicUnit pattern embedding
        self.pattern_emb = nn.Embedding(
            self.max_patterns + 1, self.d, padding_idx=0
        )
        
        # Optional LSTM for sequence modeling
        if self.enable_lstm:
            self.lstm = nn.LSTM(
                input_size=self.d,
                hidden_size=self.d // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
        # Initialize with prosodic features
        self._init_prosodic_features()
        
    def _init_fallback_components(self):
        """Initialize fallback components without PyTorch"""
        self.pattern_embeddings = {}
        
        # Initialize with deterministic values based on prosodic features
        for pattern, idx in ARABIC_SYLLABIC_UNIT_PATTERNS.items():
            if idx > 0:  # Skip padding
                embedding = self._create_prosodic_embedding(pattern)
                self.pattern_embeddings[pattern] = embedding
                
    def _create_prosodic_embedding(self, pattern: str) -> List[float]:
        """Create embedding based on prosodic features"""
        embedding = [0.0] * self.d
        
        # SyllabicUnit weight as first feature
        weight = self._get_prosodic_weight(pattern)
        embedding[0] = weight
        
        # Number of segments as second feature
        embedding[1] = len(pattern) / 6.0  # Normalized by max length
        
        # Vowel count as third feature
        vowel_count = pattern.count('V')
        embedding[2] = vowel_count / 3.0  # Normalized by max vowels
        
        # Consonant count as fourth feature
        consonant_count = pattern.count('C')
        embedding[3] = consonant_count / 4.0  # Normalized by max consonants
        
        # Fill remaining dimensions with pattern-specific values
        pattern_hash = hash(pattern) % 1000
        for i in range(4, self.d):
            embedding[i] = math.sin(pattern_hash + i) * 0.1
            
        return embedding
        
    def _get_prosodic_weight(self, pattern: str) -> float:
        """Get prosodic weight for a cv pattern"""
        if pattern in PROSODIC_WEIGHTS['light']:
            return 1.0  # Light
        elif pattern in PROSODIC_WEIGHTS['heavy']:
            return 2.0  # Heavy
        elif pattern in PROSODIC_WEIGHTS['superheavy']:
            return 3.0  # Superheavy
        else:
            return 1.5  # Default/unknown
            
    def _init_prosodic_features(self):
        """Initialize embeddings with prosodic awareness (neural mode)"""
        if not self.enable_neural or torch is None:
            return
            
        with torch.no_grad():
            # Initialize based on prosodic weights
            for pattern, idx in ARABIC_SYLLABIC_UNIT_PATTERNS.items():
                if idx > 0:  # Skip padding
                    weight = self._get_prosodic_weight(pattern)
                    self.pattern_emb.weight[idx, 0] = weight / 3.0  # Normalized
                    
                    # Segment count
                    self.pattern_emb.weight[idx, 1] = len(pattern) / 6.0
                    
                    # Vowel/consonant ratio
                    if len(pattern) > 0:
                        vowel_ratio = pattern.count('V') / len(pattern)
                        self.pattern_emb.weight[idx, 2] = vowel_ratio
                        
    def text_to_cv_pattern(self, text: str) -> List[str]:
        """
        Convert Arabic text to CV pattern representation
        
        Args:
            text: Arabic text
            
        Returns:
            List of CV patterns
        """
        # Simple CV pattern extraction (can be enhanced)
        patterns = []
        current_pattern = ""
        
        # Arabic vowel diacritics
        vowels = 'َُِاويً ٌٍْ'
        
        for char in text:
            if char.strip() == '':
                continue
                
            # Check if character is a vowel or consonant
            if char in vowels or char in 'اوي':
                current_pattern += 'V'
            elif char.isalpha():  # Arabic consonant
                current_pattern += 'C'
            else:
                # Word boundary or punctuation
                if current_pattern:
                    patterns.append(current_pattern)
                    current_pattern = ""
                    
        # Add final pattern if exists
        if current_pattern:
            patterns.append(current_pattern)
            
        return patterns
        
    def syllabic_analyze_text(self, text: str) -> List[Dict]:
        """
        SyllabicAnalyze Arabic text and extract patterns
        
        Args:
            text: Arabic text
            
        Returns:
            List of syllabic_unit dictionaries with patterns and features
        """
        cv_patterns = self.text_to_cv_pattern(text)
        syllabic_units = []
        
        for i, pattern in enumerate(cv_patterns):
            # Map to known patterns or use unknown
            if pattern in ARABIC_SYLLABIC_UNIT_PATTERNS:
                mapped_pattern = pattern
            else:
                # Try to map to closest known pattern
                mapped_pattern = self._map_to_known_pattern(pattern)
                
            syllabic_unit_info = {
                'pattern': mapped_pattern,
                'original_pattern': pattern,
                'position': i,
                'weight': self._get_prosodic_weight(mapped_pattern),
                'length': len(pattern),
                'vowel_count': pattern.count('V'),
                'consonant_count': pattern.count('C')
            }
            
            syllabic_units.append(syllabic_unit_info)
            
        return syllabic_units
        
    def _map_to_known_pattern(self, pattern: str) -> str:
        """Map unknown pattern to closest known pattern"""
        # Simple mapping based on structure
        if len(pattern) == 1:
            return 'V' if pattern == 'V' else 'CV'
        elif len(pattern) == 2:
            if pattern == 'CV':
                return 'CV'
            elif pattern == 'VC':
                return 'VC'
            elif pattern == 'VV':
                return 'VV'
            else:
                return 'CV'  # Default
        elif len(pattern) == 3:
            if 'V' in pattern:
                return 'CVC'
            else:
                return 'CVC'
        else:
            return '<UNK>'  # Unknown
            
    def embed_pattern(self, pattern: str) -> List[float]:
        """
        Get embedding for a cv pattern
        
        Args:
            pattern: CV pattern string
            
        Returns:
            Embedding vector as list of floats
        """
        if self.enable_neural and torch is not None:
            if pattern in ARABIC_SYLLABIC_UNIT_PATTERNS:
                idx = ARABIC_SYLLABIC_UNIT_PATTERNS[pattern]
                with torch.no_grad():
                    embedding = self.pattern_emb(torch.tensor([idx]))
                    return embedding[0].tolist()
            else:
                # Unknown pattern
                idx = ARABIC_SYLLABIC_UNIT_PATTERNS['<UNK>']
                with torch.no_grad():
                    embedding = self.pattern_emb(torch.tensor([idx]))
                    return embedding[0].tolist()
        else:
            return self.pattern_embeddings.get(pattern, [0.0] * self.d)
            
    def embed_sequence(self, patterns: List[str], use_lstm: bool = False) -> List[List[float]]:
        """
        Embed a sequence of cv patterns
        
        Args:
            patterns: List of CV patterns
            use_lstm: Whether to use LSTM processing
            
        Returns:
            List of embedding vectors
        """
        if self.enable_neural and torch is not None and use_lstm and self.enable_lstm:
            # Use LSTM for sequence modeling
            embeddings = []
            for pattern in patterns:
                emb = self.embed_pattern(pattern)
                embeddings.append(emb)
                
            # Convert to tensor and process with LSTM
            with torch.no_grad():
                seq_tensor = torch.tensor([embeddings])  # Add batch dimension
                lstm_output, _ = self.lstm(seq_tensor)
                return lstm_output[0].tolist()  
        else:
            # Simple embedding without LSTM
            return [self.embed_pattern(pattern) for pattern in patterns]
            
    def analyze_prosody(self, text: str) -> Dict:
        """
        Analyze prosodic structure of Arabic text
        
        Args:
            text: Arabic text
            
        Returns:
            Dictionary with prosodic analysis
        """
        syllabic_units = self.syllabic_analyze_text(text)
        
        # Calculate prosodic statistics
        total_weight = sum(syl['weight'] for syl in syllabic_units)
        avg_weight = total_weight / len(syllabic_units) if syllabic_units else 0
        
        weight_distribution = {
            'light': len([s for s in syllabic_units if s['weight'] == 1.0]),
            'heavy': len([s for s in syllabic_units if s['weight'] == 2.0]),
            'superheavy': len([s for s in syllabic_units if s['weight'] >= 3.0])
        }
        
        # Stress pattern prediction (simplified)
        stress_pattern = self._predict_stress(syllabic_units)
        
        return {
            'syllabic_units': syllabic_units,
            'syllabic_unit_count': len(syllabic_units),
            'total_prosodic_weight': total_weight,
            'average_weight': avg_weight,
            'weight_distribution': weight_distribution,
            'stress_pattern': stress_pattern,
            'rhythmic_type': self._classify_rhythm(weight_distribution)
        }
        
    def _predict_stress(self, syllabic_units: List[Dict]) -> List[int]:
        """Predict stress pattern for syllabic_units (simplified Arabic rules)"""
        if not syllabic_units:
            return []
            
        stress = [0] * len(syllabic_units)  # 0 = unstressed, 1 = stressed
        
        # Simplified Arabic stress rules
        if len(syllabic_units) == 1:
            stress[0] = 1  # Monosyllabic words are stressed
        elif len(syllabic_units) >= 2:
            # Check final syllabic weight
            final_syl = syllabic_units[-1]
            if final_syl['weight'] >= 2.0:  # Heavy or superheavy
                stress[-1] = 1  # Ultimate stress
            else:
                # Penultimate stress (common in Arabic)
                stress[-2] = 1
                
        return stress
        
    def _classify_rhythm(self, weight_dist: Dict) -> str:
        """Classify rhythmic type based on syllabic weight distribution"""
        total = sum(weight_dist.values())
        if total == 0:
            return 'unknown'
            
        light_ratio = weight_dist['light'] / total
        heavy_ratio = weight_dist['heavy'] / total
        
        if light_ratio > 0.7:
            return 'light_dominant'
        elif heavy_ratio > 0.5:
            return 'heavy_dominant'
        else:
            return 'mixed'
            
    def get_similarity(self, pattern1: str, pattern2: str) -> float:
        """
        Calculate similarity between two cv patterns
        
        Args:
            pattern1: First CV pattern
            pattern2: Second CV pattern
            
        Returns:
            Cosine similarity score
        """
        emb1 = self.embed_pattern(pattern1)
        emb2 = self.embed_pattern(pattern2)
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def get_info(self) -> Dict:
        """Get information about the syllabic_unit embedding layer"""
        return {
            'max_patterns': self.max_patterns,
            'embedding_dimension': self.d,
            'neural_enabled': self.enable_neural,
            'lstm_enabled': self.enable_lstm,
            'torch_available': TORCH_AVAILABLE,
            'supported_patterns': list(ARABIC_SYLLABIC_UNIT_PATTERNS.keys()),
            'prosodic_weights': PROSODIC_WEIGHTS
        }

# Example usage and testing
if __name__ == "__main__":
    print("🔤 Testing Arabic SyllabicUnit Embeddings")
    print("=" * 50)
    
    # Initialize embedder
    embedder = SyllabicUnitEmbedding(enable_neural=TORCH_AVAILABLE, enable_lstm=False)
    
    # Test text
    test_text = "كتاب"
    
    print(f"📝 Test text: {test_text}")
    
    # Analyze prosody
    prosody = embedder.analyze_prosody(test_text)
    
    print(f"🔤 SyllabicUnits found: {len(prosody['syllabic_units'])}")
    for i, syl in enumerate(prosody['syllabic_units']):
        print(f"  {i+1}. Pattern: {syl['pattern']}, Weight: {syl['weight']}")
        
    print(f"📊 Total prosodic weight: {prosody['total_prosodic_weight']}")
    print(f"📊 Rhythmic type: {prosody['rhythmic_type']}")
    print(f"🎵 Stress pattern: {prosody['stress_pattern']}")
    
    # Test pattern similarity
    sim = embedder.get_similarity('CV', 'CVC')
    print(f"🔗 Similarity between 'CV' and 'CVC': {sim:.3f}")
    
    print(f"\n📋 System info:")
    info = embedder.get_info()
    for key, value in info.items():
        if key not in ['supported_patterns', 'prosodic_weights']:
            print(f"  {key}: {value}")
