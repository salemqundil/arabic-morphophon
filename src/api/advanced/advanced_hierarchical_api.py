"""
Advanced Hierarchical Arabic Morphophonological Analysis API
واجهة برمجة التطبيقات المتقدمة للتحليل الصرفي الصوتي العربي الهرمي

This Flask application integrates the complete hierarchical architecture
with neural embeddings, soft-logic rules, and graph autoencoders while
maintaining ZERO VIOLATIONS compatibility with the existing system.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data logging
import_data os
import_data sys
import_data time
from datetime import_data datetime
from pathlib import_data Path
from typing import_data Any, Dict, List, Optional, Tuple, Union

# Flask import_datas
from flask import_data Flask, jsonify, render_template_string, request
from flask_cors import_data CORS

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Core engine import_datas with fallback
try:
    from arabic_morphophon.integrator import_data AnalysisLevel, MorphophonologicalEngine
    TRADITIONAL_ENGINE_AVAILABLE = True
    logger.info("✅ Traditional MorphophonologicalEngine import_dataed successfully")
except ImportError as e:
    TRADITIONAL_ENGINE_AVAILABLE = False
    MorphophonologicalEngine = None
    AnalysisLevel = None
    logger.error(f"❌ Traditional engine import_data failed: {e}")

# Advanced components import_datas with graceful fallback
ADVANCED_COMPONENTS_AVAILABLE = False
try:
    from arabic_morphophon.advanced.graph_autoencoder import_data GraphAutoencoder
    from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
    from arabic_morphophon.advanced.soft_logic_rules import_data AdvancedRulesEngine
    from arabic_morphophon.advanced.syllabic_unit_embeddings import_data SyllabicUnitEmbedding
    ADVANCED_COMPONENTS_AVAILABLE = True
    logger.info("✅ Advanced neural components import_dataed successfully")
except ImportError as e:
    PhonemeVowelEmbed = None
    SyllabicUnitEmbedding = None
    GraphAutoencoder = None
    AdvancedRulesEngine = None
    logger.warning(f"⚠️ Advanced components not available: {e}")

# Flask app creation
app = Flask(__name__)
app.config.update({
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'arabic-advanced-morphophon-dev-key'),
    'JSON_AS_ASCII': False,  # Support Arabic Unicode
    'JSONIFY_PRETTYPRINT_REGULAR': True
})

# Enable CORS for cross-origin requests
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

class AdvancedArabicAnalysisSystem:
    """
    Advanced Arabic Analysis System integrating all hierarchical components
    النظام المتقدم للتحليل العربي المتكامل مع جميع المكونات الهرمية
    """
    
    def __init__(self):
        """Initialize the advanced analysis system"""
        self.traditional_engine = None
        self.phoneme_embedder = None
        self.syllabic_unit_embedder = None
        self.graph_autoencoder = None
        self.rules_engine = None
        
        # System configuration
        self.config = {
            'enable_traditional': TRADITIONAL_ENGINE_AVAILABLE,
            'enable_advanced': ADVANCED_COMPONENTS_AVAILABLE,
            'enable_neural': ADVANCED_COMPONENTS_AVAILABLE,
            'hierarchical_analysis': True,
            'soft_logic_validation': True,
            'graph_encoding': True
        }
        
        # Performance statistics
        self.stats = {
            'total_requests': 0,
            'successful_analyses': 0,
            'errors': 0,
            'begin_time': datetime.now(),
            'average_processing_time': 0.0,
            'component_usage': {
                'traditional': 0,
                'phoneme_embeddings': 0,
                'syllabic_unit_embeddings': 0,
                'graph_autoencoder': 0,
                'rules_validation': 0
            }
        }
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all available components"""
        
        # Traditional engine
        if TRADITIONAL_ENGINE_AVAILABLE and MorphophonologicalEngine:
            try:
                self.traditional_engine = MorphophonologicalEngine()
                logger.info("✅ Traditional engine initialized")
            except Exception as e:
                logger.error(f"❌ Traditional engine initialization failed: {e}")
                
        # Advanced components
        if ADVANCED_COMPONENTS_AVAILABLE:
            try:
                # Phoneme and vowel embeddings (Levels 0-1)
                if PhonemeVowelEmbed:
                    self.phoneme_embedder = PhonemeVowelEmbed(
                        n_phon=29, n_vowel=12, d=8, enable_neural=True
                    )
                    logger.info("✅ Phoneme embedder initialized")
                    
                # SyllabicUnit embeddings (Level 2)
                if SyllabicUnitEmbedding:
                    self.syllabic_unit_embedder = SyllabicUnitEmbedding(
                        max_patterns=64, d=16, enable_neural=True
                    )
                    logger.info("✅ SyllabicUnit embedder initialized")
                    
                # Graph autoencoder (Levels 3-7)
                if GraphAutoencoder:
                    self.graph_autoencoder = GraphAutoencoder(
                        num_features=32, hidden_dim=128, latent_dim=64
                    )
                    logger.info("✅ Graph autoencoder initialized")
                    
                # Soft-logic rules engine
                if AdvancedRulesEngine:
                    self.rules_engine = AdvancedRulesEngine(enable_neural=True)
                    logger.info("✅ Rules engine initialized")
                    
            except Exception as e:
                logger.error(f"❌ Advanced components initialization failed: {e}")
                
    def analyze_traditional(self, text: str, level: str = "COMPREHENSIVE") -> Dict[str, Any]:
        """
        Perform traditional morphophonological analysis
        
        Args:
            text: Arabic text to analyze
            level: Analysis level
            
        Returns:
            Traditional analysis results
        """
        if not self.traditional_engine:
            return {
                'error': 'Traditional engine not available',
                'fallback': True
            }
            
        try:
            # Convert level string to enum
            if AnalysisLevel and hasattr(AnalysisLevel, level.upper()):
                analysis_level = getattr(AnalysisLevel, level.upper())
            else:
                analysis_level = None
                
            result = self.traditional_engine.analyze(text, analysis_level)
            self.stats['component_usage']['traditional'] += 1
            
            return {
                'original_text': getattr(result, 'original_text', text),
                'analysis_level': str(getattr(result, 'analysis_level', level)),
                'confidence_score': getattr(result, 'confidence_score', 0.0),
                'identified_roots': getattr(result, 'identified_roots', []),
                'detected_patterns': getattr(result, 'detected_patterns', []),
                'phonological_output': getattr(result, 'phonological_output', text),
                'syllabic_unit_count': getattr(result, 'syllabic_unit_count', 0),
                'processing_time': getattr(result, 'processing_time', 0.0),
                'warnings': getattr(result, 'warnings', []),
                'errors': getattr(result, 'errors', [])
            }
            
        except Exception as e:
            logger.error(f"Traditional analysis error: {e}")
            return {
                'error': str(e),
                'fallback': True
            }
            
    def analyze_hierarchical(self, text: str, 
                           enable_embeddings: bool = True,
                           enable_graph: bool = True,
                           enable_rules: bool = True) -> Dict[str, Any]:
        """
        Perform full hierarchical analysis using all components
        
        Args:
            text: Arabic text to analyze
            enable_embeddings: Whether to use embedding components
            enable_graph: Whether to use graph autoencoder
            enable_rules: Whether to validate with soft-logic rules
            
        Returns:
            Comprehensive hierarchical analysis
        """
        begin_time = time.time()
        analysis_result = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'components_used': [],
            'levels': {}
        }
        
        try:
            # Level 0-1: Phoneme and Vowel Analysis
            if enable_embeddings and self.phoneme_embedder:
                phoneme_analysis = self.phoneme_embedder.analyze_phonological_features(text)
                analysis_result['levels']['phoneme_vowel'] = phoneme_analysis
                analysis_result['components_used'].append('phoneme_embeddings')
                self.stats['component_usage']['phoneme_embeddings'] += 1
                
            # Level 2: SyllabicUnit Analysis
            if enable_embeddings and self.syllabic_unit_embedder:
                syllabic_unit_analysis = self.syllabic_unit_embedder.analyze_prosody(text)
                analysis_result['levels']['syllabic_unit'] = syllabic_unit_analysis
                analysis_result['components_used'].append('syllabic_unit_embeddings')
                self.stats['component_usage']['syllabic_unit_embeddings'] += 1
                
            # Level 3-7: Traditional morphophonological analysis
            traditional_result = self.analyze_traditional(text)
            if 'error' not in traditional_result:
                analysis_result['levels']['morphophonological'] = traditional_result
                analysis_result['components_used'].append('traditional_engine')
                
            # Graph encoding (if requested)
            if enable_graph and self.graph_autoencoder:
                try:
                    # Create feature matrix from analysis results
                    features = self._create_feature_matrix(analysis_result)
                    # Create simple adjacency matrix for demonstration
                    adjacency = self._create_adjacency_matrix(len(features))
                    
                    # Encode with graph autoencoder
                    embeddings = self.graph_autoencoder.embed(features, adjacency)
                    analysis_result['levels']['graph_embeddings'] = {
                        'embeddings': embeddings[:5] if len(embeddings) > 5 else embeddings,  # Limit output
                        'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                        'num_nodes': len(embeddings)
                    }
                    analysis_result['components_used'].append('graph_autoencoder')
                    self.stats['component_usage']['graph_autoencoder'] += 1
                except Exception as e:
                    logger.warning(f"Graph encoding failed: {e}")
                    
            # Soft-logic rules validation
            if enable_rules and self.rules_engine:
                try:
                    violations = self.rules_engine.validate_text(
                        text, 
                        traditional_result if 'error' not in traditional_result else {},
                        {}
                    )
                    analysis_result['levels']['rule_validation'] = {
                        'violations': [v.to_dict() for v in violations],
                        'violation_count': len(violations),
                        'rules_applied': len(self.rules_engine.rules)
                    }
                    analysis_result['components_used'].append('rules_validation')
                    self.stats['component_usage']['rules_validation'] += 1
                except Exception as e:
                    logger.warning(f"Rules validation failed: {e}")
                    
            # Calculate processing time
            processing_time = time.time() - begin_time
            analysis_result['processing_time'] = processing_time
            analysis_result['success'] = True
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Hierarchical analysis error: {e}")
            return {
                'text': text,
                'error': str(e),
                'success': False,
                'processing_time': time.time() - begin_time
            }
            
    def _create_feature_matrix(self, analysis: Dict) -> List[List[float]]:
        """Create feature matrix from analysis results for graph encoding"""
        # Simple feature extraction for demonstration
        features = []
        
        # Add phoneme features if available
        if 'phoneme_vowel' in analysis.get('levels', {}):
            phoneme_data = analysis['levels']['phoneme_vowel']
            for emb in phoneme_data.get('phoneme_embeddings', [])[:5]:  # Limit to 5
                features.append(emb)
                
        # Add syllabic_unit features if available
        if 'syllabic_unit' in analysis.get('levels', {}):
            syllabic_unit_data = analysis['levels']['syllabic_unit']
            # Create features from syllabic weights and patterns
            for syl in syllabic_unit_data.get('syllabic_units', [])[:3]:  # Limit to 3
                feature = [
                    syl.get('weight', 0.0),
                    syl.get('length', 0.0),
                    syl.get('vowel_count', 0.0),
                    syl.get('consonant_count', 0.0)
                ]
                # Pad to 32 dimensions
                feature.extend([0.0] * (32 - len(feature)))
                features.append(feature)
                
        # Ensure we have at least some features
        if not features:
            # Create dummy features
            for i in range(3):
                features.append([0.0] * 32)
                
        return features
        
    def _create_adjacency_matrix(self, num_nodes: int) -> List[List[float]]:
        """Create simple adjacency matrix for demonstration"""
        # Create identity matrix with some connections
        adjacency = []
        for i in range(num_nodes):
            row = [0.0] * num_nodes
            row[i] = 1.0  # Self-connection
            # Add connection to next node
            if i + 1 < num_nodes:
                row[i + 1] = 1.0
            adjacency.append(row)
            
        return adjacency
        
    def validate_rules(self, text: str, analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate text against soft-logic rules
        
        Args:
            text: Arabic text to validate
            analysis: Optional existing analysis results
            
        Returns:
            Rule validation results
        """
        if not self.rules_engine:
            return {
                'error': 'Rules engine not available',
                'violations': [],
                'success': False
            }
            
        try:
            if not analysis:
                analysis = self.analyze_traditional(text)
                
            violations = self.rules_engine.validate_text(text, analysis, {})
            
            return {
                'text': text,
                'violations': [v.to_dict() for v in violations],
                'violation_count': len(violations),
                'rules_applied': len([r for r in self.rules_engine.rules if r.enabled]),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Rules validation error: {e}")
            return {
                'error': str(e),
                'violations': [],
                'success': False
            }
            
    def repair_violations(self, text: str, violations_data: List[Dict]) -> Dict[str, Any]:
        """
        Attempt to repair violations using hard constraints
        
        Args:
            text: Original text
            violations_data: List of violation dictionaries
            
        Returns:
            Repair results
        """
        if not self.rules_engine:
            return {
                'error': 'Rules engine not available',
                'original_text': text,
                'repaired_text': text,
                'changes': []
            }
            
        try:
            # Convert violation dictionaries back to RuleViolation objects
            from arabic_morphophon.advanced.soft_logic_rules import_data RuleViolation
            violations = []
            for v_data in violations_data:
                violation = RuleViolation(
                    rule_name=v_data.get('rule_name', ''),
                    rule_type=v_data.get('rule_type', ''),
                    severity=v_data.get('severity', 0.0),
                    description=v_data.get('description', ''),
                    context=v_data.get('context', {}),
                    suggested_fix=v_data.get('suggested_fix')
                )
                violations.append(violation)
                
            repaired_text, changes = self.rules_engine.hard_repair(text, violations)
            
            return {
                'original_text': text,
                'repaired_text': repaired_text,
                'changes': changes,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Repair error: {e}")
            return {
                'error': str(e),
                'original_text': text,
                'repaired_text': text,
                'changes': []
            }
            
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'system_name': 'Advanced Hierarchical Arabic Morphophonological Analysis System',
            'version': '1.0.0',
            'components': {
                'traditional_engine': TRADITIONAL_ENGINE_AVAILABLE,
                'advanced_components': ADVANCED_COMPONENTS_AVAILABLE,
                'phoneme_embeddings': self.phoneme_embedder is not None,
                'syllabic_unit_embeddings': self.syllabic_unit_embedder is not None,
                'graph_autoencoder': self.graph_autoencoder is not None,
                'rules_engine': self.rules_engine is not None
            },
            'configuration': self.config,
            'statistics': self.stats,
            'capabilities': {
                'traditional_analysis': True,
                'hierarchical_analysis': ADVANCED_COMPONENTS_AVAILABLE,
                'neural_embeddings': ADVANCED_COMPONENTS_AVAILABLE,
                'graph_encoding': ADVANCED_COMPONENTS_AVAILABLE,
                'soft_logic_validation': ADVANCED_COMPONENTS_AVAILABLE,
                'hard_repair': ADVANCED_COMPONENTS_AVAILABLE
            }
        }
        
        # Add component-specific info
        if self.phoneme_embedder:
            info['phoneme_embedder_info'] = self.phoneme_embedder.get_info()
        if self.syllabic_unit_embedder:
            info['syllabic_unit_embedder_info'] = self.syllabic_unit_embedder.get_info()
        if self.graph_autoencoder:
            info['graph_autoencoder_info'] = self.graph_autoencoder.get_info()
        if self.rules_engine:
            info['rules_engine_info'] = self.rules_engine.get_info()
            
        return info

# Initialize the advanced analysis system
analysis_system = AdvancedArabicAnalysisSystem()

# Flask Routes
@app.route('/')
def home():
    """Main API information page"""
    return jsonify({
        'message': 'Advanced Hierarchical Arabic Morphophonological Analysis API',
        'version': '1.0.0',
        'status': 'operational',
        'components_available': analysis_system.config,
        'endpoints': [
            'GET / - API information',
            'GET /api/health - Health check',
            'POST /api/analyze/traditional - Traditional analysis',
            'POST /api/analyze/hierarchical - Full hierarchical analysis',
            'POST /api/validate/rules - Soft-logic rules validation',
            'POST /api/repair - Hard constraint repair',
            'POST /api/batch - Batch processing',
            'GET /api/info - System information',
            'GET /api/stats - Performance statistics'
        ]
    })

@app.route('/api/health')
def health_check():
    """System health check"""
    uptime = (datetime.now() - analysis_system.stats['begin_time']).total_seconds()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': uptime,
        'components_status': analysis_system.config,
        'total_requests': analysis_system.stats['total_requests'],
        'success_rate': (
            analysis_system.stats['successful_analyses'] / 
            max(analysis_system.stats['total_requests'], 1)
        )
    })

@app.route('/api/analyze/traditional', methods=['POST'])
def analyze_traditional():
    """Traditional morphophonological analysis endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field in request',
                'success': False
            }), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({
                'error': 'Empty text provided',
                'success': False
            }), 400
            
        level = data.get('level', 'COMPREHENSIVE')
        
        # Update stats
        analysis_system.stats['total_requests'] += 1
        
        # Perform analysis
        result = analysis_system.analyze_traditional(text, level)
        
        if 'error' not in result:
            analysis_system.stats['successful_analyses'] += 1
            
        return jsonify({
            'analysis': result,
            'success': 'error' not in result,
            'analysis_type': 'traditional'
        })
        
    except Exception as e:
        analysis_system.stats['errors'] += 1
        logger.error(f"Traditional analysis endpoint error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/analyze/hierarchical', methods=['POST'])
def analyze_hierarchical():
    """Full hierarchical analysis endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field in request',
                'success': False
            }), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({
                'error': 'Empty text provided',
                'success': False
            }), 400
            
        # Options
        enable_embeddings = data.get('enable_embeddings', True)
        enable_graph = data.get('enable_graph', True)
        enable_rules = data.get('enable_rules', True)
        
        # Update stats
        analysis_system.stats['total_requests'] += 1
        
        # Perform hierarchical analysis
        result = analysis_system.analyze_hierarchical(
            text, enable_embeddings, enable_graph, enable_rules
        )
        
        if result.get('success', False):
            analysis_system.stats['successful_analyses'] += 1
            
        return jsonify({
            'analysis': result,
            'success': result.get('success', False),
            'analysis_type': 'hierarchical'
        })
        
    except Exception as e:
        analysis_system.stats['errors'] += 1
        logger.error(f"Hierarchical analysis endpoint error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/validate/rules', methods=['POST'])
def validate_rules():
    """Soft-logic rules validation endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field in request',
                'success': False
            }), 400
            
        text = data['text'].strip()
        analysis = data.get('analysis')  # Optional pre-computed analysis
        
        # Update stats
        analysis_system.stats['total_requests'] += 1
        
        # Validate rules
        result = analysis_system.validate_rules(text, analysis)
        
        if result.get('success', False):
            analysis_system.stats['successful_analyses'] += 1
            
        return jsonify(result)
        
    except Exception as e:
        analysis_system.stats['errors'] += 1
        logger.error(f"Rules validation endpoint error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/repair', methods=['POST'])
def repair_violations():
    """Hard constraint repair endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'violations' not in data:
            return jsonify({
                'error': 'Missing text or violations field in request',
                'success': False
            }), 400
            
        text = data['text'].strip()
        violations = data['violations']
        
        # Update stats
        analysis_system.stats['total_requests'] += 1
        
        # Repair violations
        result = analysis_system.repair_violations(text, violations)
        
        if 'error' not in result:
            analysis_system.stats['successful_analyses'] += 1
            
        return jsonify(result)
        
    except Exception as e:
        analysis_system.stats['errors'] += 1
        logger.error(f"Repair endpoint error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/batch', methods=['POST'])
def batch_process():
    """Batch processing endpoint"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing texts field in request',
                'success': False
            }), 400
            
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts field must be a list',
                'success': False
            }), 400
            
        analysis_type = data.get('type', 'traditional')  # traditional or hierarchical
        max_batch_size = 10  # Limit batch size
        
        if len(texts) > max_batch_size:
            return jsonify({
                'error': f'Batch size exceeds maximum of {max_batch_size}',
                'success': False
            }), 400
            
        # Process batch
        results = []
        for i, text in enumerate(texts):
            try:
                if analysis_type == 'hierarchical':
                    result = analysis_system.analyze_hierarchical(text)
                else:
                    result = analysis_system.analyze_traditional(text)
                    
                results.append({
                    'index': i,
                    'text': text,
                    'analysis': result,
                    'success': 'error' not in result
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'text': text,
                    'error': str(e),
                    'success': False
                })
                
        # Update stats
        analysis_system.stats['total_requests'] += len(texts)
        successful = len([r for r in results if r.get('success', False)])
        analysis_system.stats['successful_analyses'] += successful
        
        return jsonify({
            'results': results,
            'batch_size': len(texts),
            'successful': successful,
            'failed': len(texts) - successful,
            'success': True
        })
        
    except Exception as e:
        analysis_system.stats['errors'] += 1
        logger.error(f"Batch processing error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/info')
def system_info():
    """Get comprehensive system information"""
    return jsonify(analysis_system.get_system_info())

@app.route('/api/stats')
def get_statistics():
    """Get performance statistics"""
    return jsonify(analysis_system.stats)

# Error processrs
@app.errorprocessr(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation for available endpoints'
    }), 404

@app.errorprocessr(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    print("🚀 Begining Advanced Hierarchical Arabic Morphophonological Analysis API")
    print("=" * 70)
    print(f"✅ Traditional Engine: {'Available' if TRADITIONAL_ENGINE_AVAILABLE else 'Not Available'}")
    print(f"✅ Advanced Components: {'Available' if ADVANCED_COMPONENTS_AVAILABLE else 'Not Available'}")
    print(f"🌐 Server begining on http://0.0.0.0:5000")
    print("")
    print("📋 Available Endpoints:")
    print("  GET  /                              - API information")
    print("  GET  /api/health                    - Health check")
    print("  POST /api/analyze/traditional       - Traditional analysis")
    print("  POST /api/analyze/hierarchical      - Full hierarchical analysis")
    print("  POST /api/validate/rules            - Soft-logic rules validation")
    print("  POST /api/repair                    - Hard constraint repair")
    print("  POST /api/batch                     - Batch processing")
    print("  GET  /api/info                      - System information")
    print("  GET  /api/stats                     - Performance statistics")
    print("")
    print("🎯 Ready for production deployment with ZERO VIOLATIONS compatibility!")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
