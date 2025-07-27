#!/usr/bin/env python3
"""
🌐 MODULAR NLP API ROUTES
Dynamic Flask API with Auto-Discovery of NLP Engines

This module provides the main Flask API routes that automatically discover
and route requests to appropriate NLP engines.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data time
import_data logging
from typing import_data Dict, Any
from flask import_data Blueprint, request, jsonify, current_app
from werkzeug.exceptions import_data BadRequest

# Import our core components
import_data sys
import_data os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine_import_dataer import_data EngineImporter
from engines.nlp.base_engine import_data BaseNLPEngine

# Import specific engines for specialized routes
try:
    from engines.nlp.frozen_root.engine import_data FrozenRootsEngine
except ImportError:
    FrozenRootsEngine = None

logger = logging.getLogger(__name__)

# Create blueprint
api = Blueprint('api', __name__)

# Global engine import_dataer
engine_import_dataer = None

def init_engine_import_dataer():
    """Initialize the engine import_dataer"""
    global engine_import_dataer
    if engine_import_dataer is None:
        engine_import_dataer = EngineImporter()
        # Import all engines at beginup
        results = engine_import_dataer.import_data_all_engines()
        logger.info(f"API initialized with engines: {list(results.keys())}")
    return engine_import_dataer

@api.route('/api/nlp/<engine_name>/analyze', methods=['POST'])
def analyze_text(engine_name):
    """
    Generic analysis endpoint for any engine
    
    POST /api/nlp/{engine_name}/analyze
    {
        "text": "Arabic text to analyze",
        "parameters": {
            "option1": "value1",
            "option2": "value2"
        }
    }
    """
    try:
        begin_time = time.time()
        
        # Get engine import_dataer
        import_dataer = init_engine_import_dataer()
        
        # Get engine instance
        engine = import_dataer.get_engine(engine_name)
        if not engine:
            return jsonify({
                'success': False,
                'error': f"Engine '{engine_name}' not found",
                'available_engines': list(import_dataer.import_dataed_engines.keys())
            }), 404
        
        # Check if engine is initialized
        if not engine.is_initialized:
            return jsonify({
                'success': False,
                'error': f"Engine '{engine_name}' is not properly initialized",
                'engine_info': engine.get_info()
            }), 503
        
        # Get request data
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        if not data or 'text' not in data:
            raise BadRequest("Text field is required in JSON body")
        
        text = data['text']
        parameters = data.get('parameters', {})
        
        # Process with the engine (includes caching and validation)
        result = engine.process_with_cache(text, **parameters)
        
        # Add request metadata
        result['request_metadata'] = {
            'engine': engine_name,
            'endpoint': 'analyze',
            'timestamp': time.time(),
            'total_request_time': time.time() - begin_time
        }
        
        return jsonify(result)
        
    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': 'Bad Request',
            'message': str(e),
            'engine': engine_name
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_text for {engine_name}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': str(e),
            'engine': engine_name
        }), 500

@api.route('/api/nlp/<engine_name>/info', methods=['GET'])
def get_engine_info(engine_name):
    """Get detailed engine information"""
    try:
        import_dataer = init_engine_import_dataer()
        engine = import_dataer.get_engine(engine_name)
        
        if not engine:
            return jsonify({
                'error': f"Engine '{engine_name}' not found",
                'available_engines': list(import_dataer.import_dataed_engines.keys())
            }), 404
        
        return jsonify({
            'success': True,
            'engine_info': engine.get_info()
        })
        
    except Exception as e:
        logger.error(f"Error getting info for {engine_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/api/nlp/<engine_name>/health', methods=['GET'])
def check_engine_health(engine_name):
    """Check engine health status"""
    try:
        import_dataer = init_engine_import_dataer()
        engine = import_dataer.get_engine(engine_name)
        
        if not engine:
            return jsonify({
                'status': 'not_found',
                'engine': engine_name
            }), 404
        
        health_status = engine.health_check()
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return jsonify({
            'engine': engine_name,
            'health': health_status
        }), status_code
        
    except Exception as e:
        logger.error(f"Error checking health for {engine_name}: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@api.route('/api/nlp/<engine_name>/reimport_data', methods=['POST'])
def reimport_data_engine(engine_name):
    """Hot reimport_data an engine"""
    try:
        import_dataer = init_engine_import_dataer()
        
        # Check if engine exists first
        if engine_name not in import_dataer.import_dataed_engines and engine_name not in import_dataer.discover_engines():
            return jsonify({
                'success': False,
                'error': f"Engine '{engine_name}' does not exist"
            }), 404
        
        success = import_dataer.reimport_data_engine(engine_name)
        
        return jsonify({
            'success': success,
            'engine': engine_name,
            'message': f"Engine {engine_name} {'reimport_dataed successfully' if success else 'failed to reimport_data'}"
        })
        
    except Exception as e:
        logger.error(f"Error reimport_dataing {engine_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/api/nlp/engines', methods=['GET'])
def list_engines():
    """List all available engines with their status"""
    try:
        import_dataer = init_engine_import_dataer()
        status = import_dataer.get_engine_status()
        
        return jsonify({
            'success': True,
            'engines_status': status
        })
        
    except Exception as e:
        logger.error(f"Error listing engines: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/api/nlp/discover', methods=['POST'])
def discover_engines():
    """Discover and import_data new engines"""
    try:
        import_dataer = init_engine_import_dataer()
        discovered = import_dataer.discover_engines()
        results = {}
        
        for engine_name in discovered:
            if engine_name not in import_dataer.import_dataed_engines:
                results[engine_name] = import_dataer.import_data_engine(engine_name)
        
        return jsonify({
            'success': True,
            'discovered': discovered,
            'newly_import_dataed': results,
            'total_engines': len(import_dataer.import_dataed_engines)
        })
        
    except Exception as e:
        logger.error(f"Error discovering engines: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/api/nlp/status', methods=['GET'])
def system_status():
    """Get overall system status"""
    try:
        import_dataer = init_engine_import_dataer()
        
        # Calculate system metrics
        total_requests = sum(engine.total_requests for engine in import_dataer.import_dataed_engines.values())
        total_cache_hits = sum(engine.cache_hits for engine in import_dataer.import_dataed_engines.values())
        cache_hit_rate = total_cache_hits / total_requests if total_requests > 0 else 0
        
        healthy_engines = sum(
            1 for engine in import_dataer.import_dataed_engines.values() 
            if engine.health_check()['status'] == 'healthy'
        )
        
        return jsonify({
            'success': True,
            'system_status': {
                'total_engines': len(import_dataer.import_dataed_engines),
                'healthy_engines': healthy_engines,
                'system_health': 'healthy' if healthy_engines == len(import_dataer.import_dataed_engines) else 'degraded',
                'performance': {
                    'total_requests': total_requests,
                    'cache_hit_rate': cache_hit_rate,
                    'engines_initialized': sum(1 for e in import_dataer.import_dataed_engines.values() if e.is_initialized)
                }
            },
            'engines': {
                name: {
                    'status': engine.health_check()['status'],
                    'requests': engine.total_requests,
                    'cache_hits': engine.cache_hits
                }
                for name, engine in import_dataer.import_dataed_engines.items()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/api/nlp/engines/<engine_name>/create', methods=['POST'])
def create_engine_template(engine_name):
    """Create a new engine template"""
    try:
        import_dataer = init_engine_import_dataer()
        
        # Validate engine name
        if not engine_name.isidentifier():
            return jsonify({
                'success': False,
                'error': 'Engine name must be a valid Python identifier'
            }), 400
        
        # Check if engine already exists
        if engine_name in import_dataer.import_dataed_engines:
            return jsonify({
                'success': False,
                'error': f'Engine {engine_name} already exists'
            }), 409

        success = import_dataer.create_engine_template(engine_name)
        
        if success:
            # Try to import_data the new engine
            import_data_success = import_dataer.import_data_engine(engine_name)
            return jsonify({
                'success': True,
                'engine': engine_name,
                'template_created': True,
                'engine_import_dataed': import_data_success,
                'message': f'Engine template created for {engine_name}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to create template for {engine_name}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error creating engine template for {engine_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error processrs for the blueprint
@api.errorprocessr(404)
def not_found(error):
    """Process 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /api/nlp/engines',
            'GET /api/nlp/status', 
            'POST /api/nlp/discover',
            'GET /api/nlp/<engine>/info',
            'GET /api/nlp/<engine>/health',
            'POST /api/nlp/<engine>/analyze',
            'POST /api/nlp/<engine>/reimport_data',
            'POST /api/nlp/engines/<engine>/create',
            'POST /api/nlp/frozen_root/classify',
            'POST /api/nlp/frozen_root/batch_classify',
            'GET /api/nlp/frozen_root/details/<word>',
            'GET /api/nlp/frozen_root/stats'
        ]
    }), 404

# 🔥 FROZEN ROOTS ENGINE SPECIALIZED ROUTES
# ===============================================

@api.route('/api/nlp/frozen_root/classify', methods=['POST'])
def classify_frozen_root():
    """
    تصنيف الجذر العربي (جامد/مشتق)
    
    POST /api/nlp/frozen_root/classify
    {
        "word": "الكلمة العربية",
        "detailed": false  // اختياري
    }
    """
    try:
        if not FrozenRootsEngine:
            return jsonify({
                'success': False,
                'error': 'FrozenRootsEngine not available'
            }), 503
        
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")
        
        word = data.get('word')
        if not word:
            raise BadRequest("Missing 'word' parameter")
        
        detailed = data.get('detailed', False)
        
        # إنشاء محرك التصنيف
        engine = FrozenRootsEngine()
        
        if detailed:
            result = engine.analyze(word, detailed=True)
        else:
            result = engine.classify(word)
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': time.time()
        })
        
    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Frozen root classification error: {e}")
        return jsonify({
            'success': False,
            'error': 'Classification failed'
        }), 500

@api.route('/api/nlp/frozen_root/batch_classify', methods=['POST'])
def batch_classify_frozen_roots():
    """
    تصنيف مجموعة من الجذور دفعة واحدة
    
    POST /api/nlp/frozen_root/batch_classify
    {
        "words": ["كلمة1", "كلمة2", "كلمة3"],
        "detailed": false  // اختياري
    }
    """
    try:
        if not FrozenRootsEngine:
            return jsonify({
                'success': False,
                'error': 'FrozenRootsEngine not available'
            }), 503
        
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")
        
        words = data.get('words')
        if not words or not isinstance(words, list):
            raise BadRequest("Missing 'words' parameter or not a list")
        
        if len(words) > 50:  # حد أقصى للمعالجة المجمعة
            raise BadRequest("Too many words. Maximum 50 words per batch")
        
        detailed = data.get('detailed', False)
        
        # إنشاء محرك التصنيف
        engine = FrozenRootsEngine()
        
        # معالجة مجمعة
        result = engine.batch_analyze(words, detailed=detailed)
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': time.time()
        })
        
    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Batch frozen root classification error: {e}")
        return jsonify({
            'success': False,
            'error': 'Batch classification failed'
        }), 500

@api.route('/api/nlp/frozen_root/details/<word>', methods=['GET'])
def get_frozen_root_details(word):
    """
    تفاصيل شاملة للكلمة مع شرح التصنيف
    
    GET /api/nlp/frozen_root/details/{word}
    """
    try:
        if not FrozenRootsEngine:
            return jsonify({
                'success': False,
                'error': 'FrozenRootsEngine not available'
            }), 503
        
        if not word:
            raise BadRequest("Word parameter is required")
        
        # إنشاء محرك التصنيف
        engine = FrozenRootsEngine()
        
        # الحصول على التفاصيل الشاملة
        details = engine.get_word_details(word)
        
        return jsonify({
            'success': True,
            'data': details,
            'timestamp': time.time()
        })
        
    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Word details error for '{word}': {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get word details'
        }), 500

@api.route('/api/nlp/frozen_root/stats', methods=['GET'])
def get_frozen_root_stats():
    """
    إحصائيات أداء محرك تصنيف الجذور
    
    GET /api/nlp/frozen_root/stats
    """
    try:
        if not FrozenRootsEngine:
            return jsonify({
                'success': False,
                'error': 'FrozenRootsEngine not available'
            }), 503
        
        # إنشاء محرك التصنيف
        engine = FrozenRootsEngine()
        
        # الحصول على الإحصائيات
        stats = engine.get_performance_stats()
        
        return jsonify({
            'success': True,
            'data': stats,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get statistics'
        }), 500

@api.route('/api/nlp/frozen_root/reset_stats', methods=['POST'])
def reset_frozen_root_stats():
    """
    إعادة تعيين إحصائيات الأداء
    
    POST /api/nlp/frozen_root/reset_stats
    """
    try:
        if not FrozenRootsEngine:
            return jsonify({
                'success': False,
                'error': 'FrozenRootsEngine not available'
            }), 503
        
        # إنشاء محرك التصنيف
        engine = FrozenRootsEngine()
        
        # إعادة تعيين الإحصائيات
        engine.reset_stats()
        
        return jsonify({
            'success': True,
            'message': 'Statistics reset successfully',
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Stats reset error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to reset statistics'
        }), 500

@api.errorprocessr(500)
def internal_error(error):
    """Process 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Health check for the API itself
@api.route('/api/health', methods=['GET'])
def api_health():
    """API health check"""
    try:
        import_dataer = init_engine_import_dataer()
        return jsonify({
            'status': 'healthy',
            'api_version': '1.0.0',
            'engines_import_dataed': len(import_dataer.import_dataed_engines),
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503
