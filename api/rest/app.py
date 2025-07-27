# api/rest/app.py - Enterprise Flask API for Arabic NLP System

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

from flask import_data Flask, request, jsonify, g
from flask_cors import_data CORS
from flask_limiter import_data Limiter
from flask_limiter.util import_data get_remote_address
from flask_caching import_data Cache
import_data logging
import_data time
import_data sys
from pathlib import_data Path
import_data os
from datetime import_data datetime
import_data uuid

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from engines.nlp.full_pipeline.engine import_data FullPipeline
from engines.nlp.particles.engine import_data GrammaticalParticlesEngine

class ArabicNLPAPI:
    """
    Enterprise-grade Flask API for Arabic NLP System
    
    Features:
    - RESTful endpoints for all NLP engines
    - Rate limiting and caching
    - Comprehensive error handling
    - Request/response logging
    - Performance monitoring
    - CORS support
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self._configure_app()
        self._setup_engines()
        self._setup_middleware()
        self._register_routes()
        self._setup_logging()
    
    def _configure_app(self):
        """Configure Flask application settings"""
        self.app.config.update({
            'SECRET_KEY': os.environ.get('SECRET_KEY', 'arabic-nlp-enterprise-key'),
            'CACHE_TYPE': 'simple',
            'CACHE_DEFAULT_TIMEOUT': 300,
            'JSON_AS_ASCII': False,  # Support Arabic characters
            'JSONIFY_PRETTYPRINT_REGULAR': True
        })
    
    def _setup_engines(self):
        """Initialize NLP processing engines"""
        try:
            self.full_pipeline = FullPipeline()
            self.particles_engine = GrammaticalParticlesEngine()
            logging.info("✅ All NLP engines initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize engines: {e}")
            raise
    
    def _setup_middleware(self):
        """Setup middleware components"""
        # CORS for cross-origin requests
        CORS(self.app, resources={
            r"/api/*": {
                "origins": ["http://localhost:3000", "https://arabic-nlp.com"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        # Rate limiting
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
        
        # Caching
        self.cache = Cache(self.app)
        
        # Request logging middleware
        @self.app.before_request
        def before_request():
            g.start_time = time.time()
            g.request_id = str(uuid.uuid4())
            
            logging.info(f"🔵 REQUEST START [{g.request_id}] {request.method} {request.path}")
        
        @self.app.after_request
        def after_request(response):
            duration = time.time() - g.start_time
            logging.info(f"🟢 REQUEST END [{g.request_id}] {response.status_code} - {duration:.3f}s")
            return response
    
    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            processrs=[
                logging.FileProcessr('api_access.log'),
                logging.StreamProcessr()
            ]
        )
    
    def _register_routes(self):
        """Register all API routes"""
        
        @self.app.route('/')
        def index():
            """API health check and information"""
            return jsonify({
                "name": "Arabic NLP Enterprise API",
                "version": "3.0.0",
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "endpoints": {
                    "full_analysis": "/api/v1/analyze",
                    "particles": "/api/v1/particles/analyze",
                    "phonemes": "/api/v1/phonemes/extract",
                    "syllabic_units": "/api/v1/syllabic_units/segment",
                    "morphology": "/api/v1/morphology/derive",
                    "inflection": "/api/v1/inflection/conjugate",
                    "health": "/api/v1/health"
                }
            })
        
        @self.app.route('/api/v1/health')
        def health_check():
            """Comprehensive health check endpoint"""
            try:
                # Test all engines
                test_word = "كتاب"
                
                # Test full pipeline
                pipeline_result = self.full_pipeline.analyze(test_word)
                pipeline_healthy = len(pipeline_result.get('phonemes', [])) > 0
                
                # Test particles engine
                particles_result = self.particles_engine.analyze("إن")
                particles_healthy = particles_result.get('category') == 'شرط'
                
                health_status = {
                    "status": "healthy" if (pipeline_healthy and particles_healthy) else "degraded",
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "full_pipeline": "✅ operational" if pipeline_healthy else "❌ failed",
                        "particles_engine": "✅ operational" if particles_healthy else "❌ failed"
                    },
                    "performance": {
                        "test_word": test_word,
                        "response_time_ms": f"{(time.time() - g.start_time) * 1000:.2f}"
                    }
                }
                
                status_code = 200 if (pipeline_healthy and particles_healthy) else 503
                return jsonify(health_status), status_code
                
            except Exception as e:
                return jsonify({
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 503
        
        @self.app.route('/api/v1/analyze', methods=['POST'])
        @self.limiter.limit("10 per minute")
        def full_analysis():
            """Complete morphophonological analysis endpoint"""
            try:
                data = request.get_json()
                
                if not data or 'text' not in data:
                    return jsonify({
                        "error": "Missing 'text' field in request body",
                        "example": {"text": "كتاب"}
                    }), 400
                
                text = data['text'].strip()
                if not text:
                    return jsonify({"error": "Text cannot be empty"}), 400
                
                # Check cache first
                cache_key = f"full_analysis_{hash(text)}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    cached_result['cached'] = True
                    return jsonify(cached_result)
                
                # Perform analysis
                start_time = time.time()
                result = self.full_pipeline.analyze(text)
                processing_time = time.time() - start_time
                
                # Enhanced response
                response = {
                    "input": text,
                    "analysis": result,
                    "metadata": {
                        "processing_time_ms": round(processing_time * 1000, 3),
                        "request_id": g.request_id,
                        "timestamp": datetime.now().isoformat(),
                        "cached": False
                    }
                }
                
                # Cache the result
                self.cache.set(cache_key, response, timeout=300)
                
                return jsonify(response)
                
            except Exception as e:
                logging.error(f"❌ Full analysis error [{g.request_id}]: {e}")
                return jsonify({
                    "error": "Internal server error",
                    "request_id": g.request_id,
                    "message": str(e)
                }), 500
        
        @self.app.route('/api/v1/particles/analyze', methods=['POST'])
        @self.limiter.limit("20 per minute")
        def particles_analysis():
            """Grammatical particles analysis endpoint"""
            try:
                data = request.get_json()
                
                if not data or 'particle' not in data:
                    return jsonify({
                        "error": "Missing 'particle' field in request body",
                        "example": {"particle": "إن"}
                    }), 400
                
                particle = data['particle'].strip()
                if not particle:
                    return jsonify({"error": "Particle cannot be empty"}), 400
                
                # Perform analysis
                start_time = time.time()
                result = self.particles_engine.analyze(particle)
                processing_time = time.time() - start_time
                
                response = {
                    "input": particle,
                    "analysis": result,
                    "metadata": {
                        "processing_time_ms": round(processing_time * 1000, 3),
                        "request_id": g.request_id,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                logging.error(f"❌ Particles analysis error [{g.request_id}]: {e}")
                return jsonify({
                    "error": "Internal server error",
                    "request_id": g.request_id,
                    "message": str(e)
                }), 500
        
        @self.app.route('/api/v1/particles/batch', methods=['POST'])
        @self.limiter.limit("5 per minute")
        def particles_batch_analysis():
            """Batch particles analysis endpoint"""
            try:
                data = request.get_json()
                
                if not data or 'particles' not in data:
                    return jsonify({
                        "error": "Missing 'particles' field in request body",
                        "example": {"particles": ["إن", "هل", "لا"]}
                    }), 400
                
                particles = data['particles']
                if not isinstance(particles, list) or len(particles) == 0:
                    return jsonify({"error": "Particles must be a non-empty list"}), 400
                
                if len(particles) > 50:
                    return jsonify({"error": "Maximum 50 particles per batch"}), 400
                
                # Perform batch analysis
                start_time = time.time()
                results = self.particles_engine.batch_analyze(particles)
                processing_time = time.time() - start_time
                
                response = {
                    "input": particles,
                    "results": results,
                    "metadata": {
                        "count": len(results),
                        "processing_time_ms": round(processing_time * 1000, 3),
                        "avg_time_per_particle": round((processing_time / len(particles)) * 1000, 3),
                        "request_id": g.request_id,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                logging.error(f"❌ Batch analysis error [{g.request_id}]: {e}")
                return jsonify({
                    "error": "Internal server error",
                    "request_id": g.request_id,
                    "message": str(e)
                }), 500
        
        @self.app.route('/api/v1/particles/categories')
        def get_particle_categories():
            """Get supported particle categories"""
            try:
                categories = self.particles_engine.get_supported_categories()
                stats = self.particles_engine.get_statistics()
                
                response = {
                    "categories": categories,
                    "statistics": stats['particle_distribution'],
                    "total_particles": sum(stats['particle_distribution'].values()),
                    "metadata": {
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                logging.error(f"❌ Categories endpoint error: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route('/api/v1/statistics')
        def get_api_statistics():
            """Get API and engine statistics"""
            try:
                pipeline_stats = self.full_pipeline.get_performance_stats()
                particles_stats = self.particles_engine.get_statistics()
                
                response = {
                    "api_info": {
                        "name": "Arabic NLP Enterprise API",
                        "version": "3.0.0",
                        "uptime_since": datetime.now().isoformat()
                    },
                    "engines": {
                        "full_pipeline": pipeline_stats,
                        "particles_engine": particles_stats
                    },
                    "metadata": {
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                logging.error(f"❌ Statistics endpoint error: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        # Error processrs
        @self.app.errorprocessr(404)
        def not_found(error):
            return jsonify({
                "error": "Endpoint not found",
                "message": "The requested endpoint does not exist",
                "available_endpoints": [
                    "/api/v1/analyze",
                    "/api/v1/particles/analyze",
                    "/api/v1/particles/batch",
                    "/api/v1/particles/categories",
                    "/api/v1/statistics",
                    "/api/v1/health"
                ]
            }), 404
        
        @self.app.errorprocessr(429)
        def ratelimit_processr(e):
            return jsonify({
                "error": "Rate limit exceeded",
                "message": str(e.description),
                "retry_after": e.retry_after
            }), 429
        
        @self.app.errorprocessr(500)
        def internal_error(error):
            return jsonify({
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }), 500

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logging.info(f"🚀 Starting Arabic NLP API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Create application instance
api = ArabicNLPAPI()
app = api.app  # For WSGI deployment

if __name__ == '__main__':
    # Development server
    api.run(debug=True, port=5000)
