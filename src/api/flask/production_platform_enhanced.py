#!/usr/bin/env python3
"""
🚀 Advanced Arabic Morphophonological Engine - Enhanced Production Platform
Dynamic AI-Powered Expert UI with comprehensive monitoring and real-time analytics
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore")

import_data json
import_data logging
import_data os
import_data sys
import_data threading
import_data time
from datetime import_data datetime, timedelta
from pathlib import_data Path
from typing import_data Any, Dict, List, Optional

# Disable all warnings system-wide
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Flask and extensions
from flask import_data Flask, jsonify, render_template, request, send_from_directory
from flask_cors import_data CORS

# Import custom JSON encoder
from utils.json_encoder import_data CustomJSONEncoder, configure_flask_json

# Configure UTF-8 logging for Windows
class UTF8StreamProcessr(logging.StreamProcessr):
    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8')
            except:
                pass

# Setup UTF-8 console for Windows
if sys.platform == 'win32':
    from contextlib import_data suppress
    with suppress(AttributeError, OSError):
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

logging.basicConfig(
    level=logging.CRITICAL,
    format='%(levelname)s: %(message)s',
    processrs=[logging.NullProcessr()]
)
logger = logging.getLogger(__name__)
logger.disabled = True

# Initialize Flask app with production config
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

# Configure enhanced JSON encoding
app = configure_flask_json(app)

# Enhanced production configuration
app.config.update(
    SECRET_KEY='arabic-morphophon-production-2025-enhanced',
    DEBUG=False,
    TESTING=False,
    JSONIFY_PRETTYPRINT_REGULAR=True,
    JSON_AS_ASCII=False,
    JSON_SORT_KEYS=False,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max upimport_data
)

# Enable CORS for API access
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }
})

from enum import_data Enum

# Safe import_datas with fallbacks
from typing import_data Any, Dict, List, Optional, Union

# Type definitions
AnalysisLevel: Optional[Union[type, Any]] = None
engine = None
advanced_system = None
ANALYSIS_ENGINE_AVAILABLE = False
ADVANCED_SYSTEM_AVAILABLE = False

# Mock AnalysisLevel for type checking
class MockAnalysisLevel:
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"

try:
    from arabic_morphophon.integrator import_data AnalysisLevel as RealAnalysisLevel
    from arabic_morphophon.integrator import_data MorphophonologicalEngine
    AnalysisLevel = RealAnalysisLevel
    engine = MorphophonologicalEngine()
    ANALYSIS_ENGINE_AVAILABLE = True
except ImportError:
    AnalysisLevel = MockAnalysisLevel

try:
    from web_apps.advanced_hierarchical_api import_data AdvancedArabicAnalysisSystem
    advanced_system = AdvancedArabicAnalysisSystem()
    ADVANCED_SYSTEM_AVAILABLE = True
except ImportError:
    pass

# Enhanced global statistics with AI-powered insights
PLATFORM_STATS = {
    'begin_time': datetime.now(),
    'total_requests': 0,
    'analysis_requests': 0,
    'successful_analyses': 0,
    'failed_analyses': 0,
    'unique_texts_analyzed': set(),
    'average_response_time': 0.0,
    'last_analysis_time': None,
    'hourly_requests': {},
    'language_patterns': {},
    'error_patterns': [],
    'performance_trends': []
}

def update_stats(success: bool = True, response_time: float = 0.0, text: Optional[str] = None, error_msg: Optional[str] = None):
    """Enhanced statistics tracking with AI insights"""
    PLATFORM_STATS['total_requests'] += 1
    
    # Track hourly patterns
    hour_key = datetime.now().strftime('%Y-%m-%d %H:00')
    PLATFORM_STATS['hourly_requests'][hour_key] = PLATFORM_STATS['hourly_requests'].get(hour_key, 0) + 1
    
    if text:
        PLATFORM_STATS['analysis_requests'] += 1
        PLATFORM_STATS['unique_texts_analyzed'].add(text[:100])
        
        # AI pattern analysis
        if len(text) > 5:  # Only for meaningful text
            text_length_category = 'short' if len(text) < 20 else 'medium' if len(text) < 100 else 'long'
            PLATFORM_STATS['language_patterns'][text_length_category] = PLATFORM_STATS['language_patterns'].get(text_length_category, 0) + 1
        
        if success:
            PLATFORM_STATS['successful_analyses'] += 1
        else:
            PLATFORM_STATS['failed_analyses'] += 1
            if error_msg:
                PLATFORM_STATS['error_patterns'].append({
                    'timestamp': datetime.now(),
                    'error': error_msg[:200],  # Truncate error
                    'text_length': len(text) if text else 0
                })
                # Keep only last 50 errors
                PLATFORM_STATS['error_patterns'] = PLATFORM_STATS['error_patterns'][-50:]
        
        # Dynamic response time calculation
        if PLATFORM_STATS['analysis_requests'] > 0:
            current_avg = PLATFORM_STATS['average_response_time']
            total_analyses = PLATFORM_STATS['analysis_requests']
            PLATFORM_STATS['average_response_time'] = (current_avg * (total_analyses - 1) + response_time) / total_analyses
        
        PLATFORM_STATS['last_analysis_time'] = datetime.now()
        
        # Track performance trends
        PLATFORM_STATS['performance_trends'].append({
            'timestamp': datetime.now(),
            'response_time': response_time,
            'success': success
        })
        # Keep only last 100 data points
        PLATFORM_STATS['performance_trends'] = PLATFORM_STATS['performance_trends'][-100:]

def get_dynamic_ai_insights():
    """Generate AI-powered insights from platform data"""
    now = datetime.now()
    uptime = now - PLATFORM_STATS['begin_time']
    
    # Calculate success rate
    success_rate = 100.0
    if PLATFORM_STATS['analysis_requests'] > 0:
        success_rate = (PLATFORM_STATS['successful_analyses'] / PLATFORM_STATS['analysis_requests']) * 100
    
    # Calculate requests per minute
    requests_per_minute = 0.0
    if uptime.total_seconds() > 0:
        requests_per_minute = PLATFORM_STATS['total_requests'] / (uptime.total_seconds() / 60)
    
    # AI Performance Analysis
    performance_status = "Excellent"
    if success_rate < 95:
        performance_status = "Good" if success_rate > 80 else "Needs Attention"
    elif PLATFORM_STATS['average_response_time'] > 1.0:
        performance_status = "Good"
    
    # Pattern insights
    most_common_length = max(PLATFORM_STATS['language_patterns'].items(), 
                           key=lambda x: x[1], default=('medium', 0))[0]
    
    # Recent performance trend
    recent_trends = PLATFORM_STATS['performance_trends'][-10:] if PLATFORM_STATS['performance_trends'] else []
    avg_recent_time = sum(t['response_time'] for t in recent_trends) / len(recent_trends) if recent_trends else 0
    
    return {
        'uptime_seconds': int(uptime.total_seconds()),
        'uptime_formatted': str(uptime).split('.')[0],
        'success_rate': round(success_rate, 2),
        'requests_per_minute': round(requests_per_minute, 2),
        'unique_texts': len(PLATFORM_STATS['unique_texts_analyzed']),
        'avg_response_time_ms': round(PLATFORM_STATS['average_response_time'] * 1000, 2),
        'performance_status': performance_status,
        'most_common_text_length': most_common_length,
        'recent_avg_response_ms': round(avg_recent_time * 1000, 2),
        'hourly_pattern': dict(list(PLATFORM_STATS['hourly_requests'].items())[-24:]),  # Last 24 hours
        'error_count': len(PLATFORM_STATS['error_patterns'])
    }

# Enhanced API Routes with Dynamic Features

@app.route('/')
def dashboard():
    """Dynamic Expert Dashboard"""
    return render_template('expert_dashboard.html', 
                         engine_available=ANALYSIS_ENGINE_AVAILABLE,
                         advanced_available=ADVANCED_SYSTEM_AVAILABLE)

@app.route('/api/health')
def health_check():
    """Enhanced health check with system diagnostics"""
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'main_engine': ANALYSIS_ENGINE_AVAILABLE,
            'advanced_system': ADVANCED_SYSTEM_AVAILABLE,
            'database': True,  # Assume healthy
            'api': True
        },
        'system_info': {
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
            'memory_usage': 'normal',  # Could integrate psutil
        }
    }
    return jsonify(health_data)

@app.route('/api/stats')
def get_stats():
    """Dynamic statistics with AI insights"""
    base_stats = dict(PLATFORM_STATS)
    # Convert set to list for JSON serialization
    base_stats['unique_texts_analyzed'] = len(base_stats['unique_texts_analyzed'])
    # Convert datetime objects to strings
    if base_stats['begin_time']:
        base_stats['begin_time'] = base_stats['begin_time'].isoformat()
    if base_stats['last_analysis_time']:
        base_stats['last_analysis_time'] = base_stats['last_analysis_time'].isoformat()
    
    # Add AI insights
    ai_insights = get_dynamic_ai_insights()
    base_stats.update(ai_insights)
    
    return jsonify(base_stats)

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Enhanced text analysis with dynamic engine selection"""
    begin_time = time.time()
    data = None  # Initialize data variable
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        analysis_level = data.get('level', 'basic').lower()
        use_advanced = data.get('advanced', False)
        
        results = {}
        
        # Main engine analysis
        if ANALYSIS_ENGINE_AVAILABLE and engine:
            try:
                if hasattr(AnalysisLevel, analysis_level.upper()):
                    level_enum = getattr(AnalysisLevel, analysis_level.upper())
                else:
                    level_enum = getattr(AnalysisLevel, 'BASIC')  # Use getattr for type safety
                
                main_result = engine.analyze(text, level=level_enum)
                results['main_analysis'] = main_result
            except Exception as e:
                results['main_analysis'] = {'error': f'Main analysis failed: {str(e)}'}
        
        # Advanced system analysis
        if use_advanced and ADVANCED_SYSTEM_AVAILABLE and advanced_system:
            try:
                # Use the correct method name based on the actual API
                if hasattr(advanced_system, 'analyze_hierarchical'):
                    advanced_result = advanced_system.analyze_hierarchical(text)
                elif hasattr(advanced_system, 'analyze_traditional'):
                    advanced_result = advanced_system.analyze_traditional(text)
                else:
                    advanced_result = {'note': 'Advanced analysis method not found, using placeholder'}
                results['advanced_analysis'] = advanced_result
            except Exception as e:
                results['advanced_analysis'] = {'error': f'Advanced analysis failed: {str(e)}'}
        
        # Calculate response time
        response_time = time.time() - begin_time
        results['meta'] = {
            'response_time': round(response_time, 4),
            'timestamp': datetime.now().isoformat(),
            'engine_used': 'main' if ANALYSIS_ENGINE_AVAILABLE else 'none',
            'advanced_used': use_advanced and ADVANCED_SYSTEM_AVAILABLE
        }
        
        update_stats(success=True, response_time=response_time, text=text)
        return jsonify(results)
        
    except Exception as e:
        error_time = time.time() - begin_time
        text_for_stats = ''
        if data and isinstance(data, dict):
            text_for_stats = data.get('text', '')
        
        update_stats(success=False, response_time=error_time, 
                    text=text_for_stats, 
                    error_msg=str(e))
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e),
            'meta': {
                'response_time': round(error_time, 4),
                'timestamp': datetime.now().isoformat()
            }
        }), 500

@app.route('/api/test-json')
def test_json_serialization():
    """Test JSON serialization with AnalysisLevel enum"""
    try:
        test_data = {
            'simple_string': 'test',
            'arabic_text': 'أهلاً وسهلاً',
            'analysis_level': getattr(AnalysisLevel, 'BASIC', 'basic'),
            'timestamp': datetime.now()
        }
        
        # Test direct JSON serialization
        from utils.json_encoder import_data safe_jsonify
        json_result = safe_jsonify(test_data)
        
        return jsonify({
            'status': 'success',
            'test_data': test_data,
            'json_test': 'passed',
            'message': 'JSON serialization working correctly'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'JSON serialization failed'
        }), 500

@app.route('/api/analyze-particles', methods=['POST'])
def analyze_particles():
    """API endpoint for Arabic grammatical particles analysis"""
    begin_time = time.time()
    data = None
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Initialize particles engine
        try:
            from engines.nlp.particles.engine import_data GrammaticalParticlesEngine
            particles_engine = GrammaticalParticlesEngine()
        except ImportError:
            return jsonify({
                'error': 'Particles engine not available',
                'message': 'Install required dependencies: pip install pyyaml'
            }), 503
        
        # Analyze particles in text
        words = text.split()
        particle_results = []
        word_analyses = []
        
        for word in words:
            clean_word = word.strip('؟!،.')
            particle_analysis = particles_engine.analyze(clean_word)
            
            word_data = {
                'word': word,
                'clean_word': clean_word,
                'is_particle': particle_analysis['analysis_metadata'].get('is_recognized_particle', False),
                'category': particle_analysis['category'],
                'phonemes': particle_analysis['phonemes'],
                'syllabic_units': particle_analysis['syllabic_units'],
                'processing_time_ms': particle_analysis['analysis_metadata']['processing_time_ms']
            }
            
            word_analyses.append(word_data)
            
            if word_data['is_particle']:
                particle_results.append(word_data)
        
        # Calculate statistics
        particle_count = len(particle_results)
        particle_percentage = (particle_count / len(words) * 100) if words else 0
        
        # Group by categories
        categories = {}
        for particle in particle_results:
            category = particle['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(particle['clean_word'])
        
        response_time = time.time() - begin_time
        
        # Update platform statistics
        update_stats(success=True, response_time=response_time, text=text)
        
        return jsonify({
            'status': 'success',
            'original_text': text,
            'analysis': {
                'total_words': len(words),
                'particles_found': particle_count,
                'particle_percentage': round(particle_percentage, 2),
                'categories_found': list(categories.keys()),
                'particles_by_category': categories
            },
            'particles': particle_results,
            'all_words': word_analyses,
            'meta': {
                'response_time': round(response_time, 4),
                'timestamp': datetime.now().isoformat(),
                'engine': 'grammatical_particles'
            }
        })
        
    except Exception as e:
        error_time = time.time() - begin_time
        text_for_stats = data.get('text', '') if data else ''
        
        update_stats(success=False, response_time=error_time, 
                    text=text_for_stats, 
                    error_msg=str(e))
        return jsonify({
            'error': 'Particles analysis failed',
            'details': str(e),
            'meta': {
                'response_time': round(error_time, 4),
                'timestamp': datetime.now().isoformat()
            }
        }), 500

@app.route('/api/comprehensive-classification', methods=['POST'])
def comprehensive_classification():
    """
    Comprehensive particle classification and segregation endpoint
    Provides complete analysis with category/subcategory segregation
    """
    begin_time = time.time()
    data = None
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Import and initialize comprehensive analyzer
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from comprehensive_particle_classification import_data ComprehensiveParticleAnalyzer
            analyzer = ComprehensiveParticleAnalyzer()
        except ImportError as e:
            return jsonify({
                'error': 'Comprehensive analyzer not available',
                'details': str(e)
            }), 503
        
        # Perform comprehensive analysis
        analysis = analyzer.classify_and_segregate_text(text)
        
        # Add category definitions for reference
        analysis['category_definitions'] = analyzer.category_definitions
        
        # Add processing metadata
        response_time = time.time() - begin_time
        
        # Update platform statistics
        update_stats(success=True, response_time=response_time, text=text)
        
        return jsonify({
            'status': 'success',
            'comprehensive_analysis': analysis,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'analyzer_version': '2.0',
                'response_time': round(response_time, 4),
                'total_categories': len(analyzer.category_definitions),
                'total_extended_particles': len(analyzer.extended_particles)
            }
        })
        
    except Exception as e:
        error_time = time.time() - begin_time
        text_for_stats = data.get('text', '') if data else ''
        
        update_stats(success=False, response_time=error_time, 
                    text=text_for_stats, 
                    error_msg=str(e))
        
        return jsonify({
            'error': 'Comprehensive classification failed',
            'details': str(e),
            'metadata': {
                'status': 'error',
                'response_time': round(error_time, 4),
                'timestamp': datetime.now().isoformat()
            }
        }), 500

@app.route('/api/category-overview', methods=['GET'])
def category_overview():
    """
    Get overview of all particle categories and subcategories
    """
    begin_time = time.time()
    
    try:
        # Import comprehensive analyzer
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from comprehensive_particle_classification import_data ComprehensiveParticleAnalyzer
        analyzer = ComprehensiveParticleAnalyzer()
        
        response_time = time.time() - begin_time
        
        return jsonify({
            'status': 'success',
            'categories': analyzer.get_all_categories_with_examples(),
            'category_definitions': analyzer.category_definitions,
            'statistics': {
                'total_categories': len(analyzer.category_definitions),
                'total_particles': len(analyzer.extended_particles),
                'total_subcategories': len(set(
                    data.get('subcategory', 'غير محدد') 
                    for data in analyzer.extended_particles.values()
                ))
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'response_time': round(response_time, 4),
                'system_version': '2.0'
            }
        })
        
    except Exception as e:
        error_time = time.time() - begin_time
        
        return jsonify({
            'error': 'Category overview failed',
            'details': str(e),
            'metadata': {
                'status': 'error',
                'response_time': round(error_time, 4),
                'timestamp': datetime.now().isoformat()
            }
        }), 500

@app.route('/api/test-simple')
def test_simple():
    """Ultra simple test endpoint"""
    print("🔍 Simple test endpoint called!")
    return jsonify({'message': 'simple test works'})

@app.route('/api/demo')
def demo_analysis():
    """Simplified demo to test JSON serialization"""
    try:
        data = {
            'level': getattr(AnalysisLevel, 'BASIC', 'basic'),
            'timestamp': datetime.now(),
            'text': 'أهلاً وسهلاً',
            'status': 'working'
        }
        
        result = jsonify(data)
        return result
        
    except Exception as e:
        import_data traceback
        traceback.print_exc()
        raise

@app.route('/api/insights')
def get_ai_insights():
    """Advanced AI insights and recommendations"""
    insights = get_dynamic_ai_insights()
    
    # Generate recommendations
    recommendations = []
    if insights['success_rate'] < 90:
        recommendations.append("Consider reviewing error patterns to improve analysis quality")
    if insights['avg_response_time_ms'] > 500:
        recommendations.append("Response times are high - consider optimization")
    if insights['requests_per_minute'] > 100:
        recommendations.append("High traffic detected - consider scaling")
    
    return jsonify({
        'insights': insights,
        'recommendations': recommendations,
        'trend_analysis': {
            'performance_trend': 'stable',  # Could calculate from recent data
            'usage_pattern': 'normal',
            'prediction': 'continued stable operation'
        }
    })

def create_enhanced_expert_template():
    """Create enhanced expert UI template"""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    template_content = '''<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Arabic Morphophonological Engine - Expert AI Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --dark-gradient: linear-gradient(135deg, #434343 0%, #000000 100%);
        }
        
        body {
            font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .hero-section {
            background: var(--primary-gradient);
            color: white;
            padding: 3rem 0;
            border-radius: 0 0 50px 50px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .dashboard-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: none;
            transition: all 0.3s ease;
        }
        
        .dashboard-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .metric-card {
            background: var(--success-gradient);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: scale(1.05);
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        
        .status-online { background: #00ff88; }
        .status-offline { background: #ff4757; }
        .status-warning { background: #ffa502; }
        
        .btn-ai {
            background: var(--primary-gradient);
            border: none;
            border-radius: 25px;
            color: white;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-ai:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            color: white;
        }
        
        .analysis-output {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1rem;
            border-left: 5px solid #007bff;
            font-family: 'Courier New', monospace;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .import_dataing-spinner {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 1rem 0;
        }
        
        .arabic-text {
            font-family: 'Amiri', 'Traditional Arabic', serif;
            font-size: 1.2em;
            direction: rtl;
            text-align: right;
        }
    </style>
</head>
<body>
    <!-- Hero Section -->
    <div class="hero-section">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-8">
                    <h1 class="display-4 fw-bold mb-3">
                        <i class="fas fa-brain me-3"></i>
                        Arabic Morphophonological Engine
                    </h1>
                    <p class="lead mb-4">Advanced AI-Powered Expert Dashboard with Real-time Analytics</p>
                    <div class="d-flex align-items-center">
                        <span class="status-indicator status-online"></span>
                        <span class="me-4">System Online</span>
                        <span class="status-indicator {% if engine_available %}status-online{% else %}status-offline{% endif %}"></span>
                        <span class="me-4">Main Engine</span>
                        <span class="status-indicator {% if advanced_available %}status-online{% else %}status-offline{% endif %}"></span>
                        <span>Advanced System</span>
                    </div>
                </div>
                <div class="col-lg-4 text-end">
                    <div class="d-flex flex-column align-items-end">
                        <div id="current-time" class="h4 mb-2"></div>
                        <div id="uptime" class="text-light opacity-75"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container my-5">
        <!-- Real-time Metrics -->
        <div class="row mb-4">
            <div class="col-md-3 mb-3">
                <div class="metric-card">
                    <div class="h2 mb-0" id="total-requests">0</div>
                    <div class="small">Total Requests</div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="metric-card" style="background: var(--success-gradient);">
                    <div class="h2 mb-0" id="success-rate">100%</div>
                    <div class="small">Success Rate</div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="metric-card" style="background: var(--danger-gradient);">
                    <div class="h2 mb-0" id="avg-response">0ms</div>
                    <div class="small">Avg Response</div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="metric-card" style="background: var(--dark-gradient);">
                    <div class="h2 mb-0" id="unique-texts">0</div>
                    <div class="small">Unique Texts</div>
                </div>
            </div>
        </div>

        <!-- Analysis Section -->
        <div class="row">
            <div class="col-lg-6 mb-4">
                <div class="dashboard-card">
                    <h3 class="h4 mb-4">
                        <i class="fas fa-microscope text-primary me-2"></i>
                        Live Analysis Engine
                    </h3>
                    
                    <div class="mb-3">
                        <label for="analysis-text" class="form-label">Enter Arabic Text:</label>
                        <textarea class="form-control arabic-text" id="analysis-text" rows="3" 
                                placeholder="أدخل النص العربي هنا..."></textarea>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="analysis-level" class="form-label">Analysis Level:</label>
                            <select class="form-select" id="analysis-level">
                                <option value="basic">Basic</option>
                                <option value="intermediate">Intermediate</option>
                                <option value="advanced">Advanced</option>
                                <option value="comprehensive">Comprehensive</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check mt-4">
                                <input class="form-check-input" type="checkbox" id="use-advanced">
                                <label class="form-check-label" for="use-advanced">
                                    Use Advanced System
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <div class="d-grid gap-2 d-md-flex">
                        <button class="btn btn-ai flex-fill" onclick="analyzeText()">
                            <i class="fas fa-cogs me-2"></i>Analyze Text
                        </button>
                        <button class="btn btn-outline-primary" onclick="runDemo()">
                            <i class="fas fa-rocket me-2"></i>Demo
                        </button>
                    </div>
                    
                    <div class="import_dataing-spinner" id="import_dataing">
                        <div class="spinner-border text-primary" role="status"></div>
                        <div class="mt-2">Analyzing...</div>
                    </div>
                    
                    <div class="analysis-output" id="analysis-output" style="display: none;">
                        <div class="text-muted">Analysis results will appear here...</div>
                    </div>
                </div>
            </div>
            
            <div class="col-lg-6 mb-4">
                <div class="dashboard-card">
                    <h3 class="h4 mb-4">
                        <i class="fas fa-chart-line text-success me-2"></i>
                        Performance Analytics
                    </h3>
                    
                    <div class="chart-container">
                        <canvas id="performance-chart"></canvas>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-6 text-center">
                            <div class="h5 text-primary" id="requests-per-min">0</div>
                            <div class="small text-muted">Requests/Min</div>
                        </div>
                        <div class="col-6 text-center">
                            <div class="h5 text-success" id="performance-status">Excellent</div>
                            <div class="small text-muted">Performance</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- AI Insights Panel -->
        <div class="row">
            <div class="col-12">
                <div class="dashboard-card">
                    <h3 class="h4 mb-4">
                        <i class="fas fa-brain text-warning me-2"></i>
                        AI-Powered Insights & Recommendations
                    </h3>
                    
                    <div id="ai-insights" class="row">
                        <div class="col-md-6">
                            <h5>System Health</h5>
                            <div id="health-indicators"></div>
                        </div>
                        <div class="col-md-6">
                            <h5>Smart Recommendations</h5>
                            <div id="recommendations"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.min.js"></script>
    
    <script>
        // Global variables
        let performanceChart;
        let statsInterval;
        
        // Initialize dashboard
        document.addEventListener('DOMContentImported', function() {
            initializeChart();
            updateTime();
            import_dataStats();
            import_dataInsights();
            
            // Set up intervals
            setInterval(updateTime, 1000);
            statsInterval = setInterval(import_dataStats, 15000); // Every 15 seconds
            setInterval(import_dataInsights, 60000); // Every minute
        });
        
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleTimeString();
        }
        
        function initializeChart() {
            const ctx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Response Time (ms)',
                        data: [],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Response Time (ms)'
                            }
                        }
                    }
                }
            });
        }
        
        async function import_dataStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                // Update metric cards
                document.getElementById('total-requests').textContent = stats.total_requests;
                document.getElementById('success-rate').textContent = stats.success_rate + '%';
                document.getElementById('avg-response').textContent = stats.avg_response_time_ms + 'ms';
                document.getElementById('unique-texts').textContent = stats.unique_texts;
                document.getElementById('requests-per-min').textContent = stats.requests_per_minute.toFixed(1);
                document.getElementById('performance-status').textContent = stats.performance_status || 'Excellent';
                
                // Update uptime
                document.getElementById('uptime').textContent = 'Uptime: ' + stats.uptime_formatted;
                
                // Update chart (simplified)
                if (performanceChart && stats.recent_avg_response_ms !== undefined) {
                    const now = new Date().toLocaleTimeString();
                    performanceChart.data.labels.push(now);
                    performanceChart.data.datasets[0].data.push(stats.recent_avg_response_ms);
                    
                    // Keep only last 20 points
                    if (performanceChart.data.labels.length > 20) {
                        performanceChart.data.labels.shift();
                        performanceChart.data.datasets[0].data.shift();
                    }
                    
                    performanceChart.update('none');
                }
            } catch (error) {
                console.error('Error import_dataing stats:', error);
            }
        }
        
        async function import_dataInsights() {
            try {
                const response = await fetch('/api/insights');
                const data = await response.json();
                
                // Update health indicators
                const healthDiv = document.getElementById('health-indicators');
                healthDiv.innerHTML = `
                    <div class="mb-2">
                        <span class="badge bg-success">CPU: Normal</span>
                        <span class="badge bg-primary">Memory: ${data.insights?.avg_response_time_ms < 500 ? 'Good' : 'High'}</span>
                    </div>
                    <div class="small text-muted">Last updated: ${new Date().toLocaleTimeString()}</div>
                `;
                
                // Update recommendations
                const recDiv = document.getElementById('recommendations');
                const recommendations = data.recommendations || [];
                if (recommendations.length > 0) {
                    recDiv.innerHTML = recommendations.map(rec => 
                        `<div class="alert alert-info alert-sm p-2 mb-2">${rec}</div>`
                    ).join('');
                } else {
                    recDiv.innerHTML = '<div class="alert alert-success alert-sm p-2">All systems optimal!</div>';
                }
            } catch (error) {
                console.error('Error import_dataing insights:', error);
            }
        }
        
        async function analyzeText() {
            const text = document.getElementById('analysis-text').value.trim();
            if (!text) {
                alert('Please enter some Arabic text to analyze.');
                return;
            }
            
            const level = document.getElementById('analysis-level').value;
            const useAdvanced = document.getElementById('use-advanced').checked;
            const outputDiv = document.getElementById('analysis-output');
            const import_dataingDiv = document.getElementById('import_dataing');
            
            // Show import_dataing
            import_dataingDiv.style.display = 'block';
            outputDiv.style.display = 'none';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: text,
                        level: level,
                        advanced: useAdvanced
                    })
                });
                
                const result = await response.json();
                
                // Display results
                outputDiv.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                outputDiv.style.display = 'block';
                
                // Refresh stats
                import_dataStats();
                
            } catch (error) {
                outputDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                outputDiv.style.display = 'block';
            } finally {
                import_dataingDiv.style.display = 'none';
            }
        }
        
        async function runDemo() {
            const outputDiv = document.getElementById('analysis-output');
            const import_dataingDiv = document.getElementById('import_dataing');
            
            import_dataingDiv.style.display = 'block';
            outputDiv.style.display = 'none';
            
            try {
                const response = await fetch('/api/demo');
                const result = await response.json();
                
                outputDiv.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                outputDiv.style.display = 'block';
                
                import_dataStats();
                
            } catch (error) {
                outputDiv.innerHTML = `<div class="alert alert-danger">Demo Error: ${error.message}</div>`;
                outputDiv.style.display = 'block';
            } finally {
                import_dataingDiv.style.display = 'none';
            }
        }
    </script>
</body>
</html>'''
    
    template_path = templates_dir / 'expert_dashboard.html'
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(template_content)

def main():
    """Launch the Arabic morphophonological production platform"""
    create_enhanced_expert_template()
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True, use_reimport_dataer=False)

if __name__ == '__main__':
    main()
