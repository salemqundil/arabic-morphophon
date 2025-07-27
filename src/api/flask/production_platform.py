#!/usr/bin/env python3
"""
Advanced Arabic Morphophonological Engine - Production Platform Launcher

A comprehensive production platform for Arabic morphophonological analysis
with expert UI, real-time monitoring, analytics, and advanced features.

Author: Arabic NLP Team
Version: 2.0.0
Date: July 2025
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data logging
import_data os
import_data sys

try:
    import_data webbrowser
    WEBBROWSER_AVAILABLE = True
except ImportError:
    webbrowser = None
    WEBBROWSER_AVAILABLE = False
from datetime import_data datetime
from pathlib import_data Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import robust Flask utilities with complete fallback protection
try:
    from utils.flask_utils import_data (
        CORS,
        FLASK_AVAILABLE,
        Flask,
        create_safe_flask_app,
        jsonify,
        render_template,
        request,
        safe_request_get_json,
        send_from_directory,
    )
    print("✅ Flask utilities import_dataed successfully")
except ImportError as e:
    print(f"⚠️ Flask utilities not available: {e}")
    # Create minimal fallbacks
    FLASK_AVAILABLE = False
    Flask = None
    CORS = None
    render_template = lambda *args, **kwargs: "Template not available"
    request = None
    send_from_directory = lambda *args, **kwargs: "File not available"
    jsonify = lambda x: str(x)
    create_safe_flask_app = lambda *args, **kwargs: None
    safe_request_get_json = lambda: {}


# Configure logging for production
class UTF8StreamProcessr(logging.StreamProcessr):
    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(self.stream, 'reconfigure'):
            self.stream.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    processrs=[
        logging.FileProcessr('production_platform.log', encoding='utf-8'),
        UTF8StreamProcessr()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with production config
app = create_safe_flask_app(__name__, template_folder='templates', static_folder='static')

if app is not None:
    print("✅ Flask app created successfully")
else:
    print("⚠️ Flask not available - running in minimal mode")

# Import analysis components with fallbacks
try:
    from arabic_morphophon.integrator import_data AnalysisLevel, MorphophonologicalEngine
    ANALYSIS_ENGINE_AVAILABLE = True
    engine = MorphophonologicalEngine()
    logger.info("✅ Main analysis engine import_dataed successfully")
except ImportError as e:
    ANALYSIS_ENGINE_AVAILABLE = False
    engine = None
    AnalysisLevel = None
    logger.warning(f"⚠️ Analysis engine not available: {e}")

try:
    from web_apps.advanced_hierarchical_api import_data AdvancedArabicAnalysisSystem
    ADVANCED_SYSTEM_AVAILABLE = True
    advanced_system = AdvancedArabicAnalysisSystem()
    logger.info("✅ Advanced system import_dataed successfully")
except ImportError as e:
    ADVANCED_SYSTEM_AVAILABLE = False
    advanced_system = None
    logger.warning(f"⚠️ Advanced system not available: {e}")

# Global statistics tracking
PLATFORM_STATS = {
    'begin_time': datetime.now(),
    'total_requests': 0,
    'analysis_requests': 0,
    'successful_analyses': 0,
    'failed_analyses': 0,
    'unique_texts_analyzed': set(),
    'average_response_time': 0,
    'last_analysis_time': None
}

def update_stats(success=True, response_time=0.0, text=None):
    """Update platform statistics with dynamic tracking"""
    PLATFORM_STATS['total_requests'] += 1
    if text:
        PLATFORM_STATS['analysis_requests'] += 1
        PLATFORM_STATS['unique_texts_analyzed'].add(text[:50])  # Truncate for memory
        
        if success:
            PLATFORM_STATS['successful_analyses'] += 1
        else:
            PLATFORM_STATS['failed_analyses'] += 1
            
        # Dynamic response time calculation
        current_avg = PLATFORM_STATS['average_response_time']
        total_analyses = PLATFORM_STATS['analysis_requests']
        PLATFORM_STATS['average_response_time'] = (current_avg * (total_analyses - 1) + response_time) / total_analyses
        PLATFORM_STATS['last_analysis_time'] = datetime.now()

def get_dynamic_performance_metrics():
    """Calculate advanced performance metrics dynamically"""
    now = datetime.now()
    uptime = now - PLATFORM_STATS['begin_time']
    
    success_rate = 0
    if PLATFORM_STATS['analysis_requests'] > 0:
        success_rate = (PLATFORM_STATS['successful_analyses'] / PLATFORM_STATS['analysis_requests']) * 100
    
    requests_per_minute = 0
    if uptime.total_seconds() > 0:
        requests_per_minute = PLATFORM_STATS['total_requests'] / (uptime.total_seconds() / 60)
    
    return {
        'uptime_seconds': int(uptime.total_seconds()),
        'uptime_formatted': str(uptime).split('.')[0],
        'success_rate': round(success_rate, 2),
        'requests_per_minute': round(requests_per_minute, 2),
        'unique_texts': len(PLATFORM_STATS['unique_texts_analyzed']),
        'avg_response_time_ms': round(PLATFORM_STATS['average_response_time'] * 1000, 2)
    }

# Flask routes (only available if Flask is installed and app is created)
if FLASK_AVAILABLE and app is not None:
    
    @app.route('/')
    def dashboard():
        """Expert UI Dashboard"""
        return render_template('expert_dashboard.html', 
                             stats=get_platform_stats(),
                             engines_available={
                                 'main_engine': ANALYSIS_ENGINE_AVAILABLE,
                                 'advanced_system': ADVANCED_SYSTEM_AVAILABLE
                             })

    @app.route('/api/health')
    def health_check():
        """Comprehensive health check endpoint"""
        uptime = datetime.now() - PLATFORM_STATS['begin_time']
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime.total_seconds(),
            'uptime_human': str(uptime),
            'engines': {
                'main_analysis_engine': ANALYSIS_ENGINE_AVAILABLE,
                'advanced_neural_system': ADVANCED_SYSTEM_AVAILABLE
            },
            'statistics': get_platform_stats(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': str(Path.cwd())
            }
        }
        
        update_stats(success=True)
        return jsonify(health_data)

    @app.route('/api/analyze', methods=['POST'])
    def analyze_text():
        """Advanced text analysis endpoint"""
        begin_time = datetime.now()
        data = None
        
        try:
            data = safe_request_get_json()
            if not data or 'text' not in data:
                return jsonify({
                    'error': 'Missing text parameter',
                    'status': 'error'
                }), 400
            
            text = data['text'].strip()
            if not text:
                return jsonify({
                    'error': 'Empty text provided',
                    'status': 'error'
                }), 400
            
            analysis_level = data.get('level', 'BASIC')
            use_advanced = data.get('use_advanced', False)
            
            result = {}
            # Main engine analysis
            if ANALYSIS_ENGINE_AVAILABLE and engine is not None and AnalysisLevel is not None:
                try:
                    level_enum = getattr(AnalysisLevel, analysis_level.upper(), AnalysisLevel.BASIC)
                    main_result = engine.analyze(text, level=level_enum)
                    result['traditional_analysis'] = {
                        'identified_roots': [str(root) for root in main_result.identified_roots],
                        'extract_phonemes': main_result.extract_phonemes,
                        'processing_time': main_result.processing_time,
                        'confidence_score': main_result.confidence_score
                    }
                except Exception as e:
                    result['traditional_analysis'] = {'error': str(e)}
            else:
                result['traditional_analysis'] = {'error': 'Analysis engine not available'}
            
            # Advanced system analysis
            if use_advanced and ADVANCED_SYSTEM_AVAILABLE and advanced_system is not None:
                try:
                    # Use the correct method name: analyze_hierarchical
                    advanced_result = advanced_system.analyze_hierarchical(text)
                    result['advanced_analysis'] = advanced_result
                except AttributeError:
                    # Fallback to traditional analysis method if available
                    try:
                        advanced_result = advanced_system.analyze_traditional(text)
                        result['advanced_analysis'] = advanced_result
                    except Exception as e:
                        result['advanced_analysis'] = {'error': f'Advanced analysis method not available: {str(e)}'}
                except Exception as e:
                    result['advanced_analysis'] = {'error': str(e)}
            else:
                result['advanced_analysis'] = {'error': 'Advanced system not available or not requested'}
            
            # Calculate response time
            end_time = datetime.now()
            response_time = (end_time - begin_time).total_seconds()
            
            result.update({
                'input_text': text,
                'analysis_level': analysis_level,
                'timestamp': end_time.isoformat(),
                'response_time_seconds': response_time,
                'status': 'success'
            })
            
            update_stats(success=True, response_time=response_time, text=text)
            return jsonify(result)
            
        except Exception as e:
            error_time = (datetime.now() - begin_time).total_seconds()
            error_text = ''
            if data is not None:
                try:
                    error_text = data.get('text', '')
                except Exception:
                    error_text = ''
            
            update_stats(success=False, response_time=error_time, text=error_text)
            
            logger.error(f"Analysis error: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/stats')
    def get_stats():
        """Platform statistics endpoint"""
        update_stats(success=True)
        return jsonify(get_platform_stats())

    @app.route('/api/demo')
    def demo_analysis():
        """Demo endpoint with sample texts"""
        sample_texts = [
            "أَكَلَ الوَلَدُ التُّفاحَ",
            "السَّلامُ عَلَيْكُم",
            "مَرْحَباً بِالعالَم",
            "كتب",
            "قرأ",
            "درس"
        ]
        
        results = []
        for text in sample_texts:
            if ANALYSIS_ENGINE_AVAILABLE and engine is not None and AnalysisLevel is not None:
                try:
                    result = engine.analyze(text, level=AnalysisLevel.BASIC)
                    results.append({
                        'text': text,
                        'roots': [str(root) for root in result.identified_roots],
                        'processing_time': result.processing_time
                    })
                except Exception as e:
                    results.append({
                        'text': text,
                        'error': str(e)
                    })
            else:
                results.append({
                    'text': text,
                    'error': 'Analysis engine not available'
                })
        
        update_stats(success=True)
        return jsonify({
            'demo_results': results,
            'total_samples': len(sample_texts),
            'timestamp': datetime.now().isoformat()
        })

    @app.errorprocessr(404)
    def not_found(error):
        """Process 404 errors"""
        return jsonify({
            'error': 'Endpoint not found',
            'status': 'error',
            'available_endpoints': [
                '/api/health',
                '/api/analyze',
                '/api/stats',
                '/api/demo',
                '/'
            ],
            'timestamp': datetime.now().isoformat()
        }), 404

    @app.errorprocessr(500)
    def internal_error(error):
        """Process internal server errors"""
        return jsonify({
            'error': 'Internal server error',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

# Platform statistics function (outside Flask conditional)
def get_platform_stats():
    """Get formatted platform statistics"""
    uptime = datetime.now() - PLATFORM_STATS['begin_time']
    
    return {
        'uptime_seconds': uptime.total_seconds(),
        'uptime_human': str(uptime),
        'total_requests': PLATFORM_STATS['total_requests'],
        'analysis_requests': PLATFORM_STATS['analysis_requests'],
        'successful_analyses': PLATFORM_STATS['successful_analyses'],
        'failed_analyses': PLATFORM_STATS['failed_analyses'],
        'success_rate': (PLATFORM_STATS['successful_analyses'] / max(1, PLATFORM_STATS['analysis_requests'])) * 100,
        'unique_texts_count': len(PLATFORM_STATS['unique_texts_analyzed']),
        'average_response_time': PLATFORM_STATS['average_response_time'],
        'last_analysis_time': PLATFORM_STATS['last_analysis_time'].isoformat() if PLATFORM_STATS['last_analysis_time'] else None
    }

# Template creation function
def create_expert_ui_template():
    """Create expert UI dashboard template"""
    template_dir = Path('templates')
    template_dir.mkdir(exist_ok=True)
    
    expert_ui_html = '''<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arabic Morphophonological Engine - Expert Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8f9fa; }
        .dashboard-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; }
        .stat-card { background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
        .stat-card:hover { transform: translateY(-5px); }
        .engine-status { padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
        .engine-available { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .engine-unavailable { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .analysis-section { background: white; border-radius: 15px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .btn-analyze { background: linear-gradient(45deg, #FF6B6B, #4ECDC4); border: none; color: white; font-weight: bold; }
        .btn-analyze:hover { background: linear-gradient(45deg, #FF5252, #26C6DA); color: white; }
        .result-container { background: #f8f9fa; border-radius: 10px; padding: 1.5rem; margin-top: 1rem; }
        .arabic-text { font-size: 1.2em; font-weight: bold; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1><i class="fas fa-brain"></i> Arabic Morphophonological Engine</h1>
                    <p class="lead">Expert Production Platform - Advanced Linguistic Analysis</p>
                </div>
                <div class="col-md-4 text-end">
                    <div class="badge bg-success fs-6">
                        <i class="fas fa-check-circle"></i> Production Ready
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- System Status -->
        <div class="row mb-4">
            <div class="col-12">
                <h3><i class="fas fa-server"></i> System Status</h3>
                <div class="row">
                    <div class="col-md-6">
                        <div class="engine-status {{ 'engine-available' if engines_available.main_engine else 'engine-unavailable' }}">
                            <strong>Main Analysis Engine:</strong>
                            {% if engines_available.main_engine %}
                                <i class="fas fa-check-circle text-success"></i> Available
                            {% else %}
                                <i class="fas fa-times-circle text-danger"></i> Unavailable
                            {% endif %}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="engine-status {{ 'engine-available' if engines_available.advanced_system else 'engine-unavailable' }}">
                            <strong>Advanced Neural System:</strong>
                            {% if engines_available.advanced_system %}
                                <i class="fas fa-check-circle text-success"></i> Available
                            {% else %}
                                <i class="fas fa-times-circle text-danger"></i> Unavailable
                            {% endif %}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Statistics Dashboard -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="stat-card p-3 h-100">
                    <div class="d-flex align-items-center">
                        <div class="flex-shrink-0">
                            <i class="fas fa-clock fa-2x text-primary"></i>
                        </div>
                        <div class="flex-grow-1 ms-3">
                            <div class="fw-bold">Uptime</div>
                            <div class="text-muted" id="uptime">{{ stats.uptime_human }}</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card p-3 h-100">
                    <div class="d-flex align-items-center">
                        <div class="flex-shrink-0">
                            <i class="fas fa-chart-line fa-2x text-success"></i>
                        </div>
                        <div class="flex-grow-1 ms-3">
                            <div class="fw-bold">Total Requests</div>
                            <div class="text-muted">{{ stats.total_requests }}</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card p-3 h-100">
                    <div class="d-flex align-items-center">
                        <div class="flex-shrink-0">
                            <i class="fas fa-brain fa-2x text-info"></i>
                        </div>
                        <div class="flex-grow-1 ms-3">
                            <div class="fw-bold">Analyses</div>
                            <div class="text-muted">{{ stats.successful_analyses }}</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card p-3 h-100">
                    <div class="d-flex align-items-center">
                        <div class="flex-shrink-0">
                            <i class="fas fa-percentage fa-2x text-warning"></i>
                        </div>
                        <div class="flex-grow-1 ms-3">
                            <div class="fw-bold">Success Rate</div>
                            <div class="text-muted">{{ "%.1f"|format(stats.success_rate) }}%</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Analysis Interface -->
        <div class="analysis-section">
            <h3><i class="fas fa-search"></i> Advanced Text Analysis</h3>
            <div class="row">
                <div class="col-md-8">
                    <div class="mb-3">
                        <label for="analysisText" class="form-label">Arabic Text for Analysis</label>
                        <textarea class="form-control" id="analysisText" rows="3" 
                                placeholder="أدخل النص العربي هنا للتحليل...">أَكَلَ الوَلَدُ التُّفاحَ</textarea>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="mb-3">
                        <label for="analysisLevel" class="form-label">Analysis Level</label>
                        <select class="form-select" id="analysisLevel">
                            <option value="BASIC">Basic</option>
                            <option value="INTERMEDIATE">Intermediate</option>
                            <option value="ADVANCED">Advanced</option>
                            <option value="COMPREHENSIVE">Comprehensive</option>
                        </select>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="useAdvanced">
                        <label class="form-check-label" for="useAdvanced">
                            Use Advanced Neural System
                        </label>
                    </div>
                </div>
            </div>
            <button class="btn btn-analyze btn-lg" onclick="analyzeText()">
                <i class="fas fa-brain"></i> Analyze Text
            </button>
            
            <div id="analysisResults" class="result-container" style="display: none;">
                <h5><i class="fas fa-chart-bar"></i> Analysis Results</h5>
                <div id="resultsContent"></div>
            </div>
        </div>

        <!-- Quick Demo -->
        <div class="analysis-section">
            <h3><i class="fas fa-play-circle"></i> Quick Demo</h3>
            <p>Test the system with pre-defined Arabic texts:</p>
            <button class="btn btn-outline-primary" onclick="runDemo()">
                <i class="fas fa-rocket"></i> Run Demo Analysis
            </button>
            <div id="demoResults" class="result-container" style="display: none;">
                <div id="demoContent"></div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function analyzeText() {
            const text = document.getElementById('analysisText').value;
            const level = document.getElementById('analysisLevel').value;
            const useAdvanced = document.getElementById('useAdvanced').checked;
            
            if (!text.trim()) {
                alert('Please enter text to analyze');
                return;
            }
            
            const resultsDiv = document.getElementById('analysisResults');
            const contentDiv = document.getElementById('resultsContent');
            
            resultsDiv.style.display = 'block';
            contentDiv.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Analyzing...</div>';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, level, use_advanced: useAdvanced })
                });
                
                const result = await response.json();
                
                let html = `<div class="arabic-text">Input: ${result.input_text}</div>`;
                html += `<p><strong>Response Time:</strong> ${(result.response_time_seconds * 1000).toFixed(2)}ms</p>`;
                
                if (result.traditional_analysis) {
                    html += '<h6>Traditional Analysis:</h6>';
                    if (result.traditional_analysis.identified_roots) {
                        html += `<p><strong>Roots:</strong> ${result.traditional_analysis.identified_roots.join(', ')}</p>`;
                    }
                    if (result.traditional_analysis.confidence_score) {
                        html += `<p><strong>Confidence:</strong> ${(result.traditional_analysis.confidence_score * 100).toFixed(1)}%</p>`;
                    }
                }
                
                if (result.advanced_analysis) {
                    html += '<h6>Advanced Analysis:</h6>';
                    html += `<pre>${JSON.stringify(result.advanced_analysis, null, 2)}</pre>`;
                }
                
                contentDiv.innerHTML = html;
            } catch (error) {
                contentDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            }
        }
        
        async function runDemo() {
            const demoDiv = document.getElementById('demoResults');
            const contentDiv = document.getElementById('demoContent');
            
            demoDiv.style.display = 'block';
            contentDiv.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Running demo...</div>';
            
            try {
                const response = await fetch('/api/demo');
                const result = await response.json();
                
                let html = '<h6>Demo Results:</h6>';
                result.demo_results.forEach(item => {
                    html += `<div class="mb-2">`;
                    html += `<div class="arabic-text">${item.text}</div>`;
                    if (item.roots) {
                        html += `<small>Roots: ${item.roots.join(', ')} (${item.processing_time.toFixed(3)}s)</small>`;
                    } else if (item.error) {
                        html += `<small class="text-danger">Error: ${item.error}</small>`;
                    }
                    html += `</div>`;
                });
                
                contentDiv.innerHTML = html;
            } catch (error) {
                contentDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            }
        }
        
        // Auto-refresh stats every 30 seconds
        setInterval(async () => {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                document.getElementById('uptime').textContent = stats.uptime_human;
            } catch (error) {
                console.error('Failed to refresh stats:', error);
            }
        }, 30000);
    </script>
</body>
</html>'''
    
    with open(template_dir / 'expert_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(expert_ui_html)
    
    logger.info("✅ Expert UI template created")

def main():
    """Launch the production platform"""
    print("🚀 Arabic Morphophonological Engine - Production Platform")
    print("=" * 60)
    print("🎯 Expert UI with comprehensive monitoring")
    print("🧠 Advanced neural analysis capabilities")
    print("📊 Real-time statistics and performance tracking")
    print("⚡ Production-ready with zero violations")
    print("=" * 60)

    create_expert_ui_template()
    
    # Log system status
    logger.info(f"🎯 Main Engine Available: {ANALYSIS_ENGINE_AVAILABLE}")
    logger.info(f"🧠 Advanced System Available: {ADVANCED_SYSTEM_AVAILABLE}")
    
    # Open browser automatically
    try:
        import_data threading
        THREADING_AVAILABLE = True
    except ImportError:
        threading = None
        THREADING_AVAILABLE = False
    
    def open_browser():
        try:
            import_data time
            time.sleep(2)  # Wait for server to begin
            if WEBBROWSER_AVAILABLE and webbrowser is not None:
                webbrowser.open('http://localhost:5001')
        except Exception as e:
            print(f"⚠️ Could not open browser: {e}")
    
    if THREADING_AVAILABLE and threading is not None:
        threading.Thread(target=open_browser, daemon=True).begin()
    
    print("\n🌐 Begining Expert UI Platform...")
    print("📍 Dashboard: http://localhost:5001")
    print("🔗 API Base: http://localhost:5001/api")
    print("📊 Health Check: http://localhost:5001/api/health")
    print("\n🎉 Production platform ready!")
    
    # Run the application (only if Flask is available and app is created)
    if FLASK_AVAILABLE and app is not None:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    else:
        print("⚠️ Flask not available - cannot begin web server")
        print("📦 Install Flask to enable web interface: pip install flask flask-cors")


if __name__ == '__main__':
    main()