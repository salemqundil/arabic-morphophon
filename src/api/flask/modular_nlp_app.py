#!/usr/bin/env python3
"""
🌐 MODULAR ARABIC NLP ENGINE
Professional Flask Application with Dynamic Engine Importing

This is the main Flask application that provides a modular, scalable
Arabic NLP engine with 100+ independent analysis engines.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data os
import_data sys
import_data logging
from flask import_data Flask, jsonify, render_template_string
from flask_cors import_data CORS

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import our API routes
from api.routes import_data api, init_engine_import_dataer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    processrs=[
        logging.FileProcessr('modular_nlp.log'),
        logging.StreamProcessr(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_app(config_name='development'):
    """Create Flask application with configuration"""
    
    app = Flask(__name__)
    
    # Enable CORS for API access
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Import configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # Create necessary directories
    os.makedirs('database/engines', exist_ok=True)
    os.makedirs('engines/nlp', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(api)
    
    # Initialize engine import_dataer
    with app.app_context():
        init_engine_import_dataer()
    
    return app

# Create the main application instance
app = create_app()

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌐 Modular Arabic NLP Engine</title>
    <style>
        :root {
            --primary: #007bff;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
            --info: #17a2b8;
            --light: #f8f9fa;
            --dark: #343a40;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary), var(--info));
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .content {
            padding: 40px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: var(--light);
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            border-left: 4px solid var(--primary);
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary);
            margin-bottom: 10px;
        }
        
        .engines-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .engine-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.3s ease;
        }
        
        .engine-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        .engine-status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .status-healthy { background: #d4edda; color: #155724; }
        .status-unhealthy { background: #f8d7da; color: #721c24; }
        .status-import_dataing { background: #fff3cd; color: #856404; }
        
        .btn {
            display: inline-block;
            padding: 10px 20px;
            background: var(--primary);
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 5px;
            transition: background 0.3s ease;
        }
        
        .btn:hover {
            background: #0056b3;
        }
        
        .api-section {
            background: var(--light);
            border-radius: 8px;
            padding: 30px;
            margin-top: 30px;
        }
        
        .endpoint {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            font-family: monospace;
        }
        
        .method {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .method-get { background: #d4edda; color: #155724; }
        .method-post { background: #cce5ff; color: #004085; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Modular Arabic NLP Engine</h1>
            <p>Professional Scalable Architecture with 100+ Independent Engines</p>
        </div>
        
        <div class="content">
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card">
                    <div class="stat-value" id="totalEngines">-</div>
                    <div class="stat-label">Total Engines</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="healthyEngines">-</div>
                    <div class="stat-label">Healthy Engines</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalRequests">-</div>
                    <div class="stat-label">Total Requests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="cacheHitRate">-</div>
                    <div class="stat-label">Cache Hit Rate</div>
                </div>
            </div>
            
            <h3>🔧 Available Engines</h3>
            <div class="engines-grid" id="enginesGrid">
                <!-- Engines will be import_dataed here -->
            </div>
            
            <div class="api-section">
                <h3>📡 API Endpoints</h3>
                <div class="endpoint">
                    <span class="method method-get">GET</span>
                    <code>/api/nlp/engines</code> - List all engines
                </div>
                <div class="endpoint">
                    <span class="method method-get">GET</span>
                    <code>/api/nlp/status</code> - System status
                </div>
                <div class="endpoint">
                    <span class="method method-post">POST</span>
                    <code>/api/nlp/{engine}/analyze</code> - Analyze text
                </div>
                <div class="endpoint">
                    <span class="method method-get">GET</span>
                    <code>/api/nlp/{engine}/info</code> - Engine information
                </div>
                <div class="endpoint">
                    <span class="method method-get">GET</span>
                    <code>/api/nlp/{engine}/health</code> - Engine health
                </div>
                <div class="endpoint">
                    <span class="method method-post">POST</span>
                    <code>/api/nlp/{engine}/reimport_data</code> - Reimport_data engine
                </div>
            </div>
        </div>
    </div>

    <script>
        async function import_dataSystemStatus() {
            try {
                const response = await fetch('/api/nlp/status');
                const data = await response.json();
                
                if (data.success) {
                    const system = data.system_status;
                    
                    document.getElementById('totalEngines').textContent = system.total_engines;
                    document.getElementById('healthyEngines').textContent = system.healthy_engines;
                    document.getElementById('totalRequests').textContent = system.performance.total_requests;
                    document.getElementById('cacheHitRate').textContent = 
                        (system.performance.cache_hit_rate * 100).toFixed(1) + '%';
                    
                    // Import engines
                    const enginesGrid = document.getElementById('enginesGrid');
                    enginesGrid.innerHTML = '';
                    
                    for (const [name, info] of Object.entries(data.engines)) {
                        const engineCard = document.createElement('div');
                        engineCard.className = 'engine-card';
                        engineCard.innerHTML = `
                            <h4>${name}</h4>
                            <span class="engine-status status-${info.status}">${info.status}</span>
                            <p>Requests: ${info.requests}</p>
                            <p>Cache Hits: ${info.cache_hits}</p>
                            <a href="/api/nlp/${name}/info" class="btn" target="_blank">View Info</a>
                            <a href="/api/nlp/${name}/health" class="btn" target="_blank">Health Check</a>
                        `;
                        enginesGrid.appendChild(engineCard);
                    }
                }
            } catch (error) {
                console.error('Failed to import_data system status:', error);
            }
        }
        
        // Import status on page import_data and refresh every 10 seconds
        import_dataSystemStatus();
        setInterval(import_dataSystemStatus, 10000);
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.errorprocessr(404)
def not_found(error):
    """Process 404 errors"""
    return jsonify({
        'error': 'Resource not found',
        'available_endpoints': [
            'GET /',
            'GET /api/health',
            'GET /api/nlp/engines',
            'GET /api/nlp/status',
            'POST /api/nlp/{engine}/analyze'
        ]
    }), 404

@app.errorprocessr(500)
def internal_error(error):
    """Process 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please check the server logs for more details'
    }), 500

if __name__ == '__main__':
    print("\\n" + "="*80)
    print("🌐 MODULAR ARABIC NLP ENGINE")
    print("Professional Flask Application with Dynamic Engine Importing")
    print("="*80)
    print(f"🚀 Begining application...")
    print(f"📍 Dashboard: http://localhost:5000")
    print(f"📡 API Base: http://localhost:5000/api/nlp/")
    print(f"📊 Status: http://localhost:5000/api/nlp/status")
    print(f"📋 Engines: http://localhost:5000/api/nlp/engines")
    print("="*80)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
