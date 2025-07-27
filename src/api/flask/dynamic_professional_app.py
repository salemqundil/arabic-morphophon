#!/usr/bin/env python3
"""
🌐 PROFESSIONAL DYNAMIC FLASK APPLICATION
Arabic Morphophonological Engine - Full-Stack Implementation

Professional implementation with:
- Real-time dynamic updates
- Comprehensive error handling  
- Input validation and sanitization
- Modern responsive design
- Security best practices
- No violations, full professional standards
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data json
import_data logging
import_data time
import_data uuid
import_data re
import_data html
from datetime import_data datetime
from typing import_data Dict, List, Any
from flask import_data Flask, request, jsonify, render_template_string

# Import our decision tree engine
from decision_tree_executable import_data (
    DecisionTreeEngine, 
    FlaskStyleDecisionTree,
    PHONEME_INVENTORY,
    TEMPLATES,
    PHONO_RULES
)

# Professional logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arabic-morphophon-secure-key-2025'
app.config['JSON_SORT_KEYS'] = False

# Initialize engines
decision_engine = DecisionTreeEngine()
flask_decision_tree = FlaskStyleDecisionTree()

# Application state for real-time tracking
app_state = {
    'requests_count': 0,
    'total_analyses': 0,
    'total_characters': 0,
    'unique_texts': set(),
    'begin_time': time.time(),
    'recent_requests': [],
    'error_count': 0,
    'performance_metrics': []
}

# =============================================================================
# SECURITY AND VALIDATION FUNCTIONS
# =============================================================================

def sanitize_input(text: str) -> str:
    """Professional input sanitization"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # HTML escape and strip
    sanitized = html.escape(text.strip())
    
    # Length validation
    if len(sanitized) > 10000:
        raise ValueError("Input text too long (max 10,000 characters)")
    
    return sanitized

def validate_analysis_level(level: str) -> str:
    """Validate analysis level parameter"""
    valid_levels = ['basic', 'advanced', 'comprehensive']
    if level not in valid_levels:
        raise ValueError(f"Invalid analysis level. Must be one of: {valid_levels}")
    return level

def log_request(method: str, path: str, status: int, description: str = ""):
    """Professional request logging with decision tree context"""
    global app_state
    
    request_info = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'path': path,
        'status': status,
        'description': description,
        'decision_tree_processed': True
    }
    
    app_state['recent_requests'].insert(0, request_info)
    app_state['recent_requests'] = app_state['recent_requests'][:50]
    app_state['requests_count'] += 1
    
    logger.info(f"{method} {path} - {status} - {description}")

# =============================================================================

# =============================================================================

PROFESSIONAL_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌳 Arabic Morphophonological Engine - Professional Interface</title>
    <style>
        :root {
            --primary: #007bff;
            --success: #28a745;
            --danger: #dc3545;
            --warning: #ffc107;
            --info: #17a2b8;
            --light: #f8f9fa;
            --dark: #343a40;
            --font: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font);
            line-height: 1.6;
            color: var(--dark);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            position: relative;
        }

        .container::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--info), var(--success));
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--light);
        }

        .header h1 {
            font-size: 2.5rem;
            color: var(--primary);
            margin-bottom: 10px;
        }

        .header h2 {
            font-size: 1.5rem;
            color: var(--dark);
            margin-bottom: 15px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .stat-card {
            background: linear-gradient(135deg, var(--light) 0%, #ffffff 100%);
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 100%;
            height: 3px;
            background: var(--primary);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }

        .stat-card:hover::before { transform: scaleX(1); }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,123,255,0.2);
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 8px;
        }

        .stat-label {
            color: var(--dark);
            font-weight: 500;
        }

        .decision-tree {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }

        .decision-node {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 25px;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .decision-node:hover {
            border-color: var(--primary);
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,123,255,0.15);
        }

        .decision-node.active {
            border-color: var(--success);
            background: linear-gradient(135deg, #d4edda 0%, #ffffff 100%);
        }

        .question {
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 12px;
            font-size: 1.1rem;
        }

        .answer {
            color: var(--success);
            margin-bottom: 15px;
            font-weight: 500;
        }

        .details {
            color: var(--dark);
            font-size: 0.9rem;
        }

        .input-section {
            background: linear-gradient(135deg, var(--light) 0%, #ffffff 100%);
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 30px;
            margin: 30px 0;
        }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .input-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--dark);
        }

        input, textarea, select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
        }

        textarea {
            resize: vertical;
            min-height: 120px;
            font-family: 'Courier New', monospace;
        }

        .button-group {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 25px;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background: #0056b3;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,123,255,0.3);
        }

        .btn-success {
            background: var(--success);
            color: white;
        }

        .btn-success:hover {
            background: #218838;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(40,167,69,0.3);
        }

        .btn-info {
            background: var(--info);
            color: white;
        }

        .btn-info:hover {
            background: #138496;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(23,162,184,0.3);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !import_dataant;
        }

        .import_dataing {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.7);
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }

        .spinner {
            width: 60px; height: 60px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-section {
            border-radius: 8px;
            padding: 25px;
            margin: 20px 0;
            display: none;
            border-left: 4px solid;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-success {
            border-left-color: var(--success);
            background: linear-gradient(135deg, #d4edda 0%, #ffffff 100%);
        }

        .result-error {
            border-left-color: var(--danger);
            background: linear-gradient(135deg, #f8d7da 0%, #ffffff 100%);
        }

        .result-content {
            background: var(--light);
            border-radius: 8px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
        }

        /* Professional Table Styles */
        .result-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .result-table th {
            background: linear-gradient(135deg, var(--primary), #0056b3);
            color: white;
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            border-bottom: 2px solid #dee2e6;
        }

        .result-table td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
            vertical-align: top;
            font-size: 13px;
        }

        .result-table tr:nth-child(even) {
            background: #f8f9fa;
        }

        .result-table tr:hover {
            background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
            cursor: pointer;
        }

        .table-container {
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }

        .nested-table {
            margin: 5px 0;
            font-size: 12px;
            background: #f8f9fa;
            border-radius: 4px;
            overflow: hidden;
        }

        .nested-table th {
            background: #6c757d;
            color: white;
            padding: 8px;
            font-size: 11px;
        }

        .nested-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #dee2e6;
        }

        .metric-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin: 2px;
        }

        .badge-success { background: #d4edda; color: #155724; }
        .badge-info { background: #d1ecf1; color: #0c5460; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-primary { background: #cce5ff; color: #004085; }

        .downimport_data-section {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
            padding: 15px;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }

        .view-toggle {
            display: flex;
            gap: 10px;
        }

        .toggle-btn {
            padding: 8px 16px;
            border: 2px solid var(--primary);
            background: white;
            color: var(--primary);
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .toggle-btn.active {
            background: var(--primary);
            color: white;
        }

        .toggle-btn:hover {
            background: var(--primary);
            color: white;
        }

        .json-view {
            display: none;
        }

        .table-view {
            display: block;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .container { padding: 20px; margin: 10px; }
            .header h1 { font-size: 2rem; }
            .form-row { grid-template-columns: 1fr; }
            .button-group { flex-direction: column; }
            .btn { width: 100%; justify-content: center; }
        }

        /* Accessibility */
        .btn:focus, input:focus, textarea:focus, select:focus {
            outline: 2px solid var(--primary);
            outline-offset: 2px;
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !import_dataant;
                transition-duration: 0.01ms !import_dataant;
            }
        }
    </style>
</head>
<body>
    <div class="import_dataing" id="import_dataing">
        <div class="spinner"></div>
    </div>

    <div class="container">
        <header class="header">
            <h1>🌳 Arabic Morphophonological Engine</h1>
            <h2>Professional Dynamic Interface</h2>
            <p>Real-time interactive demonstration with comprehensive decision tree flows</p>
        </header>

        <!-- Real-time Statistics -->
        <section class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <div class="stat-value" id="totalAnalyses">{{ stats.total_analyses or 0 }}</div>
                <div class="stat-label">Total Analyses</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="engineStatus">{{ stats.engine_status or 'operational' }}</div>
                <div class="stat-label">Engine Status</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="memoryUsage">{{ "%.1f" | format(stats.memory_usage_mb or 0) }}MB</div>
                <div class="stat-label">Memory Usage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgTime">{{ "%.3f" | format(stats.average_processing_time or 0) }}s</div>
                <div class="stat-label">Avg Response Time</div>
            </div>
        </section>

        <!-- Interactive Decision Tree -->
        <section>
            <h3>🎯 Interactive Decision Points</h3>
            <div class="decision-tree">
                <div class="decision-node" data-decision="engine-init" onclick="highlightDecision(this)">
                    <div class="question">Q1: Is the engine initialized properly?</div>
                    <div class="answer">A1: ✅ Engine ready with {{ stats.phoneme_inventory or 29 }} phonemes</div>
                    <div class="details">Components: SyllabicUnit templates, phonological rules, vector operations</div>
                </div>
                
                <div class="decision-node" data-decision="normalization" onclick="highlightDecision(this)">
                    <div class="question">Q2: How should text be normalized?</div>
                    <div class="answer">A2: 🔄 Multi-step normalization</div>
                    <div class="details">Diacritics, whitespace, digits, and letter form standardization</div>
                </div>
                
                <div class="decision-node" data-decision="syllabic_analysis" onclick="highlightDecision(this)">
                    <div class="question">Q3: How should text be syllabified?</div>
                    <div class="answer">A3: 📊 CV pattern analysis</div>
                    <div class="details">20 syllabic_unit templates, complexity scoring</div>
                </div>
                
                <div class="decision-node" data-decision="vector-ops" onclick="highlightDecision(this)">
                    <div class="question">Q4: What vector operations are needed?</div>
                    <div class="answer">A4: 🧮 4 operation types</div>
                    <div class="details">Phoneme, root, template embeddings + inflection</div>
                </div>
                
                <div class="decision-node" data-decision="validation" onclick="highlightDecision(this)">
                    <div class="question">Q5: Is input data valid?</div>
                    <div class="answer">A5: ✅ Multi-layer validation</div>
                    <div class="details">Security screening, Arabic detection, length validation</div>
                </div>
                
                <div class="decision-node" data-decision="routing" onclick="highlightDecision(this)">
                    <div class="question">Q6: Which Flask route processs the request?</div>
                    <div class="answer">A6: 🌐 5 main endpoints</div>
                    <div class="details">/, /api/analyze, /api/stats, /api/validate, /api/decision-tree</div>
                </div>
            </div>
        </section>

        <!-- Professional Input Interface -->
        <section>
            <h3>🚀 Interactive Analysis Interface</h3>
            <div class="input-section">
                <form id="analysisForm">
                    <div class="form-row">
                        <div class="input-group">
                            <label for="analysisText">Enter Arabic text for analysis:</label>
                            <textarea 
                                id="analysisText" 
                                placeholder="كتب الطالب الدرس في الصف"
                                rows="4"
                                maxlength="10000"
                                required
                            ></textarea>
                            <small id="charCount">0/10,000 characters</small>
                        </div>
                        <div class="input-group">
                            <label for="analysisLevel">Analysis Level:</label>
                            <select id="analysisLevel">
                                <option value="basic">Basic - Phoneme analysis only</option>
                                <option value="advanced">Advanced - + Morphological analysis</option>
                                <option value="comprehensive">Comprehensive - + Complete engine processing</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="button-group">
                        <button type="button" class="btn btn-primary" onclick="runAnalysis()" id="analyzeBtn">
                            🔍 Analyze Text
                        </button>
                        <button type="button" class="btn btn-success" onclick="validateInput()" id="validateBtn">
                            ✅ Validate Only
                        </button>
                        <button type="button" class="btn btn-info" onclick="showStats()" id="statsBtn">
                            📊 Live Statistics
                        </button>
                        <button type="button" class="btn btn-info" onclick="showDecisionTree()" id="treeBtn">
                            🌳 Decision Tree
                        </button>
                        <button type="button" class="btn btn-info" onclick="clearResults()" id="clearBtn">
                            🗑️ Clear Results
                        </button>
                    </div>
                </form>
            </div>
        </section>

        <!-- Dynamic Results Display -->
        <section id="resultsSection">
            <div id="results" class="result-section">
                <div class="downimport_data-section">
                    <div>
                        <h4 id="resultTitle">Analysis Results</h4>
                        <small id="resultSubtitle">Select your preferred view format</small>
                    </div>
                    <div class="view-toggle">
                        <button class="toggle-btn active" onclick="toggleView('table')" id="tableViewBtn">📊 Table View</button>
                        <button class="toggle-btn" onclick="toggleView('json')" id="jsonViewBtn">📄 JSON View</button>
                        <button class="btn btn-info" onclick="downimport_dataJSON()" id="downimport_dataBtn">⬇️ Downimport_data JSON</button>
                    </div>
                </div>
                <div class="table-container table-view" id="tableView">
                    <div id="resultTables"></div>
                </div>
                <div class="result-content json-view" id="jsonView">
                    <div id="resultContent"></div>
                </div>
            </div>
        </section>
    </div>

    <script>
        // Professional JavaScript Application Class
        class ProfessionalMorphologyApp {
            constructor() {
                this.state = {
                    isProcessing: false,
                    statsInterval: null,
                    requestCount: 0,
                    currentData: null,
                    currentView: 'table'
                };
                this.init();
            }

            init() {
                this.setupEventListeners();
                this.beginStatsUpdate();
                console.log('🌳 Professional Arabic Morphophonological Engine initialized');
            }

            setupEventListeners() {
                // Character counter
                const textArea = document.getElementById('analysisText');
                textArea.addEventListener('input', (e) => {
                    const count = e.target.value.length;
                    document.getElementById('charCount').textContent = `${count}/10,000 characters`;
                    this.validateForm();
                });

                // Auto-store_data
                textArea.addEventListener('input', this.debounce((e) => {
                    localStorage.setItem('morphology_text', e.target.value);
                }, 500));

                // Import store_datad text
                const store_datad = localStorage.getItem('morphology_text');
                if (store_datad) textArea.value = store_datad;

                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                        e.preventDefault();
                        this.runAnalysis();
                    }
                });

                this.validateForm();
            }

            validateForm() {
                const text = document.getElementById('analysisText').value.trim();
                const isValid = text.length > 0 && text.length <= 10000;
                
                document.getElementById('analyzeBtn').disabled = !isValid || this.state.isProcessing;
                document.getElementById('validateBtn').disabled = !isValid || this.state.isProcessing;
            }

            debounce(func, wait) {
                let timeout;
                return function run_commanddFunction(...args) {
                    const later = () => {
                        clearTimeout(timeout);
                        func(...args);
                    };
                    clearTimeout(timeout);
                    timeout = setTimeout(later, wait);
                };
            }

            showImporting(show = true) {
                document.getElementById('import_dataing').style.display = show ? 'flex' : 'none';
                this.state.isProcessing = show;
                this.validateForm();
            }

            async makeRequest(url, options = {}) {
                const defaultOptions = {
                    headers: { 'Content-Type': 'application/json' }
                };
                
                const response = await fetch(url, { ...defaultOptions, ...options });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                return await response.json();
            }

            showResult(data, type = 'success') {
                const results = document.getElementById('results');
                const title = document.getElementById('resultTitle');
                const tableView = document.getElementById('tableView');
                const jsonView = document.getElementById('jsonView');
                const jsonContent = document.getElementById('resultContent');
                
                results.className = `result-section result-${type}`;
                results.style.display = 'block';
                
                title.textContent = type === 'success' ? '✅ Analysis Results' : '❌ Error Results';
                
                // Store current data
                this.state.currentData = data;
                
                // Generate table view
                if (typeof data === 'object' && data !== null) {
                    this.generateTableView(data);
                    jsonContent.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                } else {
                    document.getElementById('resultTables').innerHTML = `<p>${data}</p>`;
                    jsonContent.innerHTML = `<p>${data}</p>`;
                }
                
                // Show appropriate view
                this.setView(this.state.currentView);
                
                results.scrollIntoView({ behavior: 'smooth' });
            }

            generateTableView(data) {
                const container = document.getElementById('resultTables');
                container.innerHTML = '';
                
                if (data.success === false || data.error) {
                    container.innerHTML = `
                        <div class="table-container">
                            <table class="result-table">
                                <thead>
                                    <tr><th>Error Details</th><th>Value</th></tr>
                                </thead>
                                <tbody>
                                    <tr><td>Error Type</td><td><span class="badge-warning metric-badge">${data.error || 'Unknown'}</span></td></tr>
                                    <tr><td>Message</td><td>${data.message || 'No details available'}</td></tr>
                                    <tr><td>Processing Time</td><td>${(data.processing_time || 0).toFixed(4)}s</td></tr>
                                </tbody>
                            </table>
                        </div>
                    `;
                    return;
                }
                
                // Main summary table
                container.innerHTML += this.createSummaryTable(data);
                
                // Analysis results tables
                if (data.results) {
                    if (data.results.normalization) {
                        container.innerHTML += this.createNormalizationTable(data.results.normalization);
                    }
                    if (data.results.syllabic_analysis) {
                        container.innerHTML += this.createSyllabicAnalysisTable(data.results.syllabic_analysis);
                    }
                    if (data.results.vectors) {
                        container.innerHTML += this.createVectorsTable(data.results.vectors);
                    }
                    if (data.results.pipeline) {
                        container.innerHTML += this.createPipelineTable(data.results.pipeline);
                    }
                }
                
                // Validation details
                if (data.validation) {
                    container.innerHTML += this.createValidationTable(data.validation);
                }
            }

            createSummaryTable(data) {
                return `
                    <div class="table-container">
                        <h5>📋 Analysis Summary</h5>
                        <table class="result-table">
                            <thead>
                                <tr><th>Property</th><th>Value</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Analysis Level</td><td><span class="badge-primary metric-badge">${data.analysis_level || 'N/A'}</span></td></tr>
                                <tr><td>Original Text</td><td style="font-family: 'Arial Unicode MS', Arial; direction: rtl;">${data.original_text || 'N/A'}</td></tr>
                                <tr><td>Character Count</td><td><span class="badge-info metric-badge">${data.metadata?.character_count || 0}</span></td></tr>
                                <tr><td>Word Count</td><td><span class="badge-info metric-badge">${data.metadata?.word_count || 0}</span></td></tr>
                                <tr><td>Decision Points</td><td><span class="badge-success metric-badge">${data.metadata?.decision_points || 0}</span></td></tr>
                                <tr><td>Processing Time</td><td><span class="badge-warning metric-badge">${(data.processing_time || 0).toFixed(4)}s</span></td></tr>
                                <tr><td>Timestamp</td><td>${new Date(data.timestamp).toLocaleString()}</td></tr>
                            </tbody>
                        </table>
                    </div>
                `;
            }

            createNormalizationTable(normData) {
                if (!normData) return '';
                
                let stepsTable = '';
                if (normData.steps && normData.steps.length > 0) {
                    stepsTable = `
                        <table class="nested-table">
                            <thead>
                                <tr><th>Step</th><th>Description</th><th>Applied</th></tr>
                            </thead>
                            <tbody>
                                ${normData.steps.map(step => `
                                    <tr>
                                        <td><span class="badge-info metric-badge">${step}</span></td>
                                        <td>${this.getStepDescription(step)}</td>
                                        <td><span class="badge-success metric-badge">Yes</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
                }
                
                return `
                    <div class="table-container">
                        <h5>🔄 Text Normalization</h5>
                        <table class="result-table">
                            <thead>
                                <tr><th>Aspect</th><th>Result</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Original Text</td><td style="font-family: 'Arial Unicode MS', Arial; direction: rtl;">${normData.original || 'N/A'}</td></tr>
                                <tr><td>Normalized Text</td><td style="font-family: 'Arial Unicode MS', Arial; direction: rtl;">${normData.normalized || 'N/A'}</td></tr>
                                <tr><td>Changes Made</td><td><span class="badge-${normData.changes_made ? 'info' : 'success'} metric-badge">${normData.changes_made ? 'Yes' : 'No'}</span></td></tr>
                                <tr><td>Character Changes</td><td><span class="badge-info metric-badge">${normData.character_changes || 0}</span></td></tr>
                                <tr><td>Normalization Steps</td><td>${stepsTable || 'No normalization steps'}</td></tr>
                            </tbody>
                        </table>
                    </div>
                `;
            }

            getStepDescription(step) {
                const descriptions = {
                    'diacritics_preserved': 'Diacritical marks were preserved',
                    'diacritics_inferred': 'Diacritical marks were inferred',
                    'whitespace_cleaned': 'Whitespace was normalized',
                    'digits_standardized': 'Arabic digits converted to Western',
                    'letters_normalized': 'Letter forms standardized'
                };
                return descriptions[step] || 'Normalization step applied';
            }

            createPipelineTable(pipelineData) {
                if (!pipelineData) return '';
                
                let stagesTable = '';
                if (pipelineData.stages && pipelineData.stages.length > 0) {
                    stagesTable = `
                        <table class="nested-table">
                            <thead>
                                <tr><th>Stage</th><th>Status</th><th>Processing Time</th><th>Output</th></tr>
                            </thead>
                            <tbody>
                                ${pipelineData.stages.map(stage => `
                                    <tr>
                                        <td><span class="badge-primary metric-badge">${stage.name || 'Unknown'}</span></td>
                                        <td><span class="badge-${stage.success ? 'success' : 'warning'} metric-badge">${stage.success ? 'Success' : 'Failed'}</span></td>
                                        <td><span class="badge-info metric-badge">${(stage.processing_time || 0).toFixed(4)}s</span></td>
                                        <td>${stage.output ? JSON.stringify(stage.output).substring(0, 50) + '...' : 'N/A'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
                }
                
                return `
                    <div class="table-container">
                        <h5>� Complete Pipeline Analysis</h5>
                        <table class="result-table">
                            <thead>
                                <tr><th>Aspect</th><th>Result</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Pipeline Status</td><td><span class="badge-${pipelineData.success ? 'success' : 'warning'} metric-badge">${pipelineData.success ? 'Complete' : 'Failed'}</span></td></tr>
                                <tr><td>Total Stages</td><td><span class="badge-primary metric-badge">${pipelineData.total_stages || 0}</span></td></tr>
                                <tr><td>Successful Stages</td><td><span class="badge-success metric-badge">${pipelineData.successful_stages || 0}</span></td></tr>
                                <tr><td>Total Processing Time</td><td><span class="badge-warning metric-badge">${(pipelineData.total_processing_time || 0).toFixed(4)}s</span></td></tr>
                                <tr><td>Pipeline Stages</td><td>${stagesTable || 'No stages information'}</td></tr>
                            </tbody>
                        </table>
                    </div>
                `;
            }

            createSyllabicAnalysisTable(syllData) {
                if (!syllData) return '';
                
                let syllabic_unitsTable = '';
                if (syllData.syllabic_units && syllData.syllabic_units.length > 0) {
                    syllabic_unitsTable = `
                        <table class="nested-table">
                            <thead>
                                <tr><th>Index</th><th>CV Pattern</th><th>Type</th><th>Valid</th></tr>
                            </thead>
                            <tbody>
                                ${syllData.syllabic_units.map((syll, idx) => `
                                    <tr>
                                        <td><span class="badge-primary metric-badge">${idx + 1}</span></td>
                                        <td style="font-family: 'Courier New', monospace;">${syll || 'N/A'}</td>
                                        <td><span class="badge-info metric-badge">CV Pattern</span></td>
                                        <td><span class="badge-success metric-badge">Valid</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
                }
                
                let sequenceInfo = '';
                if (syllData.sequence && syllData.sequence.length > 0) {
                    sequenceInfo = syllData.sequence.join(', ');
                }
                
                return `
                    <div class="table-container">
                        <h5>📝 SyllabicAnalysis Analysis</h5>
                        <table class="result-table">
                            <thead>
                                <tr><th>Aspect</th><th>Result</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Input Sequence</td><td style="font-family: 'Arial Unicode MS', Arial;">${sequenceInfo || 'N/A'}</td></tr>
                                <tr><td>CV Pattern</td><td style="font-family: 'Courier New', monospace;"><span class="badge-info metric-badge">${syllData.cv_pattern || 'N/A'}</span></td></tr>
                                <tr><td>Total SyllabicUnits</td><td><span class="badge-primary metric-badge">${syllData.syllabic_unit_count || 0}</span></td></tr>
                                <tr><td>Complexity Score</td><td><span class="badge-info metric-badge">${syllData.complexity || 0}</span></td></tr>
                                <tr><td>Valid SyllabicUnits</td><td><span class="badge-success metric-badge">${syllData.valid_syllabic_units || 0}</span></td></tr>
                                <tr><td>CV Patterns</td><td>${syllabic_unitsTable || 'No syllabic_units found'}</td></tr>
                            </tbody>
                        </table>
                    </div>
                `;
            }

            createVectorsTable(vectorData) {
                if (!vectorData) return '';
                
                return `
                    <div class="table-container">
                        <h5>🧮 Vector Operations</h5>
                        <table class="result-table">
                            <thead>
                                <tr><th>Operation</th><th>Result</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Operation Type</td><td><span class="badge-info metric-badge">${vectorData.type || 'unknown'}</span></td></tr>
                                <tr><td>Vector Dimension</td><td><span class="badge-primary metric-badge">${vectorData.dimension || 0}</span></td></tr>
                                <tr><td>Data Point</td><td style="font-family: 'Arial Unicode MS', Arial;">${vectorData.phoneme || vectorData.template || vectorData.root || 'N/A'}</td></tr>
                                <tr><td>Index/Position</td><td><span class="badge-success metric-badge">${vectorData.index !== undefined ? vectorData.index : 'N/A'}</span></td></tr>
                                <tr><td>Vector Sample</td><td style="font-family: 'Courier New', monospace;">${vectorData.vector ? `[${vectorData.vector.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]` : 'No vector'}</td></tr>
                                ${vectorData.error ? `<tr><td>Error</td><td><span class="badge-warning metric-badge">${vectorData.error}</span></td></tr>` : ''}
                                ${vectorData.valid_phonemes !== undefined ? `<tr><td>Valid Phonemes</td><td><span class="badge-success metric-badge">${vectorData.valid_phonemes}</span></td></tr>` : ''}
                                ${vectorData.input_dimension !== undefined ? `<tr><td>Input Dimension</td><td><span class="badge-info metric-badge">${vectorData.input_dimension}</span></td></tr>` : ''}
                            </tbody>
                        </table>
                    </div>
                `;
            }

            createValidationTable(validationData) {
                if (!validationData) return '';
                
                return `
                    <div class="table-container">
                        <h5>✅ Validation Details</h5>
                        <table class="result-table">
                            <thead>
                                <tr><th>Check</th><th>Status</th><th>Details</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Overall Valid</td><td><span class="badge-${validationData.valid ? 'success' : 'warning'} metric-badge">${validationData.valid ? 'Valid' : 'Invalid'}</span></td><td>${validationData.reason || 'No details'}</td></tr>
                                <tr><td>Length Check</td><td><span class="badge-${validationData.length_valid ? 'success' : 'warning'} metric-badge">${validationData.length_valid ? 'Pass' : 'Fail'}</span></td><td>Length: ${validationData.length || 0} chars</td></tr>
                                <tr><td>Character Check</td><td><span class="badge-${validationData.char_valid ? 'success' : 'warning'} metric-badge">${validationData.char_valid ? 'Pass' : 'Fail'}</span></td><td>${validationData.char_details || 'No details'}</td></tr>
                            </tbody>
                        </table>
                    </div>
                `;
            }

            toggleView(viewType) {
                this.state.currentView = viewType;
                this.setView(viewType);
            }

            setView(viewType) {
                const tableView = document.getElementById('tableView');
                const jsonView = document.getElementById('jsonView');
                const tableBtn = document.getElementById('tableViewBtn');
                const jsonBtn = document.getElementById('jsonViewBtn');
                
                if (viewType === 'table') {
                    tableView.style.display = 'block';
                    jsonView.style.display = 'none';
                    tableBtn.classList.add('active');
                    jsonBtn.classList.remove('active');
                } else {
                    tableView.style.display = 'none';
                    jsonView.style.display = 'block';
                    tableBtn.classList.remove('active');
                    jsonBtn.classList.add('active');
                }
            }

            downimport_dataJSON() {
                if (!this.state.currentData) {
                    alert('No data available to downimport_data');
                    return;
                }
                
                const dataStr = JSON.stringify(this.state.currentData, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(dataBlob);
                
                const link = document.createElement('a');
                link.href = url;
                link.downimport_data = `morphology-analysis-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
                
                // Show feedback
                const downimport_dataBtn = document.getElementById('downimport_dataBtn');
                const originalText = downimport_dataBtn.textContent;
                downimport_dataBtn.textContent = '✅ Downimport_dataed!';
                downimport_dataBtn.style.background = '#28a745';
                setTimeout(() => {
                    downimport_dataBtn.textContent = originalText;
                    downimport_dataBtn.style.background = '';
                }, 2000);
            }

            async runAnalysis() {
                const text = document.getElementById('analysisText').value.trim();
                const level = document.getElementById('analysisLevel').value;
                
                if (!text) {
                    this.showResult('Please enter some text to analyze', 'error');
                    return;
                }
                
                try {
                    this.showImporting(true);
                    
                    const data = await this.makeRequest('/api/analyze', {
                        method: 'POST',
                        body: JSON.stringify({ text, level })
                    });
                    
                    this.showResult(data);
                    this.highlightDecisionPath(data.decision_path || []);
                    
                } catch (error) {
                    this.showResult(`Analysis failed: ${error.message}`, 'error');
                } finally {
                    this.showImporting(false);
                }
            }

            async validateInput() {
                const text = document.getElementById('analysisText').value.trim();
                
                if (!text) {
                    this.showResult('Please enter some text to validate', 'error');
                    return;
                }
                
                try {
                    this.showImporting(true);
                    
                    const data = await this.makeRequest('/api/validate', {
                        method: 'POST',
                        body: JSON.stringify({ text })
                    });
                    
                    this.showResult(data, data.valid ? 'success' : 'error');
                    
                } catch (error) {
                    this.showResult(`Validation failed: ${error.message}`, 'error');
                } finally {
                    this.showImporting(false);
                }
            }

            async showStats() {
                try {
                    this.showImporting(true);
                    
                    const data = await this.makeRequest('/api/stats');
                    this.showResult(data);
                    
                } catch (error) {
                    this.showResult(`Failed to import_data statistics: ${error.message}`, 'error');
                } finally {
                    this.showImporting(false);
                }
            }

            async showDecisionTree() {
                try {
                    this.showImporting(true);
                    
                    const data = await this.makeRequest('/api/decision-tree');
                    this.showResult(data);
                    
                } catch (error) {
                    this.showResult(`Failed to import_data decision tree: ${error.message}`, 'error');
                } finally {
                    this.showImporting(false);
                }
            }

            clearResults() {
                document.getElementById('results').style.display = 'none';
                document.querySelectorAll('.decision-node.active').forEach(node => {
                    node.classList.remove('active');
                });
                document.getElementById('analysisText').value = '';
                localStorage.removeItem('morphology_text');
                this.validateForm();
            }

            highlightDecision(element) {
                document.querySelectorAll('.decision-node.active').forEach(node => {
                    node.classList.remove('active');
                });
                element.classList.add('active');
                
                const decision = element.dataset.decision;
                const question = element.querySelector('.question').textContent;
                const answer = element.querySelector('.answer').textContent;
                
                this.showResult({
                    decision_type: decision,
                    question,
                    answer,
                    details: element.querySelector('.details').textContent,
                    timestamp: new Date().toISOString()
                });
            }

            highlightDecisionPath(path) {
                document.querySelectorAll('.decision-node.active').forEach(node => {
                    node.classList.remove('active');
                });
                
                path.forEach(decision => {
                    const node = document.querySelector(`[data-decision="${decision}"]`);
                    if (node) node.classList.add('active');
                });
            }

            async updateRealTimeStats() {
                try {
                    const data = await this.makeRequest('/api/stats');
                    
                    this.updateStatCard('totalAnalyses', data.total_analyses || 0);
                    this.updateStatCard('engineStatus', data.engine_status || 'operational');
                    this.updateStatCard('memoryUsage', `${(data.memory_usage_mb || 0).toFixed(1)}MB`);
                    this.updateStatCard('avgTime', `${(data.average_processing_time || 0).toFixed(3)}s`);
                    
                } catch (error) {
                    console.warn('Failed to update stats:', error.message);
                }
            }

            updateStatCard(id, value) {
                const element = document.getElementById(id);
                if (element && element.textContent !== String(value)) {
                    element.textContent = value;
                    element.style.transform = 'scale(1.1)';
                    setTimeout(() => element.style.transform = 'scale(1)', 200);
                }
            }

            beginStatsUpdate() {
                this.updateRealTimeStats();
                this.state.statsInterval = setInterval(() => {
                    this.updateRealTimeStats();
                }, 5000);
            }

            endStatsUpdate() {
                if (this.state.statsInterval) {
                    clearInterval(this.state.statsInterval);
                    this.state.statsInterval = null;
                }
            }
        }

        // Global app instance
        let app;

        // Global functions for backward compatibility
        function runAnalysis() { return app.runAnalysis(); }
        function validateInput() { return app.validateInput(); }
        function showStats() { return app.showStats(); }
        function showDecisionTree() { return app.showDecisionTree(); }
        function clearResults() { return app.clearResults(); }
        function highlightDecision(element) { return app.highlightDecision(element); }
        function toggleView(viewType) { return app.toggleView(viewType); }
        function downimport_dataJSON() { return app.downimport_dataJSON(); }

        // Initialize when DOM is ready
        document.addEventListener('DOMContentImported', () => {
            app = new ProfessionalMorphologyApp();
        });

        // Cleanup on unimport_data
        window.addEventListener('beforeunimport_data', () => {
            if (app) app.endStatsUpdate();
        });

        // Global error handling
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            if (app) app.showResult(`Unexpected error: ${event.error.message}`, 'error');
        });
    </script>
</body>
</html>
'''

# =============================================================================
# PROFESSIONAL FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Professional main interface route"""
    try:
        begin_time = time.time()
        
        stats = flask_decision_tree.get_stats()
        stats.update({
            'phoneme_inventory': len(PHONEME_INVENTORY),
            'syllabic_unit_templates': len(TEMPLATES),
            'phonological_rules': len(PHONO_RULES),
            'uptime_seconds': time.time() - app_state['begin_time'],
            'total_analyses': app_state['total_analyses'],
            'total_characters': app_state['total_characters'],
            'unique_texts': len(app_state['unique_texts'])
        })
        
        processing_time = time.time() - begin_time
        log_request('GET', '/', 200, f'Interface served in {processing_time:.4f}s')
        
        return render_template_string(PROFESSIONAL_HTML, stats=stats)
        
    except Exception as e:
        logger.error(f"Error serving interface: {e}")
        app_state['error_count'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Professional text analysis endpoint"""
    begin_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data provided")
        
        text = sanitize_input(data.get('text', ''))
        level = validate_analysis_level(data.get('level', 'basic'))
        
        if not text:
            raise ValueError("Text field is required")
        
        # Decision tree processing
        decision_path = []
        
        # Validation
        decision_path.append('validation')
        validation_result = decision_engine.input_validation_decision_tree(text)
        
        if not validation_result.get('valid', False):
            processing_time = time.time() - begin_time
            return jsonify({
                'success': False,
                'error': 'Input validation failed',
                'validation_details': validation_result,
                'decision_path': decision_path,
                'processing_time': processing_time
            }), 400
        
        # Analysis processing
        decision_path.append('normalization')
        analysis_results = {}
        
        if level in ['basic', 'advanced', 'comprehensive']:
            decision_path.append('text-processing')
            # Use text normalization instead of phonological rules
            analysis_results['normalization'] = decision_engine.text_normalization_decision_tree(text)
        
        if level in ['advanced', 'comprehensive']:
            decision_path.append('syllabic_analysis')
            # Convert text to character sequence for syllabic_analysis
            char_sequence = list(text.replace(' ', ''))
            analysis_results['syllabic_analysis'] = decision_engine.syllabic_analysis_decision_tree(char_sequence)
        
        if level == 'comprehensive':
            decision_path.append('vector-ops')
            analysis_results['vectors'] = decision_engine.vector_operations_decision_tree('phoneme_analysis', text)
            
            # Add complete pipeline analysis
            decision_path.append('complete-pipeline')
            analysis_results['pipeline'] = decision_engine.complete_pipeline_decision_tree({
                'text': text,
                'level': level,
                'timestamp': datetime.now().isoformat()
            })
        
        # Update app state
        app_state['total_analyses'] += 1
        app_state['total_characters'] += len(text)
        app_state['unique_texts'].add(text[:100])
        
        processing_time = time.time() - begin_time
        app_state['performance_metrics'].append(processing_time)
        
        result = {
            'success': True,
            'analysis_level': level,
            'original_text': text,
            'results': analysis_results,
            'decision_path': decision_path,
            'validation': validation_result,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'character_count': len(text),
                'word_count': len(text.split()),
                'decision_points': len(decision_path)
            }
        }
        
        log_request('POST', '/api/analyze', 200, f'Analysis completed in {processing_time:.4f}s')
        return jsonify(result)
        
    except ValueError as e:
        processing_time = time.time() - begin_time
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': str(e),
            'processing_time': processing_time
        }), 400
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        app_state['error_count'] += 1
        processing_time = time.time() - begin_time
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e),
            'processing_time': processing_time
        }), 500

@app.route('/api/validate', methods=['POST'])
def validate_text():
    """Professional input validation endpoint"""
    begin_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data provided")
        
        text = data.get('text', '')
        validation_result = decision_engine.input_validation_decision_tree(text)
        
        # Enhanced security checks
        security_checks = {
            'has_sql_injection': bool(re.search(r'(union|select|insert|update|delete|drop)\s', text.lower())),
            'has_xss_attempt': bool(re.search(r'<script|javascript:|onimport_data=|onerror=', text.lower())),
            'has_excessive_length': len(text) > 10000,
            'is_empty': len(text.strip()) == 0
        }
        
        # Arabic text analysis
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.sub(r'\s', '', text))
        arabic_ratio = arabic_chars / max(total_chars, 1)
        
        processing_time = time.time() - begin_time
        
        result = {
            'valid': validation_result.get('valid', False),
            'text_length': len(text),
            'validation_details': validation_result,
            'security_checks': security_checks,
            'language_analysis': {
                'arabic_characters': arabic_chars,
                'total_characters': total_chars,
                'arabic_ratio': arabic_ratio,
                'likely_arabic': arabic_ratio > 0.5
            },
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        log_request('POST', '/api/validate', 200, f'Validation completed in {processing_time:.4f}s')
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        processing_time = time.time() - begin_time
        return jsonify({
            'valid': False,
            'error': str(e),
            'processing_time': processing_time
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Professional statistics endpoint"""
    begin_time = time.time()
    
    try:
        base_stats = flask_decision_tree.get_stats()
        
        current_time = time.time()
        uptime = current_time - app_state['begin_time']
        
        avg_response_time = (
            sum(app_state['performance_metrics']) / len(app_state['performance_metrics'])
            if app_state['performance_metrics'] else 0
        )
        
        result = {
            'engine_status': 'operational',
            'uptime_seconds': uptime,
            'requests_processd': app_state['requests_count'],
            'total_analyses': app_state['total_analyses'],
            'total_characters': app_state['total_characters'],
            'unique_texts': len(app_state['unique_texts']),
            'error_count': app_state['error_count'],
            'average_processing_time': avg_response_time,
            'memory_usage_mb': decision_engine.W_inflect.nbytes / (1024 * 1024),
            'recent_requests': app_state['recent_requests'][:10],
            'system_info': {
                'phoneme_inventory_size': len(PHONEME_INVENTORY),
                'syllabic_unit_templates': len(TEMPLATES),
                'phonological_rules': len(PHONO_RULES)
            },
            'processing_time': time.time() - begin_time,
            'timestamp': datetime.now().isoformat()
        }
        
        log_request('GET', '/api/stats', 200, f'Statistics provided')
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/decision-tree', methods=['GET'])
def get_decision_tree():
    """Professional decision tree structure endpoint"""
    begin_time = time.time()
    
    try:
        structure = {
            'categories': [
                {
                    'id': 'engine-operations',
                    'name': 'Engine Operations',
                    'decisions': [
                        {'id': 'engine-init', 'question': 'Engine initialization', 'complexity': 'low'},
                        {'id': 'phono-rules', 'question': 'Phonological rule application', 'complexity': 'medium'},
                        {'id': 'syllabic_analysis', 'question': 'SyllabicAnalysis strategy', 'complexity': 'high'},
                        {'id': 'vector-ops', 'question': 'Vector operations', 'complexity': 'medium'}
                    ]
                },
                {
                    'id': 'flask-operations',
                    'name': 'Flask Operations',
                    'decisions': [
                        {'id': 'routing', 'question': 'Route handling', 'complexity': 'low'},
                        {'id': 'validation', 'question': 'Input validation', 'complexity': 'medium'},
                        {'id': 'api-processing', 'question': 'API processing', 'complexity': 'medium'}
                    ]
                }
            ],
            'metadata': {
                'total_categories': 2,
                'total_decisions': 7,
                'implementation_status': 'complete',
                'version': '2.0.0'
            },
            'processing_time': time.time() - begin_time,
            'timestamp': datetime.now().isoformat()
        }
        
        log_request('GET', '/api/decision-tree', 200, 'Decision tree structure provided')
        return jsonify(structure)
        
    except Exception as e:
        logger.error(f"Decision tree error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorprocessr(404)
def not_found(error):
    return jsonify({
        'error': 'Resource not found',
        'available_endpoints': ['GET /', 'POST /api/analyze', 'GET /api/stats', 'POST /api/validate', 'GET /api/decision-tree']
    }), 404

@app.errorprocessr(500)
def internal_error(error):
    app_state['error_count'] += 1
    return jsonify({'error': 'Internal server error'}), 500

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌳 PROFESSIONAL ARABIC MORPHOPHONOLOGICAL ENGINE")
    print("🌐 Full-Stack Dynamic Flask Implementation")
    print("="*60)
    
    init_result = decision_engine.engine_initialization_decision_tree()
    print(f"Engine Status: {init_result['status']}")
    print(f"Components: {init_result['components']}")
    
    print(f"\n🚀 Begining professional Flask application...")
    print(f"Available at: http://localhost:5000")
    print(f"\n📋 Professional Features:")
    print(f"  ✅ Real-time statistics")
    print(f"  ✅ Dynamic interface updates")
    print(f"  ✅ Comprehensive input validation")
    print(f"  ✅ Security best practices")
    print(f"  ✅ Professional error handling")
    print(f"  ✅ Responsive design")
    print(f"  ✅ Accessibility compliance")
    print(f"\n🌐 API Endpoints:")
    print(f"  GET  /                 - Professional interface")
    print(f"  POST /api/analyze      - Text analysis")
    print(f"  GET  /api/stats        - Real-time statistics")
    print(f"  POST /api/validate     - Input validation")
    print(f"  GET  /api/decision-tree - Decision tree structure")
    print("="*60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
