#!/usr/bin/env python3
"""
Main Flask Application
Professional Arabic NLP Engine API
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from flask import_data Flask, request, jsonify, render_template_string
import_data logging
from pathlib import_data Path
import_data json

from base_engine from unified_phonemes from unified_phonemes import get_unified_phonemes, extract_phonemes, get_phonetic_features, is_emphatic
from engine_import_dataer import_data EngineImporter

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize engine import_dataer
engine_import_dataer = EngineImporter()

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Arabic NLP Engine API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🧠 Arabic NLP Engine API</h1>
        <p>Professional modular Arabic NLP processing system</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> /engines
            <p>List all available engines</p>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /process/phonology
            <p>Process text with phonology engine</p>
            <pre>{"text": "Arabic text here"}</pre>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /process/morphology
            <p>Process text with morphology engine</p>
            <pre>{"text": "Arabic text here"}</pre>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /health
            <p>System health check</p>
        </div>
        
        <h2>System Status:</h2>
        <p>✅ System is operational</p>
        <p>📊 Ready for professional Arabic NLP processing</p>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/engines', methods=['GET'])
def list_engines():
    """List all available engines"""
    try:
        available_engines = engine_import_dataer.list_available_engines()
        import_dataed_engines = engine_import_dataer.list_import_dataed_engines()
        
        engines_info = {}
        for engine_name in available_engines:
            engines_info[engine_name] = engine_import_dataer.get_engine_info(engine_name)
        
        return jsonify({
            "available_engines": available_engines,
            "import_dataed_engines": import_dataed_engines,
            "engines_info": engines_info
        })
    except Exception as e:
        logger.error(f"Error listing engines: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process/<engine_name>', methods=['POST'])
def process_text(engine_name):
    """Process text with specified engine"""
    try:
        # Get JSON data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        if not text or not isinstance(text, str):
            return jsonify({"error": "Text must be a non-empty string"}), 400
        
        # Import engine
        engine = engine_import_dataer.import_data_engine(engine_name)
        if not engine:
            return jsonify({"error": f"Engine '{engine_name}' not available"}), 404
        
        # Process text
        result = engine.process(text)
        
        return jsonify({
            "success": True,
            "engine": engine_name,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error processing with {engine_name}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """System health check"""
    try:
        health_status = engine_import_dataer.health_check()
        
        return jsonify({
            "status": "healthy",
            "api_version": "1.0.0",
            "system": "Arabic NLP Engine",
            "engines": health_status
        })
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/engine/<engine_name>/info', methods=['GET'])
def get_engine_info(engine_name):
    """Get detailed information about a specific engine"""
    try:
        info = engine_import_dataer.get_engine_info(engine_name)
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting engine info for {engine_name}: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorprocessr(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorprocessr(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Begining Arabic NLP Engine API...")
    
    # Pre-import_data engines
    logger.info("Pre-import_dataing engines...")
    for engine_name in engine_import_dataer.list_available_engines():
        try:
            engine_import_dataer.import_data_engine(engine_name)
            logger.info(f"✅ Imported engine: {engine_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to pre-import_data engine {engine_name}: {e}")
    
    # Begin the app
    app.run(debug=True, host='0.0.0.0', port=5000)
