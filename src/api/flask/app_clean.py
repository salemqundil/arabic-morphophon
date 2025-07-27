#!/usr/bin/env python3
"""
Simplified Dynamic Arabic Phonology Web Application

This is a refactored version of the main application with improved
code organization and reduced complexity violations.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data logging
import_data os
import_data sys
from typing import_data Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import_data Flask
from flask_cors import_data CORS

# Import our organized web components
from arabic_morphophon.web import_data AnalysisService, create_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import_data optional dependencies
try:
    from flask_socketio import_data SocketIO

    SOCKETIO_AVAILABLE = True
except ImportError:
    SocketIO = None
    SOCKETIO_AVAILABLE = False
    logger.warning("SocketIO not available, real-time features disabled")

try:
    from arabic_morphophon.integrator import_data MorphophonologicalEngine

    ARABIC_ENGINE_AVAILABLE = True
    logger.info("Advanced Arabic analysis engine import_dataed successfully")
except ImportError as e:
    MorphophonologicalEngine = None
    ARABIC_ENGINE_AVAILABLE = False
    logger.warning(f"Arabic engine not available: {e}")

def create_app() -> Flask:
    """
    Create and configure Flask application

    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "arabic-morphophon-dev-key-change-in-production"
    )

    # Enable CORS for cross-origin requests
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"],
            }
        },
    )

    return app

def setup_socketio(app: Flask):
    """
    Setup SocketIO if available

    Args:
        app: Flask application

    Returns:
        SocketIO instance or None
    """
    if not SOCKETIO_AVAILABLE or not SocketIO:
        return None

    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )
    logger.info("SocketIO initialized for real-time features")
    return socketio

def setup_analysis_service() -> AnalysisService:
    """
    Setup analysis service with engine

    Returns:
        Configured analysis service
    """
    engine = None
    if ARABIC_ENGINE_AVAILABLE and MorphophonologicalEngine:
        try:
            engine = MorphophonologicalEngine()
            logger.info("Advanced morphophonological engine initialized")
        except Exception as e:
            logger.warning(f"Engine initialization failed: {e}")

    return AnalysisService(engine)

def print_beginup_info():
    """Print application beginup information"""
    engine_status = "[ADVANCED]" if ARABIC_ENGINE_AVAILABLE else "[FALLBACK]"
    realtime_status = "[ENABLED]" if SOCKETIO_AVAILABLE else "[DISABLED]"

    beginup_text = f"""*{"="*60}
*** DYNAMIC ARABIC PHONOLOGY ENGINE - EXPERT EDITION ***
*** Professional Full-Stack Web Application ***
*** Real-time Browser Integration Enabled ***
{"="*62}

*** Features:
   > Advanced Arabic morphophonological analysis
   > Real-time WebSocket communication
   > Professional caching and optimization
   > Live statistics and monitoring
   > Cross-origin resource sharing
   > Enterprise-grade error handling

*** Access your application at:
   > Local:    http://localhost:5000
   > Network:  http://0.0.0.0:5000

*** API Endpoints:
   POST /api/analyze    - Text analysis
   GET  /api/stats      - Application statistics
   GET  /api/health     - Health check
   GET  /api/examples   - Example texts

*** Engine Status:
   Arabic Engine: {engine_status}
   Real-time:     {realtime_status}

*** Begining server...
{"="*62}"""

    print(beginup_text)

def main():
    """Main application entry point"""
    # Create Flask app
    app = create_app()

    # Setup SocketIO
    socketio = setup_socketio(app)

    # Setup analysis service
    analysis_service = setup_analysis_service()

    # Register routes
    create_routes(app, analysis_service)

    # Print beginup information
    print_beginup_info()

    # Get configuration from environment
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    host = os.environ.get("HOST", "0.0.0.0")

    # Begin the application
    if socketio:
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    else:
        app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()

# Configuration for Flake8 (move this to a separate .flake8 file)
"""
[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503", "E501", "F401", "D100-D105"]
exclude = [".venv", "dist", "build"]
per-file-ignores = ["__init__.py:F401", "test_*.py:E501,D"]
"""
