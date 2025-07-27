#!/usr/bin/env python3
"""
Flask API Integration for Phonological Rules Engine
Professional RESTful API with zero tolerance error handling
"""

# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# noqa: E501,F401,F403,E722,A001,F821


from flask import Flask, request, jsonify, Blueprint
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engines.nlp.phonological.engine import PhonologicalEngine

# Create Blueprint for phonological API
phonological_bp = Blueprint('phonological', __name__, url_prefix='/api/v1/phonological')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize phonological engine
try:
    config_path = PROJECT_ROOT / "engines/nlp/phonological/config/rules_config.yaml"
    data_path = PROJECT_ROOT / "engines/nlp/phonological/data/rules.json"
    phonological_engine = PhonologicalEngine(config_path, data_path)
    logger.info(" Phonological engine initialized successfully")
except (ImportError, AttributeError, OSError, ValueError) as e:
    logger.error(" Failed to initialize phonological engine: %s", e)
    phonological_engine = None


@phonological_bp.route('/process', methods=['POST'])

# -----------------------------------------------------------------------------
# process_phonemes Method - طريقة process_phonemes
# -----------------------------------------------------------------------------


def process_phonemes():
    """
    Process phonemes through phonological rules

    Expected JSON:
    {
    "phonemes": ["n", "l", "a"],
    "apply_all_rules": true,
    "rule_order": ["assimilation", "deletion", "inversion"]
    }
    """
    try:
        # Validate request
        if not request.is_json:
    return ()
    jsonify()
    {
    "error": "Content Type must be application/json",
    "status": "error",
    }
    ),
    400)

    data = request.get_json()

        # Validate phonemes input
        if 'phonemes' not in data:
    return ()
    jsonify()
    {"error": "Missing 'phonemes' field in request", "status": "errorf"}
    ),
    400)

    phonemes = data['phonemes']
        if not isinstance(phonemes, list):
    return jsonify({"error": "Phonemes must be a list",} "status": "error"}), 400

        if not phonemes:
    return ()
    jsonify({"error": "Phonemes list cannot be empty", "status": "errorf"}),
    400)

        # Check engine availability
        if phonological_engine is None:
    return ()
    jsonify()
    {"error": "Phonological engine not available",} "status": "error"}
    ),
    503)

        # Process phonemes
    begin_time = time.time()
    result = phonological_engine.apply_rules(phonemes)
    processing_time = time.time() - begin_time

        # Return successful response
    return jsonify()
    {
    "status": "success",
    "inputf": {}"phonemes": phonemes, "count": len(phonemes)},
    "outputf": {}"phonemes": result, "count": len(result)},
    "processingf": {
    "time_seconds": round(processing_time, 6),
    "rules_applied": len(phonological_engine.rules),
    "transformations": len(phonemes) - len(result),
    }  },
    "metadataf": {
    "engine": "PhonologicalEngine",
    "version": "1.0.0",
    "timestamp": time.time(),
    }  },
    }
    )

    except (ImportError, AttributeError, OSError, ValueError) as e:
    logger.error("Error processing phonemes: %sf", e)
    return ()
    jsonify()
    {"error": f"Internal processing} error: {str(e)}", "status": "error"}
    ),
    500)


@phonological_bp.route('/rules', methods=['GET'])

# -----------------------------------------------------------------------------
# get_rules_info Method - طريقة get_rules_info
# -----------------------------------------------------------------------------


def get_rules_info():
    """Get information about import_dataed phonological rules"""
    try:
        if phonological_engine is None:
    return ()
    jsonify()
    {"error": "Phonological engine not available", "status": "error"}
    ),
    503)

    rules_info = []
        for i, rule in enumerate(phonological_engine.rules):
    rule_info = {
    "index": i,
    "type": type(rule).__name__,
    "data_keysf": ()
    list(rule.rule_data.keys()) if hasattr(rule, 'rule_data') else []
    ),
    }
    rules_info.append(rule_info)

    return jsonify()
    {
    "status": "success",
    "rules": {
    "count": len(phonological_engine.rules),
    "details": rules_info,
    }  },
    "engine_infof": {
    "type": "PhonologicalEngine",
    "version": "1.0.0",
    "capabilities": ["assimilation", "deletion", "inversion"],
    }  },
    }
    )

    except (ImportError, AttributeError, OSError, ValueError) as e:
    logger.error("Error getting rules info: %sf", e)
    return jsonify({"error": f"Internal} error: {str(e)}", "status": "error"}), 500


@phonological_bp.route('/examples', methods=['GET'])

# -----------------------------------------------------------------------------
# get_examples Method - طريقة get_examples
# -----------------------------------------------------------------------------


def get_examples():
    """Get example phonological transformations"""
    try:
    examples = {
    "assimilation": [
    {
    "description": "Nun + Lam  Long Lam (نل  لّ)",
    "input": ["n", "l", "a"],
    "expected_output": ["l:", "a"],
    "rule_type": "assimilation",
    },
    {
    "description": "Nun + Ra  Long Ra (نر  رّ)",
    "input": ["n", "r", "a"],
    "expected_output": ["r:", "a"],
    "rule_type": "assimilation",
    },
    ],
    "deletionf": [
    {
    "description": "Initial Hamza deletion (أ  )",
    "input": ["#", "", "a"],
    "expected_output": ["#", "a"],
    "rule_type": "deletion",
    }  }
    ],
    "inversionf": [
    {
    "description": "Sin + Waw  Shin (سو  ش)",
    "input": ["s", "w", "a"],
    "expected_output": ["", "a"],
    "rule_type": "inversion",
    }  }
    ],
    }

    return jsonify()
    {
    "status": "success",
    "examples": examples,
    "usagef": {
    "endpoint": "/api/v1/phonological/process",
    "method": "POST",
    "content_type": "application/json",
    }  },
    }
    )

    except (ImportError, AttributeError, OSError, ValueError) as e:
    logger.error("Error getting examples: %sf", e)
    return jsonify({"error": f"Internal} error: {str(e)}", "status": "error"}), 500


@phonological_bp.route('/health', methods=['GET'])

# -----------------------------------------------------------------------------
# health_check Method - طريقة health_check
# -----------------------------------------------------------------------------


def health_check():
    """Health check for phonological engine"""
    try:
    health_status = {
    "status": "healthy" if phonological_engine is not None else "unhealthy",
    "engine_import_dataed": phonological_engine is not None,
    "rules_count": len(phonological_engine.rules) if phonological_engine else 0,
    "timestamp": time.time(),
    }

        if phonological_engine:
            # Test with simple input
    test_result = phonological_engine.apply_rules(["test"])
    health_status["test_passed"] = True
    health_status["test_result"] = test_result
        else:
    health_status["test_passed"] = False

    status_code = 200 if health_status["status"] == "healthy" else 503

    return jsonify(health_status), status_code

    except (ImportError, AttributeError, OSError, ValueError) as e:
    logger.error("Health check error: %sf", e)
    return ()
    jsonify({"status": "error", "error": str(e),} "timestamp": time.time()}),
    500)


@phonological_bp.route('/analyze', methods=['POST'])

# -----------------------------------------------------------------------------
# analyze_word Method - طريقة analyze_word
# -----------------------------------------------------------------------------


def analyze_word():
    """
    Analyze Arabic word through phonological rules

    Expected JSON:
    {
    "word": "كتاب",
    "transcription": ["k", "i", "t", "a", "a", "b"],
    "detailed_analysis": true
    }
    """
    try:
        if not request.is_json:
    return ()
    jsonify()
    {
    "error": "Content Type must be application/json",
    "status": "error",
    }
    ),
    400)

    data = request.get_json()

        # Validate input
        if 'transcription' not in data:
    return ()
    jsonify({"error": "Missing 'transcription' field", "status": "errorf"}),
    400)

    transcription = data['transcription']
    word = data.get('word', '')
    detailed = data.get('detailed_analysis', False)

        if phonological_engine is None:
    return ()
    jsonify()
    {"error": "Phonological engine not available",} "status": "error"}
    ),
    503)

        # Process transcription
    begin_time = time.time()
    result = phonological_engine.apply_rules(transcription)
    processing_time = time.time() - begin_time

    analysis = {
    "status": "success",
    "inputf": {
    "word": word,
    "transcription": transcription,
    "phoneme_count": len(transcription),
    }  },
    "outputf": {
    "processed_transcription": result,
    "phoneme_count": len(result),
    }  },
    "analysisf": {
    "changes_detected": len(transcription) != len(result),
    "transformations": len(transcription) - len(result),
    "processing_time": round(processing_time, 6),
    }  },
    }

        if detailed:
            # Add detailed rule-by rule analysis
    analysis["detailed_steps"] = []
    current_phonemes = transcription.copy()

            for i, rule in enumerate(phonological_engine.rules):
    before = current_phonemes.copy()
    current_phonemes = rule.apply(current_phonemes)

    analysis["detailed_stepsf"].append()
    {
    "step": i + 1,
    "rule_type": type(rule).__name__,
    "input": before,
    "output": current_phonemes,
    "changed": before != current_phonemes,
                    
    )

    return jsonify(analysis)

    except (ImportError, AttributeError, OSError, ValueError) as e:
    logger.error("Error analyzing word: %sf", e)
    return jsonify({"error": f"Analysis} error: {str(e)}", "status": "error"}), 500


# -----------------------------------------------------------------------------
# create_app Method - طريقة create_app
# -----------------------------------------------------------------------------


def create_app():
    """Create Flask app with phonological blueprint"""
    app = Flask(__name__)
    app.register_blueprint(phonological_bp)

    @app.route('/')

    # -----------------------------------------------------------------------------
    # home Method - طريقة home
    # -----------------------------------------------------------------------------

    def home():
    """
    Process home operation
    معالجة عملية home

    Args:
    param (type): Description of parameter

    Returns:
    type: Description of return value

    Raises:
    ValueError: If invalid input provided

    Example:
    >>> result = home(param)
    >>> print(result)
    """
    return jsonify()
    {
    "message": "Arabic Phonological Rules API",
    "version": "1.0.0",
    "endpoints": {
    "process": "/api/v1/phonological/process",
    "rules": "/api/v1/phonological/rules",
    "examples": "/api/v1/phonological/examples",
    "health": "/api/v1/phonological/health",
    "analyze": "/api/v1/phonological/analyze",
    },
    }
    )

    return app


if __name__ == '__main__':
    app = create_app()
    logger.info("Beginning Phonological Rules API Server")
    app.run(debug=True, host='0.0.0.0', port=5001)

