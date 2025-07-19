"""
Production-Ready Web API for Phonological Engine
Professional implementation with thread-safe error handling
"""

from dataclasses import dataclass
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    render_template_string,
    g,
)
from flask_cors import CORS
from threading import Lock
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
import os
from abc import ABC, abstractmethod
import json
import uuid
from functools import wraps

# Optional Redis import for distributed deployments
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add engine to path
ENGINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(ENGINE_ROOT))

# Define static directory at module level for reuse
STATIC_DIR = ENGINE_ROOT / "static"

try:
    from phonology.classifier import Phoneme
    from phonology.syllabifier import syllabify
    from phonology.phoneme_registry import phoneme_registry

    logger.info("✅ Engine modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import engine modules: {e}")
    logger.error(
        "Ensure the 'phonology' module is installed or accessible in the Python path."
    )
    sys.exit(1)

# Flask app setup
app = Flask(__name__)
CORS(app)
app.config["JSON_SORT_KEYS"] = False

# Configuration for different deployment modes
DEPLOYMENT_MODE = os.getenv(
    "DEPLOYMENT_MODE", "development"
)  # development, production, distributed
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
INSTANCE_ID = str(uuid.uuid4())[:8]


class StatsBackend(ABC):
    """Abstract base class for statistics backends"""

    @abstractmethod
    def increment_errors(self) -> None:
        pass

    @abstractmethod
    def increment_requests(self) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class MemoryStatsBackend(StatsBackend):
    """Thread-safe in-memory statistics for single-process deployments"""

    def __init__(self):
        self._lock = Lock()
        self._error_count = 0
        self._request_count = 0
        self._start_time = datetime.now()
        self._instance_id = INSTANCE_ID

    def increment_errors(self):
        with self._lock:
            self._error_count += 1

    def increment_requests(self):
        with self._lock:
            self._request_count += 1

    def get_stats(self):
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()
            return {
                "backend": "memory",
                "instance_id": self._instance_id,
                "error_count": self._error_count,
                "request_count": self._request_count,
                "uptime_seconds": round(uptime, 2),
                "error_rate": round(
                    self._error_count / max(self._request_count, 1) * 100, 2
                ),
                "deployment_mode": DEPLOYMENT_MODE,
            }


class RedisStatsBackend(StatsBackend):
    """Redis-based statistics for multi-process/distributed deployments"""

    def __init__(self, redis_url: str = REDIS_URL):
        if not REDIS_AVAILABLE or redis is None:
            raise ImportError(
                "Redis package not available. Install with: pip install redis"
            )

        try:
            self._redis = redis.from_url(redis_url, decode_responses=True)  # type: ignore
            self._redis.ping()  # Test connection
            self._instance_id = INSTANCE_ID
            self._start_time = datetime.now()

            # Initialize instance-specific keys
            self._register_instance()
            logger.info(
                f"✅ Redis stats backend initialized (instance: {self._instance_id})"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Redis connection failed: {e}. Falling back to memory backend."
            )
            raise

    def _register_instance(self):
        """Register this instance in Redis"""
        instance_key = f"phonology:instances:{self._instance_id}"
        self._redis.hset(
            instance_key,
            mapping={
                "start_time": self._start_time.isoformat(),
                "last_seen": datetime.now().isoformat(),
                "deployment_mode": DEPLOYMENT_MODE,
            },
        )
        self._redis.expire(instance_key, 3600)  # Expire after 1 hour of inactivity

        # Add to active instances set
        self._redis.sadd("phonology:active_instances", self._instance_id)

    def _update_heartbeat(self):
        """Update instance heartbeat"""
        instance_key = f"phonology:instances:{self._instance_id}"
        self._redis.hset(instance_key, "last_seen", datetime.now().isoformat())
        self._redis.expire(instance_key, 3600)

    def increment_errors(self):
        try:
            self._redis.hincrby("phonology:global_stats", "error_count", 1)
            self._redis.hincrby(
                f"phonology:instance_stats:{self._instance_id}", "error_count", 1
            )
            self._update_heartbeat()
        except Exception as e:
            logger.error(f"Redis error increment failed: {e}")

    def increment_requests(self):
        try:
            self._redis.hincrby("phonology:global_stats", "request_count", 1)
            self._redis.hincrby(
                f"phonology:instance_stats:{self._instance_id}", "request_count", 1
            )
            self._update_heartbeat()
        except Exception as e:
            logger.error(f"Redis request increment failed: {e}")

    def get_stats(self):
        try:
            # Get global stats
            global_stats = self._redis.hgetall("phonology:global_stats")
            instance_stats = self._redis.hgetall(
                f"phonology:instance_stats:{self._instance_id}"
            )

            # Get active instances
            active_instances = list(self._redis.smembers("phonology:active_instances"))

            uptime = (datetime.now() - self._start_time).total_seconds()

            global_errors = int(global_stats.get("error_count", 0))
            global_requests = int(global_stats.get("request_count", 0))

            return {
                "backend": "redis",
                "instance_id": self._instance_id,
                "deployment_mode": DEPLOYMENT_MODE,
                "global_stats": {
                    "error_count": global_errors,
                    "request_count": global_requests,
                    "error_rate": round(
                        global_errors / max(global_requests, 1) * 100, 2
                    ),
                    "active_instances": len(active_instances),
                    "instance_list": active_instances,
                },
                "instance_stats": {
                    "error_count": int(instance_stats.get("error_count", 0)),
                    "request_count": int(instance_stats.get("request_count", 0)),
                    "uptime_seconds": round(uptime, 2),
                },
            }
        except Exception as e:
            logger.error(f"Redis stats retrieval failed: {e}")
            return {
                "backend": "redis_error",
                "error": str(e),
                "instance_id": self._instance_id,
            }


def create_stats_backend() -> StatsBackend:
    """Factory function to create appropriate stats backend based on deployment mode"""
    if DEPLOYMENT_MODE in ["production", "distributed"] and REDIS_AVAILABLE:
        try:
            try:
                return RedisStatsBackend()
            except Exception as redis_error:
                logger.error(f"RedisStatsBackend initialization failed: {redis_error}")
                logger.warning("Falling back to MemoryStatsBackend.")
                return MemoryStatsBackend()
        except ImportError as import_error:
            logger.error(f"Redis module not available: {import_error}")
            logger.warning("Falling back to MemoryStatsBackend.")
            return MemoryStatsBackend()
    else:
        if DEPLOYMENT_MODE in ["production", "distributed"]:
            logger.warning(
                "Redis not available for distributed deployment, using memory backend"
            )
        else:
            logger.info("Using memory stats backend for development mode")
        return MemoryStatsBackend()


# Initialize stats backend based on deployment configuration
api_stats = create_stats_backend()


@dataclass
class ValidationResult:
    """Validation result model"""

    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


def validate_syllabify_request(data: Dict[str, Any]) -> ValidationResult:
    """Enhanced validation with sophisticated user guidance for input limitations"""
    # Check basic data structure
    if not data:
        return ValidationResult(
            False,
            "Request body is empty. Please provide a JSON object with 'word' field.",
        )

    if "word" not in data:
        return ValidationResult(
            False, 'Missing required field: \'word\'. Example: {"word": "pataka"}'
        )

    # Get and validate word type
    word = data["word"]
    if not isinstance(word, str):
        return ValidationResult(
            False, 'Word must be a string. Example: {"word": "hello"}'
        )

    # Clean and validate word content
    word = word.strip()
    if not word:
        return ValidationResult(
            False, "Word cannot be empty. Try examples: pataka, mama, hello, kitab"
        )

    if len(word) > 100:
        return ValidationResult(
            False, "Word too long (max 100 characters). Please use shorter words."
        )

    # Check for Arabic script - NOW SUPPORTED
    arabic_chars = [c for c in word if "\u0600" <= c <= "\u06ff"]
    if arabic_chars:
        return ValidationResult(
            True,
            None,
            f"Arabic script detected: {word}. Processing with enhanced Arabic phonology support.",
        )

    # Check for other non-Latin scripts - provide guidance
    # Optimized: Only check non-ASCII and non-Arabic, skipping allowed punctuation/whitespace
    allowed_ascii = {"-", "'", " "}
    non_latin_chars = [
        c
        for c in word
        if c not in allowed_ascii
        and not c.isascii()
        and not ("\u0600" <= c <= "\u06ff")
    ]
    if non_latin_chars:
        return ValidationResult(
            True,
            None,
            f"Non-Latin script detected. Attempting analysis: {', '.join(set(non_latin_chars))}",
        )
    if non_latin_chars:
        return ValidationResult(
            True,
            None,
            f"Non-Latin script detected. Attempting analysis: {', '.join(set(non_latin_chars))}",
        )
        return ValidationResult(
            True,
            None,
            f"Non-Latin script detected. Attempting analysis: {', '.join(set(non_latin_chars))}",
        )

    # Must have alphabetic content
    alphabetic_chars = [c for c in word if c.isalpha()]
    if not alphabetic_chars:
        return ValidationResult(
            False, "Word must contain alphabetic characters. Try: pataka, hello, test"
        )

    # Check for non-standard but acceptable characters (warning only)
    standard_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'")
    non_standard = [c for c in word if c not in standard_chars and c.isascii()]

    warning = None
    if non_standard:
        warning = f"Contains unusual characters that may not be processed correctly: {', '.join(set(non_standard))}"

    return ValidationResult(True, None, warning)


# Get phoneme mapping from registry for maintainability
PHONEME_MAPPING = phoneme_registry.get_mapping_dict()


@app.before_request
def before_request():
    """Track request timing and increment counters"""
    g.start_time = time.time()
    api_stats.increment_requests()


@app.after_request
def after_request(response):
    """Log request completion"""
    if hasattr(g, "start_time"):
        duration = round((time.time() - g.start_time) * 1000, 2)
        logger.info(
            f"{request.method} {request.path} - {response.status_code} - {duration}ms"
        )
    return response


def monitor_performance(func):
    """Decorator for monitoring API endpoint performance"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Log performance metrics
            logger.info(f"✅ {func.__name__} completed in {duration:.3f}s")

            # Add performance info to response if it's a JSON response
            if hasattr(result, "get_json") and result.get_json():
                json_data = result.get_json()
                if isinstance(json_data, dict) and "data" in json_data:
                    json_data["performance"] = {
                        "execution_time_ms": round(duration * 1000, 2),
                        "timestamp": datetime.now().isoformat(),
                    }

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {duration:.3f}s: {e}")
            api_stats.increment_errors()
            raise

    return wrapper


@app.route("/")
def home():
    """Serve the interactive web interface"""
    index_file = STATIC_DIR / "index.html"

    if index_file.exists():
        return send_from_directory(STATIC_DIR, "index.html")
    else:
        # Enhanced fallback HTML with sophisticated user guidance
        return render_template_string(
            """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phonological Engine API</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 50px auto; 
                    padding: 20px; 
                    background: #f5f5f5; 
                }
                .container { 
                    background: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                }
                input[type="text"] { 
                    width: 70%; 
                    padding: 12px; 
                    border: 2px solid #bdc3c7; 
                    border-radius: 5px; 
                    font-size: 16px; 
                }
                button { 
                    padding: 12px 20px; 
                    background: #3498db; 
                    color: white; 
                    border: none; 
                    border-radius: 5px; 
                    cursor: pointer; 
                    font-size: 14px; 
                    margin: 5px; 
                }
                button:hover { background: #2980b9; }
                #result { 
                    margin-top: 20px; 
                    min-height: 100px; 
                    border: 2px solid #ecf0f1; 
                    border-radius: 8px; 
                    padding: 15px; 
                    background: #fafafa;
                }
                pre { 
                    white-space: pre-wrap; 
                    word-wrap: break-word; 
                    font-size: 13px; 
                    line-height: 1.4;
                }
                .loading { color: #3498db; font-style: italic; }
                .error { color: #e74c3c; font-weight: bold; }
                .success { color: #27ae60; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Phonological Engine API</h1>
                <p>Professional syllabification service for <strong>Arabic and Latin script</strong></p>
                <h3>Quick Test:</h3>
                <p><em>✨ Now supports Arabic script! Enter Arabic text directly or use romanized forms.</em></p>
                <input type="text" id="word" placeholder="Enter Arabic or romanized word (كتاب, kitab, hello)" value="كتاب">
                <br><br>
                <button onclick="testAPI()">Test Syllabify</button>
                <button onclick="testPhonemes()" style="background: #e74c3c;">Analyze Phonemes</button>
                <div id="result">
                    <p class="loading">Click a button to see results...</p>
                </div>
                <div id="examples" style="margin-top: 15px; font-size: 12px; color: #666;">
                    <strong>Examples:</strong><br>
                    • English: hello, testing, beautiful<br>
                    • Arabic (script): كتاب, مدرسة, المستقيمة, حبيبي<br>
                    • Arabic (romanized): kitab, madrasa, al-mustaqima, habibi<br>
                    • General: pataka, mama, banana
                </div>
            <script>
                async function testAPI() {
                    const word = document.getElementById('word').value;
                    document.getElementById('result').innerHTML = '<p>Loading syllabification...</p>';
                    
                    try {
                        const response = await fetch('/api/syllabify', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({word: word})
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const result = await response.json();
                        document.getElementById('result').innerHTML = 
                            '<h4>✅ Syllabification Result:</h4><pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">' + 
                            JSON.stringify(result, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('result').innerHTML = 
                            '<h4>❌ Error:</h4><p style="color: red;">' + error.message + '</p>';
                        console.error('Syllabify error:', error);
                    }
                }
                
                async function testPhonemes() {
                    const word = document.getElementById('word').value;
                    document.getElementById('result').innerHTML = '<p>Loading phoneme analysis...</p>';
                    
                    try {
                        const response = await fetch('/api/phonemes', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({word: word})
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const result = await response.json();
                        document.getElementById('result').innerHTML = 
                            '<h4>✅ Phoneme Analysis Result:</h4><pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">' + 
                            JSON.stringify(result, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('result').innerHTML = 
                            '<h4>❌ Error:</h4><p style="color: red;">' + error.message + '</p>';
                        console.error('Phonemes error:', error);
                    }
                }
                
                // Auto-test on page load to show immediate results
                window.onload = function() {
                    testPhonemes();
                };
            </script>
        </body>
        </html>
        """
        )


@app.route("/api/syllabify", methods=["POST"])
@monitor_performance
def api_syllabify():
    """Enhanced syllabification endpoint with comprehensive error handling"""
    try:
        # Validate request
        data = request.get_json()
        validation = validate_syllabify_request(data)

        if not validation.is_valid:
            api_stats.increment_errors()
            return jsonify({"success": False, "error": validation.error_message}), 400

        word = data["word"].strip().lower()

        # Convert to phonemes with detailed analysis
        phones = []
        phoneme_analysis = []
        skipped_chars = []

        for i, char in enumerate(word):
            if char in PHONEME_MAPPING:
                meta = PHONEME_MAPPING[char].copy()
                meta["geminated"] = False

                # Enhanced phoneme analysis
                analysis = {
                    "position": i,
                    "character": char,
                    "type": meta["type"],
                    "acoustic_weight": meta["acoustic_weight"],
                    "script": get_script_type(char),
                    "phonological_class": get_phonological_class(char, meta),
                    "articulatory_features": get_articulatory_features(char, meta),
                }

                phones.append(Phoneme(char=char, meta=meta))
                phoneme_analysis.append(analysis)
            elif char in " -'":
                continue  # Skip whitespace and common punctuation
            else:
                skipped_chars.append(char)

        if not phones:
            api_stats.increment_errors()
            return jsonify(
                {"success": False, "error": "No valid phonemes found in input"}, 400
            )

        # Perform syllabification
        result = syllabify(phones)

        # Enhanced syllable analysis
        syllables = []
        for syl_idx, syl in enumerate(result):
            syllable_phonemes = []
            for phone in syl.phones:
                # Find corresponding analysis
                phone_analysis = next(
                    (p for p in phoneme_analysis if p["character"] == phone.char), None
                )
                syllable_phonemes.append(
                    {
                        "character": phone.char,
                        "type": phone.meta.get("type", "unknown"),
                        "acoustic_weight": phone.meta.get("acoustic_weight", 0.5),
                        "analysis": phone_analysis,
                    }
                )

            syllables.append(
                {
                    "index": syl_idx + 1,
                    "phones": [p.char for p in syl.phones],
                    "phonemes": syllable_phonemes,
                    "pattern": syl.pattern,
                    "weight": syl.weight,
                    "syllable_type": classify_syllable_type(syl.pattern),
                    "complexity": calculate_syllable_complexity(syl),
                }
            )

        # Enhanced response with comprehensive phonological analysis
        response_data = {
            "success": True,
            "data": {
                "input": {
                    "word": data["word"],
                    "processed_word": word,
                    "character_count": len(data["word"]),
                    "phoneme_count": len(phones),
                },
                "phoneme_analysis": {
                    "individual_phonemes": phoneme_analysis,
                    "script_distribution": get_script_distribution(phoneme_analysis),
                    "phonological_summary": get_phonological_summary(phoneme_analysis),
                },
                "syllabification": {
                    "syllables": syllables,
                    "syllable_count": len(syllables),
                    "total_weight": round(sum(syl["weight"] for syl in syllables), 2),
                    "average_complexity": (
                        round(
                            sum(syl["complexity"] for syl in syllables)
                            / len(syllables),
                            2,
                        )
                        if syllables
                        else 0
                    ),
                    "syllable_patterns": [syl["pattern"] for syl in syllables],
                    "syllable_types": [syl["syllable_type"] for syl in syllables],
                },
                "linguistic_features": {
                    "primary_script": get_primary_script(phoneme_analysis),
                    "consonant_vowel_ratio": calculate_cv_ratio(phoneme_analysis),
                    "phonological_complexity": calculate_word_complexity(
                        syllables, phoneme_analysis
                    ),
                    "prosodic_weight": round(
                        sum(syl["weight"] for syl in syllables), 2
                    ),
                },
            },
            "warnings": [],
        }

        # Add warnings
        if validation.warning_message:
            response_data["warnings"].append(validation.warning_message)

        if skipped_chars:
            response_data["warnings"].append(
                f"Skipped characters: {', '.join(set(skipped_chars))}"
            )

        return jsonify(response_data)

    except Exception as e:
        api_stats.increment_errors()
        logger.error(f"Syllabification error: {e}")
        logger.error(traceback.format_exc())
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Internal server error during syllabification",
                }
            ),
            500,
        )


@app.route("/api/health", methods=["GET"])
def api_health():
    """Comprehensive health check endpoint"""
    try:
        # Test core functionality
        test_phoneme = Phoneme(
            char="a", meta={"type": "V", "acoustic_weight": 1.0, "geminated": False}
        )
        test_result = syllabify([test_phoneme])

        return jsonify(
            {
                "status": "healthy",
                "engine": "phonological",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "test_syllable_count": len(test_result),
            }
        )
    except Exception as e:
        api_stats.increment_errors()
        logger.error(f"Health check failed: {e}")
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/api/stats", methods=["GET"])
def api_stats_endpoint():
    """API usage statistics endpoint"""
    try:
        stats = api_stats.get_stats()
        stats["timestamp"] = datetime.now().isoformat()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": "Failed to retrieve stats"}), 500


@app.route("/api/phonemes", methods=["POST"])
@monitor_performance
def api_phoneme_analysis():
    """Detailed phoneme analysis endpoint"""
    try:
        # Validate request
        data = request.get_json()
        validation = validate_syllabify_request(data)

        if not validation.is_valid:
            api_stats.increment_errors()
            return jsonify({"success": False, "error": validation.error_message}), 400

        word = data["word"].strip().lower()

        # Detailed phoneme analysis
        phoneme_analysis = []
        character_details = []

        for i, char in enumerate(word):
            char_detail = {
                "position": i,
                "character": char,
                "unicode_codepoint": f"U+{ord(char):04X}",
                "script": get_script_type(char),
                "in_phoneme_mapping": char in PHONEME_MAPPING,
            }

            if char in PHONEME_MAPPING:
                meta = PHONEME_MAPPING[char].copy()
                analysis = {
                    "position": i,
                    "character": char,
                    "type": meta["type"],
                    "acoustic_weight": meta["acoustic_weight"],
                    "script": get_script_type(char),
                    "phonological_class": get_phonological_class(char, meta),
                    "articulatory_features": get_articulatory_features(char, meta),
                    "unicode_info": {
                        "codepoint": f"U+{ord(char):04X}",
                        "name": char,
                        "category": "Letter",
                    },
                }
                phoneme_analysis.append(analysis)
                char_detail["phoneme_data"] = analysis
            elif char in " -'":
                char_detail["category"] = "Punctuation/Whitespace"
            else:
                char_detail["category"] = "Unrecognized"

            character_details.append(char_detail)

        # Comprehensive analysis
        response_data = {
            "success": True,
            "data": {
                "input": {
                    "original_word": data["word"],
                    "processed_word": word,
                    "character_count": len(data["word"]),
                    "phoneme_count": len(phoneme_analysis),
                },
                "character_analysis": character_details,
                "phoneme_analysis": phoneme_analysis,
                "linguistic_statistics": {
                    "script_distribution": get_script_distribution(phoneme_analysis),
                    "phonological_summary": get_phonological_summary(phoneme_analysis),
                    "consonant_vowel_analysis": calculate_cv_ratio(phoneme_analysis),
                    "primary_script": get_primary_script(phoneme_analysis),
                },
                "advanced_features": {
                    "articulatory_inventory": get_articulatory_inventory(
                        phoneme_analysis
                    ),
                    "phoneme_frequency": get_phoneme_frequency(phoneme_analysis),
                    "script_transitions": analyze_script_transitions(character_details),
                },
            },
            "warnings": [],
        }

        # Add warnings
        if validation.warning_message:
            response_data["warnings"].append(validation.warning_message)

        unrecognized = [
            cd for cd in character_details if cd.get("category") == "Unrecognized"
        ]
        if unrecognized:
            response_data["warnings"].append(
                f"Unrecognized characters: {[cd['character'] for cd in unrecognized]}"
            )

        return jsonify(response_data)

    except Exception as e:
        api_stats.increment_errors()
        logger.error(f"Phoneme analysis error: {e}")
        logger.error(traceback.format_exc())
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Internal server error during phoneme analysis",
                }
            ),
            500,
        )


def get_articulatory_inventory(phoneme_analysis: list) -> dict:
    """Get inventory of articulatory features"""
    manners = set()
    places = set()
    voicing = set()

    for analysis in phoneme_analysis:
        features = analysis["articulatory_features"]
        manners.add(features["manner"])
        places.add(features["place"])
        voicing.add(features["voicing"])

    return {
        "manners_of_articulation": list(manners),
        "places_of_articulation": list(places),
        "voicing_values": list(voicing),
        "diversity_score": len(manners) + len(places) + len(voicing),
    }


def get_phoneme_frequency(phoneme_analysis: list) -> dict:
    """Analyze phoneme frequency distribution"""
    char_freq = {}
    type_freq = {"V": 0, "C": 0}
    class_freq = {}

    for analysis in phoneme_analysis:
        char = analysis["character"]
        phoneme_type = analysis["type"]
        phoneme_class = analysis["phonological_class"]

        char_freq[char] = char_freq.get(char, 0) + 1
        type_freq[phoneme_type] += 1
        class_freq[phoneme_class] = class_freq.get(phoneme_class, 0) + 1

    return {
        "character_frequency": char_freq,
        "type_frequency": type_freq,
        "class_frequency": class_freq,
        "most_frequent_character": (
            max(char_freq.items(), key=lambda x: x[1]) if char_freq else None
        ),
        "most_frequent_class": (
            max(class_freq.items(), key=lambda x: x[1]) if class_freq else None
        ),
    }


def analyze_script_transitions(character_details: list) -> dict:
    """Analyze transitions between different scripts"""
    transitions = []
    script_boundaries = []

    for i in range(len(character_details) - 1):
        current_script = character_details[i]["script"]
        next_script = character_details[i + 1]["script"]

        if current_script != next_script:
            transition = f"{current_script} → {next_script}"
            transitions.append(
                {
                    "position": i,
                    "from_script": current_script,
                    "to_script": next_script,
                    "transition": transition,
                }
            )
            script_boundaries.append(i)

    return {
        "transition_count": len(transitions),
        "transitions": transitions,
        "script_boundaries": script_boundaries,
        "is_mixed_script": len(transitions) > 0,
    }


def get_script_type(char: str) -> str:
    """Determine the script type of a character"""
    if "\u0600" <= char <= "\u06ff":
        return "Arabic"
    elif char.isascii() and char.isalpha():
        return "Latin"
    elif char in "ًٌٍَُِْ":  # Arabic diacritics
        return "Arabic_Diacritic"
    else:
        return "Other"


def get_phonological_class(char: str, meta: dict) -> str:
    """Get detailed phonological classification"""
    if meta.get("type") == "V":
        # Vowel classification
        if char in "اآأإ":
            return "Long_Vowel_Central"
        elif char == "و":
            return "Long_Vowel_Back"
        elif char == "ي":
            return "Long_Vowel_Front"
        elif char in "aeiouAEIOU":
            vowel_map = {
                "a": "Low_Central",
                "e": "Mid_Front",
                "i": "High_Front",
                "o": "Mid_Back",
                "u": "High_Back",
            }
            return vowel_map.get(char.lower(), "Vowel")
        else:
            return "Vowel"
    else:
        # Consonant classification
        if char in "بتثجحخدذرزسشصضطظعغفقكلمنه":
            # Arabic consonant classification
            if char in "بتدطك":
                return "Stop"
            elif char in "ثحخذزسشصضظغف":
                return "Fricative"
            elif char in "من":
                return "Nasal"
            elif char in "لر":
                return "Liquid"
            elif char in "جقع":
                return "Pharyngeal"
            else:
                return "Arabic_Consonant"
        else:
            # Latin consonant classification
            if char in "pbtdkg":
                return "Stop"
            elif char in "fvszh":
                return "Fricative"
            elif char in "mn":
                return "Nasal"
            elif char in "lr":
                return "Liquid"
            elif char in "wj":
                return "Glide"
            else:
                return "Consonant"


def get_articulatory_features(char: str, meta: dict) -> dict:
    """Get detailed articulatory features"""
    features = {
        "manner": "unknown",
        "place": "unknown",
        "voicing": "unknown",
        "length": "short",
    }

    if meta.get("type") == "V":
        features["manner"] = "vowel"
        if char in "اآأإ":
            features["place"] = "central"
            features["length"] = "long"
        elif char == "و":
            features["place"] = "back"
            features["length"] = "long"
        elif char == "ي":
            features["place"] = "front"
            features["length"] = "long"
        elif char.lower() in "aeiou":
            vowel_places = {
                "a": "central",
                "e": "front",
                "i": "front",
                "o": "back",
                "u": "back",
            }
            features["place"] = vowel_places.get(char.lower(), "unknown")
    else:
        # Arabic consonants
        arabic_features = {
            "ب": {"manner": "stop", "place": "bilabial", "voicing": "voiced"},
            "ت": {"manner": "stop", "place": "alveolar", "voicing": "voiceless"},
            "ث": {"manner": "fricative", "place": "dental", "voicing": "voiceless"},
            "ج": {"manner": "affricate", "place": "postalveolar", "voicing": "voiced"},
            "ح": {"manner": "fricative", "place": "pharyngeal", "voicing": "voiceless"},
            "خ": {"manner": "fricative", "place": "uvular", "voicing": "voiceless"},
            "د": {"manner": "stop", "place": "alveolar", "voicing": "voiced"},
            "ذ": {"manner": "fricative", "place": "dental", "voicing": "voiced"},
            "ر": {"manner": "trill", "place": "alveolar", "voicing": "voiced"},
            "ز": {"manner": "fricative", "place": "alveolar", "voicing": "voiced"},
            "س": {"manner": "fricative", "place": "alveolar", "voicing": "voiceless"},
            "ش": {
                "manner": "fricative",
                "place": "postalveolar",
                "voicing": "voiceless",
            },
            "ص": {
                "manner": "fricative",
                "place": "alveolar_pharyngealized",
                "voicing": "voiceless",
            },
            "ض": {
                "manner": "stop",
                "place": "alveolar_pharyngealized",
                "voicing": "voiced",
            },
            "ط": {
                "manner": "stop",
                "place": "alveolar_pharyngealized",
                "voicing": "voiceless",
            },
            "ظ": {
                "manner": "fricative",
                "place": "alveolar_pharyngealized",
                "voicing": "voiced",
            },
            "ع": {"manner": "fricative", "place": "pharyngeal", "voicing": "voiced"},
            "غ": {"manner": "fricative", "place": "uvular", "voicing": "voiced"},
            "ف": {
                "manner": "fricative",
                "place": "labiodental",
                "voicing": "voiceless",
            },
            "ق": {"manner": "stop", "place": "uvular", "voicing": "voiceless"},
            "ك": {"manner": "stop", "place": "velar", "voicing": "voiceless"},
            "ل": {"manner": "lateral", "place": "alveolar", "voicing": "voiced"},
            "م": {"manner": "nasal", "place": "bilabial", "voicing": "voiced"},
            "ن": {"manner": "nasal", "place": "alveolar", "voicing": "voiced"},
            "ه": {"manner": "fricative", "place": "glottal", "voicing": "voiceless"},
        }

        # Latin consonants
        latin_features = {
            "p": {"manner": "stop", "place": "bilabial", "voicing": "voiceless"},
            "b": {"manner": "stop", "place": "bilabial", "voicing": "voiced"},
            "t": {"manner": "stop", "place": "alveolar", "voicing": "voiceless"},
            "d": {"manner": "stop", "place": "alveolar", "voicing": "voiced"},
            "k": {"manner": "stop", "place": "velar", "voicing": "voiceless"},
            "g": {"manner": "stop", "place": "velar", "voicing": "voiced"},
            "f": {
                "manner": "fricative",
                "place": "labiodental",
                "voicing": "voiceless",
            },
            "v": {"manner": "fricative", "place": "labiodental", "voicing": "voiced"},
            "s": {"manner": "fricative", "place": "alveolar", "voicing": "voiceless"},
            "z": {"manner": "fricative", "place": "alveolar", "voicing": "voiced"},
            "h": {"manner": "fricative", "place": "glottal", "voicing": "voiceless"},
            "m": {"manner": "nasal", "place": "bilabial", "voicing": "voiced"},
            "n": {"manner": "nasal", "place": "alveolar", "voicing": "voiced"},
            "l": {"manner": "lateral", "place": "alveolar", "voicing": "voiced"},
            "r": {"manner": "approximant", "place": "alveolar", "voicing": "voiced"},
            "w": {"manner": "glide", "place": "bilabial", "voicing": "voiced"},
            "j": {"manner": "glide", "place": "palatal", "voicing": "voiced"},
        }

        if char in arabic_features:
            features.update(arabic_features[char])
        elif char.lower() in latin_features:
            features.update(latin_features[char.lower()])

    return features


def classify_syllable_type(pattern: str) -> str:
    """Classify syllable type based on CV pattern"""
    if pattern == "V":
        return "Open_Light"
    elif pattern == "CV":
        return "Open_Light"
    elif pattern == "CVC":
        return "Closed_Heavy"
    elif pattern == "CVV":
        return "Open_Heavy"
    elif pattern == "CVCC":
        return "Closed_Superheavy"
    elif pattern.startswith("CC"):
        return "Complex_Onset"
    elif pattern.endswith("CC"):
        return "Complex_Coda"
    else:
        return f'Complex_{len([c for c in pattern if c == "C"])}C_{len([c for c in pattern if c == "V"])}V'


def calculate_syllable_complexity(syllable) -> float:
    """Calculate syllable complexity score"""
    complexity = 0.0

    # Base complexity from pattern
    pattern = syllable.pattern
    consonant_count = pattern.count("C")
    vowel_count = pattern.count("V")

    # More consonants = higher complexity
    complexity += consonant_count * 0.5

    # Multiple vowels add moderate complexity
    complexity += max(0, vowel_count - 1) * 0.3

    # Cluster complexity
    if "CC" in pattern:
        complexity += 1.0  # Consonant clusters are complex

    # Weight-based complexity
    complexity += syllable.weight * 0.2

    return round(complexity, 2)


def get_script_distribution(phoneme_analysis: list) -> dict:
    """Analyze script distribution in the phoneme analysis"""
    script_counts = {}
    for analysis in phoneme_analysis:
        script = analysis["script"]
        script_counts[script] = script_counts.get(script, 0) + 1

    total = len(phoneme_analysis)
    return {
        "counts": script_counts,
        "percentages": {
            script: round((count / total) * 100, 1)
            for script, count in script_counts.items()
        },
    }


def get_phonological_summary(phoneme_analysis: list) -> dict:
    """Generate phonological summary statistics"""
    vowels = [p for p in phoneme_analysis if p["type"] == "V"]
    consonants = [p for p in phoneme_analysis if p["type"] == "C"]

    return {
        "total_phonemes": len(phoneme_analysis),
        "vowel_count": len(vowels),
        "consonant_count": len(consonants),
        "vowel_types": list(set(p["phonological_class"] for p in vowels)),
        "consonant_types": list(set(p["phonological_class"] for p in consonants)),
        "acoustic_weight_average": (
            round(
                sum(p["acoustic_weight"] for p in phoneme_analysis)
                / len(phoneme_analysis),
                2,
            )
            if phoneme_analysis
            else 0
        ),
    }


def get_primary_script(phoneme_analysis: list) -> str:
    """Determine the primary script of the word"""
    script_counts = {}
    for analysis in phoneme_analysis:
        script = analysis["script"]
        script_counts[script] = script_counts.get(script, 0) + 1

    if not script_counts:
        return "Unknown"

    primary = max(script_counts.items(), key=lambda x: x[1])
    return primary[0]


def calculate_cv_ratio(phoneme_analysis: list) -> dict:
    """Calculate consonant-vowel ratio and distribution"""
    vowels = len([p for p in phoneme_analysis if p["type"] == "V"])
    consonants = len([p for p in phoneme_analysis if p["type"] == "C"])

    total = vowels + consonants
    if total == 0:
        return {"ratio": 0, "vowel_percentage": 0, "consonant_percentage": 0}

    return {
        "consonant_vowel_ratio": (
            round(consonants / vowels, 2) if vowels > 0 else float("inf")
        ),
        "vowel_percentage": round((vowels / total) * 100, 1),
        "consonant_percentage": round((consonants / total) * 100, 1),
        "vowel_count": vowels,
        "consonant_count": consonants,
    }


def calculate_word_complexity(syllables: list, phoneme_analysis: list) -> dict:
    """Calculate overall word complexity metrics"""
    if not syllables or not phoneme_analysis:
        return {"score": 0, "factors": []}

    complexity_factors = []
    base_score = 0

    # Syllable count complexity
    syllable_count = len(syllables)
    if syllable_count > 3:
        complexity_factors.append(f"High syllable count ({syllable_count})")
        base_score += syllable_count * 0.5

    # Average syllable complexity
    avg_complexity = sum(syl["complexity"] for syl in syllables) / len(syllables)
    if avg_complexity > 2.0:
        complexity_factors.append(
            f"Complex syllable structure (avg: {avg_complexity:.1f})"
        )
        base_score += avg_complexity

    # Script mixing complexity
    scripts = set(p["script"] for p in phoneme_analysis)
    if len(scripts) > 1:
        complexity_factors.append(f"Mixed scripts ({', '.join(scripts)})")
        base_score += len(scripts) * 0.8

    # Consonant cluster complexity
    consonant_clusters = sum("CC" in syl["pattern"] for syl in syllables)
    if consonant_clusters > 0:
        complexity_factors.append(f"Consonant clusters ({consonant_clusters})")
        base_score += consonant_clusters * 0.7

    # Phoneme diversity
    unique_phonological_classes = set(p["phonological_class"] for p in phoneme_analysis)
    if len(unique_phonological_classes) > 4:
        complexity_factors.append(
            f"High phoneme diversity ({len(unique_phonological_classes)} classes)"
        )
        base_score += len(unique_phonological_classes) * 0.2
        complexity_factors.append(
            f"High phoneme diversity ({len(unique_phonological_classes)} classes)"
        )
        base_score += len(unique_phonological_classes) * 0.2

    return {
        "score": round(base_score, 2),
        "level": "Low" if base_score < 2 else "Medium" if base_score < 5 else "High",
        "factors": complexity_factors,
        "syllable_contribution": round(avg_complexity, 2),
        "phoneme_contribution": round(len(unique_phonological_classes) * 0.2, 2),
    }


@app.route("/api/registry", methods=["GET"])
def api_phoneme_registry():
    """Phoneme registry information and statistics"""
    try:
        registry_stats = phoneme_registry.get_statistics()

        # Add deployment and configuration info
        response_data = {
            "success": True,
            "data": {
                "registry_statistics": registry_stats,
                "deployment_info": {
                    "mode": DEPLOYMENT_MODE,
                    "stats_backend": type(api_stats).__name__,
                    "redis_available": REDIS_AVAILABLE,
                    "instance_id": INSTANCE_ID,
                },
                "configuration": {
                    "supported_scripts": ["Latin", "Arabic"],
                    "api_version": "2.0.0",
                    "features": [
                        "Multi-script phoneme analysis",
                        "Distributed statistics tracking",
                        "Modular phoneme registry",
                        "Advanced articulatory features",
                        "Production-ready architecture",
                    ],
                },
                "registry_info": {
                    "total_phonemes": len(PHONEME_MAPPING),
                    "registry_backend": "modular",
                    "maintainability": "high",
                    "extensibility": "supported",
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(response_data)

    except Exception as e:
        api_stats.increment_errors()
        logger.error(f"Registry info error: {e}")
        return (
            jsonify(
                {"success": False, "error": "Failed to retrieve registry information"}
            ),
            500,
        )


if __name__ == "__main__":
    print("🚀 Starting Professional Phonological Engine Web API...")
    print("📍 URL: http://localhost:5000")
    print("📖 Interactive Interface: http://localhost:5000")
    print("🔧 API Endpoints:")
    print("   POST /api/syllabify    - Enhanced syllable analysis with phoneme details")
    print("   POST /api/phonemes     - Comprehensive phoneme analysis with linguistics")
    print("   GET  /api/registry     - Phoneme registry info and system configuration")
    print("   GET  /api/health       - Health check and system status")
    print("   GET  /api/stats        - API usage statistics (distributed-ready)")

    # Create static directory if it doesn't exist - using module-level STATIC_DIR
    STATIC_DIR.mkdir(exist_ok=True)

    app.run(debug=True, host="0.0.0.0", port=5000)

active_sessions = {}
