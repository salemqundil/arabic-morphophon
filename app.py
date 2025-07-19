"""
🚀 Dynamic Full-Stack Arabic Phonology Engine Web Application
Real-time analysis with WebSocket support and advanced features
"""
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global session storage for when SocketIO is not available
active_sessions = {}

# Try to import optional dependencies with proper fallbacks
try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False
    CORS = None
    print("⚠️ Flask-CORS not available. CORS features disabled.")

try:
    from flask_socketio import SocketIO, emit
    socketio_available = True
except ImportError:
    socketio_available = False
    SocketIO = None
    emit = None
    print("⚠️ Flask-SocketIO not available. Real-time features disabled.")

# Try to import analysis modules with proper fallbacks
try:
    from arabic_morphophon.phonology.analyzer import analyze_phonemes
    from arabic_morphophon.phonology.normalizer import normalize_text
    from arabic_morphophon.phonology.syllable_encoder import encode_syllables
    modules_available = True
    print("✅ Analysis modules loaded successfully.")
except ImportError as e:
    modules_available = False
    print(f"⚠️ Analysis modules not available: {e}")
    
    # Create fallback functions
    def analyze_phonemes(text):
        """Fallback function for phoneme analysis"""
        return [{"char": char, "type": "unknown"} for char in text if char.strip()]
    
    def normalize_text(text):
        """Fallback function for text normalization"""
        return text.strip()
    
    def encode_syllables(text):
        """Fallback function for syllable encoding"""
        return [{"letter": char, "classification": "unknown", "vowel": None, "syllable_code": "unknown"} 
                for char in text if char.strip()]

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'arabic_phonology_engine_secret_2025'

# Initialize extensions with error handling
if cors_available and CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})

socketio = None
if socketio_available and SocketIO:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for statistics and caching
analysis_stats = {
    'total_analyses': 0,
    'total_characters': 0,
    'unique_texts': set(),
    'start_time': datetime.now()
}

analysis_cache = {}

# Thread safety locks
cache_lock = threading.Lock()
stats_lock = threading.Lock()

def generate_session_id():
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def get_cache_key(text):
    """Generate a secure cache key for text"""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

def update_stats(text):
    """Update global statistics with thread safety"""
    with stats_lock:
        analysis_stats['total_analyses'] += 1
        analysis_stats['total_characters'] += len(text)
        
        # Limit unique_texts to prevent memory issues
        if len(analysis_stats['unique_texts']) < 1000:
            analysis_stats['unique_texts'].add(text.strip())

def cleanup_old_cache():
    """Remove old cache entries (older than 1 hour)"""
    with cache_lock:
        current_time = time.time()
        keys_to_remove = []
        for key, data in analysis_cache.items():
            if current_time - data.get('timestamp', 0) > 3600:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del analysis_cache[key]

def safe_emit(event, data):
    """Safely emit SocketIO events with fallback"""
    if socketio and emit:
        try:
            socketio.emit(event, data)
        except Exception as e:
            print(f"Warning: Failed to emit {event}: {e}")

def get_session_id_from_request():
    """Safely get session ID from request"""
    if hasattr(request, 'sid'):
        return request.sid
    return generate_session_id()

# Routes
@app.route('/')
def index():
    """Serve the dynamic main page with enhanced features"""
    return render_template('index.html', 
                         stats=get_application_stats(),
                         version="2.0.0",
                         features=['Real-time Analysis', 'WebSocket Support', 'Caching', 'Statistics'])

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Enhanced API endpoint for analyzing Arabic text with caching and real-time updates"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        session_id = data.get('session_id', generate_session_id())
        real_time = data.get('real_time', False)
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Check cache first
        cache_key = get_cache_key(text)
        with cache_lock:
            if cache_key in analysis_cache:
                cached_result = analysis_cache[cache_key]
                if real_time:
                    safe_emit('analysis_cached', {
                        'session_id': session_id,
                        'cached': True,
                        'timestamp': cached_result['timestamp']
                    })
                return jsonify(cached_result['data'])
        
        # Emit real-time status updates
        if real_time:
            safe_emit('analysis_progress', {
                'session_id': session_id,
                'step': 'starting',
                'progress': 0
            })
        
        # Perform analysis with progress updates
        start_time = time.time()
        
        # Step 1: Normalize
        if real_time:
            safe_emit('analysis_progress', {
                'session_id': session_id,
                'step': 'normalization',
                'progress': 25
            })
        
        normalized = normalize_text(text)
        
        # Step 2: Phoneme Analysis
        if real_time:
            safe_emit('analysis_progress', {
                'session_id': session_id,
                'step': 'phoneme_analysis',
                'progress': 50
            })
        
        phoneme_analysis = analyze_phonemes(text)
        
        # Step 3: Syllable Encoding
        if real_time:
            safe_emit('analysis_progress', {
                'session_id': session_id,
                'step': 'syllable_encoding',
                'progress': 75
            })
        
        syllable_encoding = encode_syllables(text)
        
        # Calculate processing time and additional metrics
        processing_time = time.time() - start_time
        
        # Prepare enhanced response
        response = {
            'original_text': text,
            'normalized_text': normalized,
            'phoneme_analysis': phoneme_analysis,
            'syllable_encoding': syllable_encoding,
            'session_id': session_id,
            'processing_time': round(processing_time, 4),
            'character_count': len(text),
            'word_count': len(text.split()),
            'phoneme_count': len(phoneme_analysis) if phoneme_analysis else 0,
            'syllable_count': len(syllable_encoding) if syllable_encoding else 0,
            'timestamp': time.time(),
            'status': 'success',
            'modules_available': modules_available
        }
        
        # Cache the result
        with cache_lock:
            analysis_cache[cache_key] = {
                'data': response,
                'timestamp': time.time()
            }
        
        # Update statistics
        update_stats(text)
        
        # Emit completion event
        if real_time:
            safe_emit('analysis_completed', {
                'session_id': session_id,
                'processing_time': processing_time,
                'progress': 100
            })
        
        # Cleanup old cache entries periodically
        if len(analysis_cache) > 100:
            cleanup_old_cache()
        
        return jsonify(response)
        
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format'}), 400
    except UnicodeDecodeError:
        return jsonify({'error': 'Text encoding error'}), 400
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get application statistics"""
    return jsonify(get_application_stats())

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear analysis cache"""
    with cache_lock:
        analysis_cache.clear()
    return jsonify({'message': 'Cache cleared successfully', 'timestamp': time.time()})

@app.route('/api/examples')
def get_examples():
    """Get predefined Arabic text examples"""
    examples = [
        {
            'id': 1,
            'title': 'السلام عليكم',
            'text': 'السَّلامُ عَلَيْكُم',
            'description': 'Traditional Arabic greeting',
            'difficulty': 'beginner'
        },
        {
            'id': 2,
            'title': 'مرحبا بالعالم',
            'text': 'مَرْحَباً بِالعالَم',
            'description': 'Hello World in Arabic',
            'difficulty': 'beginner'
        },
        {
            'id': 3,
            'title': 'التعلم العميق',
            'text': 'نَحْنُ نَتَعَلَّمُ العَرَبِيَّة بِالذَّكاءِ الاصْطِناعِيّ',
            'description': 'Learning Arabic with AI',
            'difficulty': 'advanced'
        },
        {
            'id': 4,
            'title': 'الشعر العربي',
            'text': 'وَمَا نَيْلُ المَطالِبِ بِالتَّمَنّي وَلَكِن تُؤخَذُ الدُّنيا غِلاباً',
            'description': 'Classical Arabic poetry',
            'difficulty': 'expert'
        }
    ]
    return jsonify(examples)

def get_application_stats():
    """Get comprehensive application statistics"""
    uptime = datetime.now() - analysis_stats['start_time']
    with stats_lock:
        stats = {
            'total_analyses': analysis_stats['total_analyses'],
            'total_characters': analysis_stats['total_characters'],
            'unique_texts': len(analysis_stats['unique_texts']),
            'cache_size': len(analysis_cache),
            'active_sessions': len(active_sessions),
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime).split('.')[0],
            'avg_chars_per_analysis': (
                analysis_stats['total_characters'] / analysis_stats['total_analyses'] 
                if analysis_stats['total_analyses'] > 0 else 0
            ),
            'server_status': 'healthy',
            'modules_available': modules_available,
            'socketio_available': socketio_available,
            'features': {
                'real_time': socketio_available,
                'caching': True,
                'statistics': True,
                'examples': True
            }
        }
    return stats

# WebSocket Events (if SocketIO is available)
if socketio and emit:
    @socketio.on('connect')
    def on_connect():
        session_id = generate_session_id()
        client_id = get_session_id_from_request()
        active_sessions[client_id] = {
            'session_id': session_id,
            'connected_at': time.time(),
            'analyses_count': 0
        }
        emit('connected', {
            'session_id': session_id,
            'server_time': time.time(),
            'message': 'Connected to Arabic Phonology Engine'
        })
        print(f"✅ Client connected: {client_id}")

    @socketio.on('disconnect')
    def on_disconnect():
        client_id = get_session_id_from_request()
        if client_id in active_sessions:
            session_data = active_sessions[client_id]
            print(f"❌ Client disconnected: {client_id} (Session: {session_data['session_id']})")
            del active_sessions[client_id]

    @socketio.on('analyze_realtime')
    def handle_realtime_analysis(data):
        """Handle real-time analysis requests via WebSocket"""
        try:
            text = data.get('text', '').strip()
            if not text:
                emit('analysis_error', {'error': 'No text provided'})
                return

            client_id = get_session_id_from_request()
            session_id = active_sessions.get(client_id, {}).get('session_id', generate_session_id())
            
            # Emit analysis started
            emit('analysis_started', {
                'session_id': session_id,
                'text_length': len(text),
                'timestamp': time.time()
            })

            # Check cache
            cache_key = get_cache_key(text)
            with cache_lock:
                if cache_key in analysis_cache:
                    emit('analysis_cached', analysis_cache[cache_key]['data'])
                    return

            # Perform analysis with progress updates
            start_time = time.time()

            # Progress updates
            emit('analysis_progress', {'step': 'normalization', 'progress': 25})
            normalized = normalize_text(text)

            emit('analysis_progress', {'step': 'phoneme_analysis', 'progress': 50})
            phoneme_analysis = analyze_phonemes(text)

            emit('analysis_progress', {'step': 'syllable_encoding', 'progress': 75})
            syllable_encoding = encode_syllables(text)

            processing_time = time.time() - start_time

            # Prepare result
            result = {
                'original_text': text,
                'normalized_text': normalized,
                'phoneme_analysis': phoneme_analysis,
                'syllable_encoding': syllable_encoding,
                'session_id': session_id,
                'processing_time': round(processing_time, 4),
                'character_count': len(text),
                'word_count': len(text.split()),
                'phoneme_count': len(phoneme_analysis) if phoneme_analysis else 0,
                'syllable_count': len(syllable_encoding) if syllable_encoding else 0,
                'timestamp': time.time(),
                'status': 'success',
                'modules_available': modules_available
            }

            # Cache result
            with cache_lock:
                analysis_cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time()
                }

            # Update session stats
            if client_id in active_sessions:
                active_sessions[client_id]['analyses_count'] += 1

            # Update global stats
            update_stats(text)

            # Emit completion
            emit('analysis_completed', result)

        except Exception as e:
            emit('analysis_error', {'error': str(e), 'timestamp': time.time()})

    @socketio.on('get_stats')
    def handle_get_stats():
        """Handle real-time stats requests"""
        emit('stats_update', get_application_stats())

@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Arabic Phonology Engine',
        'version': '2.0.0',
        'timestamp': time.time(),
        'modules_available': modules_available,
        'socketio_available': socketio_available,
        'cors_available': cors_available,
        'features': ['Dynamic Analysis', 'Real-time Updates', 'Caching', 'Statistics'],
        'uptime': str(datetime.now() - analysis_stats['start_time']).split('.')[0]
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files with caching headers"""
    try:
        response = send_from_directory('static', filename)
        # Add caching headers for static files
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

# Background task for periodic cleanup
def background_cleanup():
    """Background task to clean up old cache entries"""
    while True:
        time.sleep(300)  # Run every 5 minutes
        try:
            cleanup_old_cache()
            print(f"🧹 Cache cleanup completed. Current cache size: {len(analysis_cache)}")
        except Exception as e:
            print(f"Error during cache cleanup: {e}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Start background cleanup task
    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    cleanup_thread.start()
    
    print("🚀 Starting Dynamic Arabic Phonology Engine Server...")
    print("📍 Access the application at: http://localhost:5000")
    print("🔄 Features: Real-time analysis, WebSocket support, caching, statistics")
    print(f"📊 Modules available: {modules_available}")
    print(f"⚡ SocketIO available: {socketio_available}")
    print(f"🌐 CORS available: {cors_available}")
    
    try:
        if socketio:
            # Run with SocketIO support
            socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
        else:
            # Fallback to regular Flask
            app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Falling back to basic Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
