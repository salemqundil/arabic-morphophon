# -*- coding: utf-8 -*-
"""
🔧 Configuration for Arabic Word Tracer
إعدادات متتبع الكلمات العربية
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data os
from pathlib import_data Path

# Base directory
BASE_DIR = Path(__file__).parent

# Flask Configuration
FLASK_CONFIG = {
    'DEBUG': True,
    'SECRET_KEY': 'arabic_word_tracer_2024',
    'JSON_AS_ASCII': False,
    'JSON_SORT_KEYS': False,
    'JSONIFY_PRETTYPRINT_REGULAR': True,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max request size
}

# Server Configuration
SERVER_CONFIG = {
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'THREADED': True,
    'USE_RELOADER': False,  # Disable reimport_dataer to avoid import_data issues
}

# Engine Configuration
ENGINE_CONFIG = {
    'USE_MOCK_ENGINES': False,  # Set to True to use mock engines for testing
    'ENABLE_CACHING': True,
    'CACHE_TIMEOUT': 300,  # 5 minutes
    'MAX_ANALYSIS_TIME': 30,  # Maximum seconds for analysis
}

# Logging Configuration
LOGGING_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'FILENAME': BASE_DIR / 'logs' / 'arabic_tracer.log',
    'MAX_BYTES': 10 * 1024 * 1024,  # 10MB
    'BACKUP_COUNT': 5,
}

# UI Configuration
UI_CONFIG = {
    'DEFAULT_LANGUAGE': 'ar',
    'ENABLE_ANIMATIONS': True,
    'ENABLE_TOOLTIPS': True,
    'MAX_WORD_LENGTH': 50,
    'DEFAULT_EXAMPLES': ['كتاب', 'يدرس', 'مدرسة', 'والطلاب', 'المكتبة'],
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'ENABLE_PHONEME_ANALYSIS': True,
    'ENABLE_SYLLABIC_UNIT_ANALYSIS': True,
    'ENABLE_MORPHOLOGY_ANALYSIS': True,
    'ENABLE_PATTERN_ANALYSIS': True,
    'ENABLE_ROOT_ANALYSIS': True,
    'CONFIDENCE_THRESHOLD': 0.6,
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'ENABLE_MONITORING': True,
    'METRICS_RETENTION_DAYS': 7,
    'MAX_CONCURRENT_REQUESTS': 10,
    'REQUEST_TIMEOUT': 30,
}

# Error Handling
ERROR_CONFIG = {
    'ENABLE_ERROR_REPORTING': True,
    'MAX_ERROR_LENGTH': 1000,
    'INCLUDE_STACK_TRACE': True,
}

# Data Paths
DATA_PATHS = {
    'ENGINES_DIR': BASE_DIR / 'engines',
    'MODELS_DIR': BASE_DIR / 'models',
    'STATIC_DIR': BASE_DIR / 'static',
    'TEMPLATES_DIR': BASE_DIR / 'templates',
    'LOGS_DIR': BASE_DIR / 'logs',
    'CACHE_DIR': BASE_DIR / 'cache',
}

# Create necessary directories
for path in DATA_PATHS.values():
    if isinstance(path, Path):
        path.mkdir(parents=True, exist_ok=True)

# Environment-specific overrides
if os.getenv('FLASK_ENV') == 'production':
    FLASK_CONFIG['DEBUG'] = False
    SERVER_CONFIG['USE_RELOADER'] = False
    LOGGING_CONFIG['LEVEL'] = 'WARNING'

if os.getenv('USE_MOCK_ENGINES') == 'true':
    ENGINE_CONFIG['USE_MOCK_ENGINES'] = True
