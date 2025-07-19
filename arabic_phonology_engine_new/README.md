# Arabic Phonology Engine

## Overview

A comprehensive Arabic phonology analysis system with zero-tolerance for encoding issues and expert-level linguistic accuracy.

## Features

- **Expert-level Phoneme Database**: 23+ phonemes with detailed linguistic properties
- **Advanced Analysis**: Syllable analysis, phoneme classification, and normalization
- **Web Interface**: RESTful API and web application
- **Zero Tolerance**: Robust handling of encoding issues
- **Production Ready**: Docker support, comprehensive testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/arabic-phonology/arabic_phonology_engine_new.git
cd arabic_phonology_engine_new

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Basic Usage

```python
from arabic_phonology import ArabicPhonologyEngine

# Initialize the engine
engine = ArabicPhonologyEngine()

# Analyze Arabic text (marhaba - hello)
result = engine.analyze("مرحبا")
print(result.phonemes)
print(result.syllables)
```

### Web API

```bash
# Start the web server
python -m arabic_phonology.web.app

# Test the API
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا"}'
```

## Project Structure

```text
arabic_phonology_engine_new/
├── src/
│   └── arabic_phonology/          # Main package
│       ├── core/                  # Core engine components
│       ├── analysis/              # Analysis modules
│       ├── data/                  # Data models
│       ├── utils/                 # Utilities
│       └── web/                   # Web interface
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── performance/               # Performance tests
├── docs/                          # Documentation
├── scripts/                       # Setup/deployment scripts
├── tools/                         # Development tools
├── config/                        # Configuration files
└── examples/                      # Usage examples
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arabic_phonology

# Run specific test types
pytest tests/unit/         # Unit tests only
pytest tests/integration/  # Integration tests only
pytest tests/performance/  # Performance tests only
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Docker Development

```bash
# Build container
docker build -t arabic-phonology-engine .

# Run container
docker run -p 5000:5000 arabic-phonology-engine

# Development with docker-compose
docker-compose up -d
```

## API Reference

### Core Classes

- `ArabicPhonologyEngine`: Main analysis engine
- `PhonemeDatabase`: Phoneme data management
- `ArabicAnalyzer`: Text analysis and processing

### Analysis Methods

- `analyze(text)`: Complete phonological analysis
- `extract_phonemes(text)`: Extract phoneme sequence
- `syllabify(text)`: Break text into syllables
- `classify_phonemes(phonemes)`: Classify phoneme types

## Contributing

See the development guidelines in the repository documentation.

## License

MIT License - see [LICENSE](LICENSE) for details.
