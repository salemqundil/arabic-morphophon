# Arabic Morphophonology Engine

A comprehensive Arabic morphophonological analysis engine with modern NLP capabilities.

## Overview

This project provides advanced Arabic language processing tools including:

- **Morphological Analysis**: Root extraction, pattern recognition, and word formation analysis
- **Phonological Processing**: Sound change rules, syllable structure analysis
- **Derivation Engine**: Automatic word derivation from roots and patterns
- **Web Interface**: User-friendly interface for testing and analysis
- **API Integration**: RESTful API for programmatic access

## Features

### Core Engines
- **Derivation Engine**: Advanced word derivation from tri- and quadri-literal roots
- **Frozen Root Engine**: Detection and analysis of frozen/lexicalized forms
- **Phonological Engine**: Comprehensive phonological rule application
- **Morphological Engine**: Detailed morphological decomposition
- **Sentiment Analysis**: Arabic sentiment classification
- **Tokenization**: Advanced Arabic text tokenization

### Development Environment
- **Docker Support**: Complete containerized development environment
- **Local Development**: Hot-reload development setup
- **Production Deployment**: Scalable production configuration
- **Testing Suite**: Comprehensive test coverage

## Quick Start

### Local Development

1. **Prerequisites**
   - Docker and Docker Compose
   - Node.js and npm
   - Python 3.11+

2. **Start Development Environment**
   ```bash
   # Using the local development script
   .\local-dev.ps1 start

   # Or using npm scripts
   npm run docker:dev:detached
   ```

3. **Access Services**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Database Admin: http://localhost:5050

## Installation / التثبيت

```bash
pip install -r requirements.txt
```

## Usage / الاستخدام

```python
from arabic morphophonological engine import_data Engine
engine = Engine()
result = engine.process(text)
```

## Documentation / التوثيق

For detailed documentation, see the [docs](docs/) directory.
للحصول على التوثيق المفصل، راجع دليل [docs](docs/).

## Contributing / المساهمة

Contributions are welcome! Please read the contributing guidelines.
نرحب بالمساهمات! يرجى قراءة إرشادات المساهمة.

## License / الترخيص

This project is licensed under the MIT License.
هذا المشروع مرخص تحت رخصة MIT.

## Author / المؤلف

Arabic NLP Team
