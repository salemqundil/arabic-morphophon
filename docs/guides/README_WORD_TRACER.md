# 🔍 Arabic Word Tracer - متتبع الكلمات العربية

A comprehensive browser-based interface for tracing Arabic words through all linguistic levels from phonemes to roots. This application combines UI expertise with advanced NLP capabilities to provide detailed morphophonological analysis.

## 🌟 Features

### Linguistic Analysis Levels
- **🔊 Phonemes (الأصوات)**: Phonetic decomposition and IPA representation
- **📝 Harakat (الحركات)**: Diacritical marks analysis and density metrics
- **🔗 SyllabicUnits (المقاطع)**: SyllabicAnalysis with CV patterns and prosodic weight
- **⚛️ Particles (الجسيمات)**: Grammatical particle identification and classification
- **🏷️ Word Class (تصنيف الكلمة)**: Noun/verb/particle classification with features
- **⚖️ Patterns (الأوزان)**: Morphological pattern matching and weight analysis
- **📏 Weight (الوزن الصرفي)**: Morphological weight calculation and categorization
- **🌳 Root (الجذر)**: Root extraction with semantic field identification
- **📎 Affixes (اللواصق)**: Prefix/suffix identification and analysis
- **💎 Infinitive & Pure (المصدر والمجرد)**: Base form derivation and complexity assessment

### Advanced UI Features
- **🎨 Professional Arabic-first design** with RTL support
- **📊 Interactive visualizations** for each linguistic level
- **⚡ Real-time analysis** with progressive disclosure
- **📱 Responsive design** for all devices
- **🔍 Step-by-step tracing** with detailed explanations
- **🎯 Expert-level insights** and confidence metrics
- **♿ Accessibility support** with keyboard navigation
- **🌙 Dark mode support** and high contrast options

## 🚀 Quick Begin

### Prerequisites
```bash
pip install flask flask-cors
```

### Running the Application

#### Option 1: Using the Launch Script (Recommended)
```bash
python run_tracer.py
```

#### Option 2: With Custom Configuration
```bash
# Development mode
python run_tracer.py --debug --host localhost --port 8000

# Mock engines for testing
python run_tracer.py --mock-engines

# Setup only (no server begin)
python run_tracer.py --setup-only
```

#### Option 3: Direct Flask App
```bash
python arabic_word_tracer_app.py
```

### Accessing the Interface
1. Open your browser
2. Navigate to `http://localhost:5000`
3. Enter an Arabic word in the input field
4. Click "تتبع الكلمة" (Trace Word)
5. Explore the comprehensive linguistic analysis

## 📁 Project Structure

```
arabic-word-tracer/
├── arabic_word_tracer_app.py      # Main Flask application
├── run_tracer.py                  # Launch script with configuration
├── config.py                      # Configuration settings
├── templates/
│   └── word_tracer.html          # Main UI template
├── static/
│   ├── advanced-styles.css       # Enhanced CSS
│   └── advanced-enhancements.js  # Advanced JavaScript features
├── engines/                      # NLP engines (if available)
│   ├── nlp/
│   │   ├── particles/
│   │   ├── morphology/
│   │   ├── phonology/
│   │   └── frozen_root/
│   └── ...
├── logs/                         # Application logs
└── sample_data/                  # Test data
```

## 🔧 Configuration

### Engine Configuration
The application supports both real NLP engines and mock engines for testing:

```python
# config.py
ENGINE_CONFIG = {
    'USE_MOCK_ENGINES': False,  # Set to True for testing
    'ENABLE_CACHING': True,
    'MAX_ANALYSIS_TIME': 30,
}
```

### UI Configuration
```python
UI_CONFIG = {
    'DEFAULT_LANGUAGE': 'ar',
    'ENABLE_ANIMATIONS': True,
    'MAX_WORD_LENGTH': 50,
    'DEFAULT_EXAMPLES': ['كتاب', 'يدرس', 'مدرسة'],
}
```

## 🎯 API Endpoints

### POST /api/trace
Analyze an Arabic word through all linguistic levels.

**Request:**
```json
{
    "word": "كتاب"
}
```

**Response:**
```json
{
    "input_word": "كتاب",
    "trace_id": "trace_1234567890_123",
    "linguistic_levels": {
        "phonemes": {
            "phonemes_list": ["ك", "ت", "ا", "ب"],
            "phoneme_count": 4,
            "status": "success"
        },
        "syllabic_units": {
            "syllabic_units": ["كِ", "تاب"],
            "syllabic_unit_count": 2,
            "cv_pattern": "CV-CVVC",
            "status": "success"
        },
        // ... other levels
    },
    "trace_summary": {
        "word_complexity_score": 0.65,
        "analysis_confidence": 0.89,
        "dominant_characteristics": ["جذر ثلاثي"]
    }
}
```

### GET /api/stats
Get performance statistics.

### GET /api/engines
Check engine availability and status.

## 🎨 UI Components

### Linguistic Level Cards
Each linguistic level is presented in an interactive card with:
- **Header with icon and status indicator**
- **Collapsible content area**
- **Data visualization components**
- **Progress and confidence metrics**

### Visualization Types
- **Phoneme containers**: Individual phoneme display
- **SyllabicUnit structures**: Visual syllabic_unit breakdown
- **Morphological trees**: Hierarchical structure display
- **Pattern matching**: Visual pattern comparison
- **Confidence meters**: Animated confidence indicators

### Interactive Features
- **Click to expand/collapse** linguistic levels
- **Hover effects** with detailed tooltips
- **Keyboard navigation** for accessibility
- **Real-time example selection**

## 🔬 Linguistic Analysis Details

### Phoneme Analysis
- Phonetic decomposition of Arabic letters
- IPA (International Phonetic Alphabet) representation
- Phoneme classification (consonants, vowels, semivowels)
- Sound feature analysis

### Harakat Analysis
- Diacritical mark identification and counting
- Diacritization density calculation
- Harakat type breakdown (fatha, kasra, damma, etc.)
- Text normalization (clean word extraction)

### SyllabicUnit Analysis
- SyllabicUnit boundary detection
- CV pattern generation
- SyllabicUnit type classification (light, heavy, superheavy)
- Prosodic weight calculation
- Stress pattern prediction

### Morphological Analysis
- Root extraction and identification
- Pattern (wazn) matching
- Affix identification (prefixes, suffixes, infixes)
- Morphological complexity assessment
- Derivational analysis

## 🧠 NLP Integration

The application integrates with multiple specialized engines:

### Available Engines
- **GrammaticalParticlesEngine**: Particle identification
- **MorphologyEngine**: Morphological analysis
- **PhonologyEngine**: Phonetic processing
- **SyllabicUnitEngine**: SyllabicAnalysis
- **FrozenRootsEngine**: Root classification
- **PatternRepository**: Pattern matching

### Mock Engine Support
For testing and demonstration, the application includes mock engines that provide realistic sample data without requiring the full NLP infrastructure.

## 🎯 Use Cases

### Educational Applications
- **Arabic language learning**: Step-by-step word breakdown
- **Linguistic research**: Detailed morphophonological analysis
- **Academic study**: Pattern recognition and root analysis

### Professional Applications
- **NLP development**: Testing and validation of Arabic processing
- **Computational linguistics**: Research and analysis tool
- **Language technology**: Component integration testing

### Research Applications
- **Morphological studies**: Pattern analysis and classification
- **Phonological research**: SyllabicUnit structure investigation
- **Comparative analysis**: Cross-dialectal word comparison

## 🌐 Browser Support

- **Chrome/Chromium**: Full support with all features
- **Firefox**: Full support with animations
- **Safari**: Full support with minor limitations
- **Edge**: Full support
- **Mobile browsers**: Responsive design with touch support

## ♿ Accessibility Features

- **ARIA labels** for screen readers
- **Keyboard navigation** with tab support
- **High contrast mode** support
- **Reduced motion** preferences
- **Focus management** for better UX
- **Screen reader optimization**

## 🔧 Development

### Adding New Linguistic Levels
1. Extend the `ArabicWordTracer` class with new `_trace_*` methods
2. Add corresponding UI generation in the frontend
3. Update the level configuration in the HTML template

### Customizing Visualizations
1. Modify the `AdvancedVisualization` class in `advanced-enhancements.js`
2. Add new CSS classes in `advanced-styles.css`
3. Update the visualization generation logic

### Engine Integration
1. Add new engine import_datas in `arabic_word_tracer_app.py`
2. Update the `_initialize_engines` method
3. Create corresponding analysis methods

## 📊 Performance

- **Real-time analysis**: Typically <500ms per word
- **Concurrent requests**: Supports up to 10 simultaneous analyses
- **Memory efficient**: Lightweight mock engines for testing
- **Responsive UI**: Optimized animations and interactions

## 🐛 Troubleshooting

### Common Issues

**Import Errors**: Use `--mock-engines` flag for testing without full NLP stack
```bash
python run_tracer.py --mock-engines
```

**Port Conflicts**: Change the port number
```bash
python run_tracer.py --port 8080
```

**Permission Issues**: Run with appropriate permissions or change host
```bash
python run_tracer.py --host 127.0.0.1
```

## 📝 License

This project is part of the Arabic Morphophonological Engine suite and follows the same licensing terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add linguistic analysis features or UI enhancements
4. Test with sample Arabic words
5. Submit a pull request

## 📧 Support

For issues, questions, or feature requests, please check the project documentation or create an issue in the repository.

---

**🔍 متتبع الكلمات العربية - Bridging UI expertise with NLP innovation for comprehensive Arabic linguistic analysis.**
