# 🎯 Arabic Phonology Engine - Full Stack Web Application

A comprehensive web-based Arabic phonological analysis system that provides real-time text normalization, phoneme analysis, and syllable encoding.

## 🌟 Features

- **🔄 Text Normalization**: Handles Arabic diacritics, tatweel removal, and alif variants
- **🔍 Phoneme Analysis**: Detailed phonetic classification (place, manner, voicing, emphatic)
- **📊 Syllable Encoding**: Advanced syllable structure analysis with CV patterns
- **🏷️ Morphological Classification**: Letter classification (core, extra, functional, weak)
- **🌐 Web Interface**: Modern, responsive Arabic-RTL interface
- **🚀 Real-time Processing**: Instant analysis with beautiful visualizations

## 🏗️ Architecture

### Backend (Flask)
- **Framework**: Flask with CORS support
- **API Endpoints**: RESTful JSON API
- **Processing**: Real-time phonological analysis
- **Error Handling**: Comprehensive error management

### Frontend (Modern Web)
- **UI Framework**: Bootstrap 5 with custom RTL styling
- **JavaScript**: Vanilla JS with modern ES6+ features
- **Styling**: Custom CSS with Arabic typography support
- **UX**: Responsive design with smooth animations

## 📁 Project Structure

```
new engine/
├── app.py                          # Flask backend server
├── requirements.txt                # Python dependencies
├── data/
│   └── phoneme_db.py              # Arabic phoneme database
├── phonology/
│   ├── analyzer.py                # Core phoneme analyzer
│   ├── normalizer.py              # Text normalization
│   ├── classifier.py              # Morphological classification
│   └── syllable_encoder.py       # Syllable encoding
├── templates/
│   └── index.html                 # Main web interface
└── static/
    ├── css/
    │   └── style.css              # Custom styles
    └── js/
        └── app.js                 # Frontend JavaScript
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Run the Flask development server
python app.py
```

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

## 🔧 API Endpoints

### Health Check
```http
GET /api/health
```

### Text Analysis
```http
POST /api/analyze
Content-Type: application/json

{
    "text": "أَكَلَ الوَلَدُ التُّفاحَ"
}
```

**Response:**
```json
{
    "original_text": "أَكَلَ الوَلَدُ التُّفاحَ",
    "normalized_text": "اكل الولد التفاح",
    "phoneme_analysis": [...],
    "syllable_encoding": [...],
    "status": "success"
}
```

## 📊 Analysis Features

### Phoneme Analysis
- **Place of Articulation**: labial, dental, alveolar, etc.
- **Manner of Articulation**: plosive, fricative, nasal, etc.
- **Voicing**: voiced/voiceless
- **Emphasis**: emphatic/non-emphatic
- **Acoustic Weight**: numerical weight for processing

### Syllable Encoding
- **CV Patterns**: Consonant-Vowel structure analysis
- **Diacritic Handling**: Shadda, sukoon, tanween processing
- **Long Vowels**: Madd detection and encoding
- **Morphological Integration**: Combined with letter classification

### Text Normalization
- **Hamza Unification**: أ، إ، آ → ا
- **Alif Variants**: ى، ئ، ؤ normalization
- **Diacritic Removal**: Configurable diacritic handling
- **Tatweel Removal**: ـ character elimination

## 🎨 Web Interface Features

### Arabic-First Design
- **RTL Layout**: Proper right-to-left text flow
- **Arabic Typography**: Amiri font integration
- **Cultural Colors**: Arabic-inspired color scheme
- **Responsive Design**: Mobile-friendly interface

### User Experience
- **Real-time Analysis**: Instant processing feedback
- **Example Texts**: Quick-start example buttons
- **Error Handling**: User-friendly error messages
- **Loading States**: Visual processing indicators

### Visualization
- **Phoneme Cards**: Individual character analysis cards
- **Syllable Tables**: Structured encoding results
- **Classification Badges**: Color-coded morphological types
- **Smooth Animations**: CSS transitions and effects

## 🧪 Testing Examples

Try these examples in the web interface:

```arabic
أَكَلَ الوَلَدُ التُّفاحَ
السَّلامُ عَلَيْكُم
مَرْحَباً بِالعالَم
نَحْنُ نَتَعَلَّمُ العَرَبِيَّة
```

## 🔍 Technical Details

### Backend Technologies
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **Python 3.8+**: Core language

### Frontend Technologies
- **HTML5**: Semantic markup
- **CSS3**: Advanced styling with Grid/Flexbox
- **JavaScript ES6+**: Modern browser features
- **Bootstrap 5**: UI framework
- **Font Awesome**: Icon library

### Performance
- **Lightweight**: Minimal dependencies
- **Fast Processing**: Optimized algorithms
- **Caching**: Smart result caching
- **Scalable**: Designed for high traffic

## 🛠️ Development

### Adding New Features

1. **Backend**: Extend the analysis modules in `phonology/`
2. **API**: Add new endpoints in `app.py`
3. **Frontend**: Update `static/js/app.js` for new UI features
4. **Styling**: Customize `static/css/style.css`

### Customization

- **Phoneme Database**: Modify `data/phoneme_db.py`
- **Classification Rules**: Update `phonology/classifier.py`
- **UI Theme**: Customize CSS variables
- **API Responses**: Extend response formats

## 📈 Future Enhancements

- [ ] **Multi-language Support**: English interface option
- [ ] **Advanced Visualizations**: Interactive phoneme charts
- [ ] **Export Features**: PDF/CSV export capabilities
- [ ] **Batch Processing**: Multiple text analysis
- [ ] **User Accounts**: Save and manage analyses
- [ ] **API Rate Limiting**: Production-ready API
- [ ] **Mobile App**: Native mobile application
- [ ] **Audio Integration**: Text-to-speech capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is open source and available under the MIT License.

---

**🎯 Arabic Phonology Engine** - Making Arabic linguistic analysis accessible through modern web technology.
