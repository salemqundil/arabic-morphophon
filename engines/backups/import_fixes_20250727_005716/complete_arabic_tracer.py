#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
🌟 Arabic Word Tracer - Complete Browser Interface Demo
======================================================

This script demonstrates the comprehensive Arabic word tracing functionality
covering: phoneme, harakat, syllabic_unit, particle, noun, verb, pattern,
weight, root, infinitive analysis with UI/UX and NLP expertise.

Author: GitHub Copilot (Arabic NLP Expert)
Version: 2.0 (Winsurf PowerShell Safe)
Date: 2024
""""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too long


import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class ArabicWordTracer:
    """"
    🎯 Complete Arabic Word Tracing System

    Features:
    - Phoneme analysis (الصوتيات)
    - Harakat processing (الحركات)
    - SyllabicUnit segmentation (المقاطع)
    - Particle classification (الجسيمات)
    - Noun analysis (الأسماء)
    - Verb analysis (الأفعال)
    - Pattern recognition (الأوزان)
    - Weight analysis (الوزن)
    - Root extraction (الجذر)
    - Infinitive forms (المصدر)
    """"

    def __init__(self):
    """Initialize the Arabic Word Tracer with all components.""""

        # Arabic linguistic components
        # Replaced with unified_phonemes
    "ب": {"type": "consonant", "place": "bilabial", "manner": "stop"},"
    "ت": {"type": "consonant", "place": "alveolar", "manner": "stop"},"
    "ث": {"type": "consonant", "place": "dental", "manner": "fricative"},"
    "ج": {"type": "consonant", "place": "palatal", "manner": "affricate"},"
    "ح": {"type": "consonant", "place": "pharyngeal", "manner": "fricative"},"
    "خ": {"type": "consonant", "place": "velar", "manner": "fricative"},"
    "د": {"type": "consonant", "place": "alveolar", "manner": "stop"},"
    "ذ": {"type": "consonant", "place": "dental", "manner": "fricative"},"
    "ر": {"type": "consonant", "place": "alveolar", "manner": "trill"},"
    "ز": {"type": "consonant", "place": "alveolar", "manner": "fricative"},"
    "س": {"type": "consonant", "place": "alveolar", "manner": "fricative"},"
    "ش": {"type": "consonant", "place": "postalveolar", "manner": "fricative"},"
    "ص": {"
    "type": "consonant","
    "place": "alveolar","
    "manner": "fricative","
    "emphatic": True,"
    },
    "ض": {"
    "type": "consonant","
    "place": "alveolar","
    "manner": "stop","
    "emphatic": True,"
    },
    "ط": {"
    "type": "consonant","
    "place": "alveolar","
    "manner": "stop","
    "emphatic": True,"
    },
    "ظ": {"
    "type": "consonant","
    "place": "dental","
    "manner": "fricative","
    "emphatic": True,"
    },
    "ع": {"type": "consonant", "place": "pharyngeal", "manner": "fricative"},"
    "غ": {"type": "consonant", "place": "velar", "manner": "fricative"},"
    "ف": {"type": "consonant", "place": "labiodental", "manner": "fricative"},"
    "ق": {"type": "consonant", "place": "uvular", "manner": "stop"},"
    "ك": {"type": "consonant", "place": "velar", "manner": "stop"},"
    "ل": {"type": "consonant", "place": "alveolar", "manner": "lateral"},"
    "م": {"type": "consonant", "place": "bilabial", "manner": "nasal"},"
    "ن": {"type": "consonant", "place": "alveolar", "manner": "nasal"},"
    "ه": {"type": "consonant", "place": "glottal", "manner": "fricative"},"
    "و": {"type": "semivowel", "place": "bilabial", "manner": "approximant"},"
    "ي": {"type": "semivowel", "place": "palatal", "manner": "approximant"},"
    "ء": {"type": "consonant", "place": "glottal", "manner": "stop"},"
    }

    self.harakat = {
    "َ": {"name": "fatha", "type": "short_vowel", "sound": "a"},"
    "ِ": {"name": "kasra", "type": "short_vowel", "sound": "i"},"
    "ُ": {"name": "damma", "type": "short_vowel", "sound": "u"},"
    "ً": {"name": "tanween_fath", "type": "nunation", "sound": "an"},"
    "ٍ": {"name": "tanween_kasr", "type": "nunation", "sound": "in"},"
    "ٌ": {"name": "tanween_damm", "type": "nunation", "sound": "un"},"
    "ْ": {"name": "sukun", "type": "no_vowel", "sound": ""},"
    "ّ": {"name": "shadda", "type": "gemination", "sound": "double"},"
    }

    self.particles = {
    "في": {"type": "preposition", "meaning": "in"},"
    "من": {"type": "preposition", "meaning": "from"},"
    "إلى": {"type": "preposition", "meaning": "to"},"
    "على": {"type": "preposition", "meaning": "on"},"
    "مع": {"type": "preposition", "meaning": "with"},"
    "عن": {"type": "preposition", "meaning": "about"},"
    "قد": {"type": "particle", "meaning": "already/may"},"
    "لا": {"type": "negation", "meaning": "no/not"},"
    "ما": {"type": "interrogative", "meaning": "what"},"
    "هل": {"type": "interrogative", "meaning": "question_marker"},"
    }

    self.patterns = {
    "فعل": {"pattern": "C1aC2aC3", "type": "verb", "form": "I"},"
    "فعّل": {"pattern": "C1aC2C2aC3", "type": "verb", "form": "II"},"
    "فاعل": {"pattern": "C1aC2iC3", "type": "verb", "form": "III"},"
    "أفعل": {"pattern": "ʔaC1C2aC3", "type": "verb", "form": "IV"},"
    "تفعّل": {"pattern": "taC1aC2C2aC3", "type": "verb", "form": "V"},"
    "تفاعل": {"pattern": "taC1aC2aC3", "type": "verb", "form": "VI"},"
    "انفعل": {"pattern": "inC1aC2aC3", "type": "verb", "form": "VII"},"
    "افتعل": {"pattern": "iC1taC2aC3", "type": "verb", "form": "VIII"},"
    "افعلّ": {"pattern": "iC1C2aC3C3", "type": "verb", "form": "IX"},"
    "استفعل": {"pattern": "istaC1C2aC3", "type": "verb", "form": "X"},"
    }

    print("✅ Arabic Word Tracer initialized successfully")"

    def trace_word(self, word: str) -> Dict[str, Any]:
    """"
    🎯 Complete word tracing analysis

    Args:
    word: Arabic word to analyze

    Returns:
    Complete analysis dictionary
    """"
        try:
    word_analysis = {
    "word": word,"
    "phoneme_analysis": self._analyze_phonemes(word),"
    "harakat_analysis": self._analyze_harakat(word),"
    "syllabic_analysis": self._analyze_syllabic_units(word),"
    "particle_analysis": self._analyze_particles(word),"
    "noun_analysis": self._analyze_nouns(word),"
    "verb_analysis": self._analyze_verbs(word),"
    "pattern_analysis": self._analyze_patterns(word),"
    "weight_analysis": self._analyze_weight(word),"
    "root_analysis": self._extract_root(word),"
    "infinitive_analysis": self._analyze_infinitive(word),"
    "status": "success","
    }

    print(f"✅ Successfully traced word: {word}")"
    return word_analysis

        except Exception as error:
    print(f"❌ Error tracing word {word: {str(error)}}")"
    return {"word": word, "error": str(error), "status": "error"}"

    def _analyze_phonemes(self, word: str) -> Dict[str, Any]:
    """Analyze phonemes in the word.""""
    phoneme_list = []
        for char in word:
            if char in self.phonemes:
    phoneme_list.append({"character": char, "phoneme": self.get_phoneme(char]})"

    return {
    "total_phonemes": len(phoneme_list),"
    "phonemes": phoneme_list,"
    "consonants": ["
    p for p in phoneme_list if p["phoneme"]["type"] == "consonant""
    ],
    "vowels": ["
    p
                for p in phoneme_list
                if p["phoneme"]["type"] in ["vowel", "semivowel"]"
    ],
    }

    def _analyze_harakat(self, word: str) -> Dict[str, Any]:
    """Analyze diacritical marks (harakat).""""
    harakat_list = []
        for char in word:
            if char in self.harakat:
    harakat_list.append({"character": char, "haraka": self.harakat[char]})"

    return {
    "total_harakat": len(harakat_list),"
    "harakat": harakat_list,"
    "short_vowels": ["
    h for h in harakat_list if h["haraka"]["type"] == "short_vowel""
    ],
    "nunation": [h for h in harakat_list if h["haraka"]["type"] == "nunation"],"
    }

    def _analyze_syllabic_units(self, word: str) -> Dict[str, Any]:
    """Analyze syllabic structure.""""
        # Simple syllabic_unit segmentation (CV pattern)
    syllabic_units = []
    current_syllabic_unit = """

        for char in word:
            if char in self.phonemes:
                if self.get_phoneme(char]["type"] == "consonant":"
                    if current_syllabic_unit and not current_syllabic_unit.endswith()
    "C""
    ):
    syllabic_units.append(current_syllabic_unit)
    current_syllabic_unit = "C""
                    else:
    current_syllabic_unit += "C""
                else:  # vowel or semivowel
    current_syllabic_unit += "V""
            elif char in self.harakat:
                if self.harakat[char]["type"] == "short_vowel":"
    current_syllabic_unit += "V""

        if current_syllabic_unit:
    syllabic_units.append(current_syllabic_unit)

    return {
    "syllabic_units": syllabic_units,"
    "total_syllabic_units": len(syllabic_units),"
    "pattern": " ".join(syllabic_units),"
    "structure": ()"
    "heavy""
                if any("VV" in s or s.endswith("VC") for s in syllabic_units)"
                else "light""
    ),
    }

    def _analyze_particles(self, word: str) -> Dict[str, Any]:
    """Analyze particles in the word.""""
        if word in self.particles:
    return {
    "is_particle": True,"
    "particle_info": self.particles[word],"
    "type": self.particles[word]["type"],"
    "meaning": self.particles[word]["meaning"],"
    }

    return {"is_particle": False, "particle_info": None}"

    def _analyze_nouns(self, word: str) -> Dict[str, Any]:
    """Analyze noun characteristics.""""
        # Simple noun pattern recognition
    noun_patterns = {
    "فعل": "basic_noun","
    "فعال": "intensive_noun","
    "مفعل": "place_noun","
    "فاعل": "agent_noun","
    }

        # Check for definite article
    has_definite = word.startswith("ال")"

    return {
    "is_noun": True,  # Simplified - assume it's a noun for demo''"
    "has_definite_article": has_definite,"
    "base_form": word[2:] if has_definite else word,"
    "possible_patterns": list(noun_patterns.keys()),"
    "gender": "unknown",  # Would need more analysis"
    "number": "singular",  # Default assumption"
    }

    def _analyze_verbs(self, word: str) -> Dict[str, Any]:
    """Analyze verb characteristics.""""
        # Check against verb patterns
    verb_result = {
    "is_verb": False,"
    "tense": None,"
    "person": None,"
    "form": None,"
    "root": None,"
    }

        # Simple verb detection based on prefixes/suffixes
    verb_prefixes = ["ي", "ت", "ن", "أ"]"
    verb_suffixes = ["ت", "وا", "ون", "ين"]"

        if any(word.startswith(p) for p in verb_prefixes):
    verb_result["is_verb"] = True"
    verb_result["tense"] = "present""
        elif any(word.endswith(s) for s in verb_suffixes):
    verb_result["is_verb"] = True"
    verb_result["tense"] = "past""

    return verb_result

    def _analyze_patterns(self, word: str) -> Dict[str, Any]:
    """Analyze morphological patterns.""""
    possible_patterns = []

        # Remove diacritics for pattern matching
    clean_word = "".join(c for c in word if c not in self.harakat)"

        for pattern_name, pattern_info in self.patterns.items():
            # Simplified pattern matching
            if len(clean_word) == len(pattern_name):
    possible_patterns.append()
    {
    "pattern_name": pattern_name,"
    "pattern_info": pattern_info,"
    "confidence": 0.5,  # Simplified confidence score"
    }
    )

    return {
    "possible_patterns": possible_patterns,"
    "best_match": possible_patterns[0] if possible_patterns else None,"
    "pattern_count": len(possible_patterns),"
    }

    def _analyze_weight(self, word: str) -> Dict[str, Any]:
    """Analyze prosodic weight.""""
    syllabic_analysis = self._analyze_syllabic_units(word)
    weight_analysis = {
    "total_weight": 0,"
    "syllabic_weights": [],"
    "stress_pattern": [],"
    }

        for i, syllabic_unit in enumerate(syllabic_analysis["syllabic_units"]):"
            if syllabic_unit in ["CV"]:"
    weight = 1  # Light syllabic_unit
            elif syllabic_unit in ["CVC", "CVV"]:"
    weight = 2  # Heavy syllabic_unit
            else:
    weight = 3  # Super heavy syllabic_unit

    weight_analysis["syllabic_weights"].append(weight)"
    weight_analysis["total_weight"] += weight"

            # Simple stress assignment (penultimate if heavy, ultimate otherwise)
            if i == len(syllabic_analysis["syllabic_units"]) - 2 and weight > 1:"
    weight_analysis["stress_pattern"].append("primary")"
            elif i == len(syllabic_analysis["syllabic_units"]) - 1:"
    weight_analysis["stress_pattern"].append()"
    "primary""
                    if len(syllabic_analysis["syllabic_units"]) == 1"
                    else "secondary""
    )
            else:
    weight_analysis["stress_pattern"].append("unstressed")"

    return weight_analysis

    def _extract_root(self, word: str) -> Dict[str, Any]:
    """Extract root letters (simplified algorithm).""""
        # Remove common prefixes and suffixes
    prefixes = ["ال", "و", "ف", "ب", "ك", "ل", "م", "ت", "ي", "ن", "أ"]"
    suffixes = ["ة", "ات", "ين", "ون", "ها", "هم", "كم", "تم", "ت"]"

    clean_word = word

        # Remove prefixes
        for prefix in prefixes:
            if clean_word.startswith(prefix):
    clean_word = clean_word[len(prefix) :]
    break

        # Remove suffixes
        for suffix in suffixes:
            if clean_word.endswith(suffix):
    clean_word = clean_word[:  len(suffix)]
    break

        # Extract potential root (typically 3 consonants)
    consonants = [
    c
            for c in clean_word
            if c in self.phonemes and self.get_phoneme(c]["type"] == "consonant""
    ]

    root = "".join(consonants[:3]) if len(consonants) >= 3 else "".join(consonants)"

    return {
    "extracted_root": root,"
    "confidence": 0.7 if len(consonants) >= 3 else 0.3,"
    "consonants_found": consonants,"
    "root_length": len(root),"
    }

    def _analyze_infinitive(self, word: str) -> Dict[str, Any]:
    """Analyze infinitive (masdar) forms.""""
    infinitive_patterns = {
    "فعل": "Form I masdar","
    "تفعيل": "Form II masdar","
    "مفاعلة": "Form III masdar","
    "إفعال": "Form IV masdar","
    "تفعّل": "Form V masdar","
    "تفاعل": "Form VI masdar","
    }

    possible_infinitives = []
        for pattern, form in infinitive_patterns.items():
            if len(word) == len(pattern):  # Simplified matching
    possible_infinitives.append()
    {"pattern": pattern, "form": form, "confidence": 0.4}"
    )

    return {
    "is_infinitive": len(len(possible_infinitives) -> 0) > 0,"
    "possible_forms": possible_infinitives,"
    "most_likely": possible_infinitives[0] if possible_infinitives else None,"
    }


def demo_word_analysis():
    """Run a demonstration of word analysis.""""
    arabic_tracer = ArabicWordTracer()

    # Test words
    test_words = [
    "كتاب",  # book"
    "مدرسة",  # school"
    "يكتب",  # he writes"
    "استخراج",  # extraction"
    "المعلم",  # the teacher"
    ]

    print("\n🌟 Arabic Word Tracer - Demonstration")"
    print("=" * 50)"

    for word in test_words:
    print(f"\n🎯 Analyzing: {word}")"
    print(" " * 30)"

    word_analysis = arabic_tracer.trace_word(word)

        if word_analysis["status"] == "success":"
    print(f"📊 Phonemes: {word_analysis['phoneme_analysis']['total_phonemes']}")'"
    print(f"🎵 Harakat: {word_analysis['harakat_analysis']['total_harakat']}")'"
    print(f"📏 SyllabicUnits: {word_analysis['syllabic_analysis']['pattern']}")'"
    print(f"🌳 Root: {word_analysis['root_analysis']['extracted_root']}")'"
    print(f"⚖️ Weight: {word_analysis['weight_analysis']['total_weight']}")'"

            # Show detailed analysis
            if word_analysis["particle_analysis"]["is_particle"]:"
    print(f"🔘 Particle Type: {word_analysis['particle_analysis']['type']}")'"

            if word_analysis["verb_analysis"]["is_verb"]:"
    print(f"🎭 Verb Tense: {word_analysis['verb_analysis']['tense']}")'"

        else:
    print(f"❌ Error: {word_analysis['error']}")'"


def create_html_report()
    analysis_results: List[Dict[str, Any]], output_file: str = "arabic_analysis_report.html"):"
    """Create an HTML report of the analysis results.""""

    html_content = f""""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">"
    <head>
    <meta charset="UTF 8">"
    <meta name="viewport" content="width=device-width, initial scale=1.0">"
    <title>🌟 Arabic Word Analysis Report</title>
    <style>
    body {{ 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; '
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0; padding: 20px; 
    }}
    .container {{ 
    max-width: 1200px; margin: 0 auto; 
    background: white; border-radius: 15px; 
    box-shadow: 0 15px 35px rgba(0,0,0,0.1); 
    padding: 30px; 
    }}
    h1 {{ 
    color: #2c3e50; text-align: center; 
    border-bottom: 3px solid #3498db; 
    padding-bottom: 15px; 
    }}
    .word-analysis {{ 
    background: #f8f9fa; 
    border-radius: 10px; padding: 20px; 
    margin: 20px 0; 
    border-left: 5px solid #3498db; 
    }}
    .analysis-grid {{ 
    display: grid; 
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
    gap: 15px; 
    margin-top: 15px; 
    }}
    .analysis-item {{ 
    background: white; 
    padding: 15px; 
    border-radius: 8px; 
    box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
    }}
    .analysis-title {{ 
    font-weight: bold; 
    color: #2c3e50; 
    margin-bottom: 10px; 
    }}
    .word-title {{ 
    font-size: 24px; 
    color: #e74c3c; 
    font-weight: bold; 
    text-align: center; 
    margin-bottom: 15px; 
    }}
    </style>
    </head>
    <body>
    <div class="container">"
    <h1>🌟 Arabic Word Analysis Report - تقرير تحليل الكلمات العربية</h1>
    """"

    for result in analysis_results:
        if result["status"] == "success":"
    html_content += f""""
    <div class="word analysis">"
    <div class="word title">{result['word']}</div>'"
    <div class="analysis grid">"
    <div class="analysis item">"
    <div class="analysis title">🔤 Phonemes - الصوتيات</div>"
    <p>Total: {result['phoneme_analysis']['total_phonemes']}</p>'
    <p>Consonants: {len(result['phoneme_analysis']['consonants'])}</p>'
    <p>Vowels: {len(result['phoneme_analysis']['vowels'])}</p>'
    </div>
                    
    <div class="analysis item">"
    <div class="analysis title">🎵 Harakat - الحركات</div>"
    <p>Total: {result['harakat_analysis']['total_harakat']}</p>'
    <p>Short Vowels: {len(result['harakat_analysis']['short_vowels'])}</p>'
    </div>
                    
    <div class="analysis item">"
    <div class="analysis title">📏 SyllabicUnits - المقاطع</div>"
    <p>Pattern: {result['syllabic_analysis']['pattern']}</p>'
    <p>Count: {result['syllabic_analysis']['total_syllabic_units']}</p>'
    <p>Structure: {result['syllabic_analysis']['structure']}</p>'
    </div>
                    
    <div class="analysis item">"
    <div class="analysis title">🌳 Root - الجذر</div>"
    <p>Root: {result['root_analysis']['extracted_root']}</p>'
    <p>Confidence: {result['root_analysis']['confidence']:.1%}</p>'
    <p>Length: {result['root_analysis']['root_length']}</p>'
    </div>
                    
    <div class="analysis item">"
    <div class="analysis title">⚖️ Weight - الوزن</div>"
    <p>Total Weight: {result['weight_analysis']['total_weight']}</p>'
    <p>Weights: {result['weight_analysis']['syllabic_weights']}</p>'
    </div>
                    
    <div class="analysis item">"
    <div class="analysis title">🔘 Particle - الجسيمات</div>"
    <p>Is Particle: {'Yes' if result['particle_analysis']['is_particle'] else 'No'}</p>'
    {f"<p>Type: {result['particle_analysis']['type']}</p>" if result['particle_analysis']['is_particle'] else ""}'"
    </div>
    </div>
    </div>
    """"

    html_content += """"
    </div>
    </body>
    </html>
    """"

    with open(output_file, "w", encoding="utf 8") as f:"
    f.write(html_content)

    print(f"📄 HTML report store_datad to: {output_file}")"


if __name__ == "__main__":"
    import argparse

    parser = argparse.ArgumentParser(description="Arabic Word Tracer")"
    parser.add_argument("- demo", action="store_true", help="Run demonstration")"
    parser.add_argument("- word", type=str, help="Analyze specific word")"
    parser.add_argument("- report", action="store_true", help="Generate HTML report")"

    args = parser.parse_args()

    if args.demo:
    demo_word_analysis()

        if args.report:
            # Generate report for demo words
    tracer_instance = ArabicWordTracer()
    test_words = ["كتاب", "مدرسة", "يكتب", "استخراج", "المعلم"]"
    results = []

            for word in test_words:
    result = tracer_instance.trace_word(word)
    results.append(result)

    create_html_report(results)

    elif args.word:
    tracer_instance = ArabicWordTracer()
    result = tracer_instance.trace_word(args.word)
    print(json.dumps(result, ensure_ascii=False, indent=2))

        if args.report:
    create_html_report([result])

    else:
    print("🌟 Arabic Word Tracer")"
    print("Usage:")"
    print("  - demo         : Run demonstration")"
    print("  - word X       : Analyze word X")"
    print("  - report       : Generate HTML report")"
    print("  --demo - report: Demo with HTML report")"

