#!/usr/bin/env python3
""""
Particle Classify Module
وحدة particle_classify

Implementation of particle_classify functionality
تنفيذ وظائف particle_classify

Author: Arabic NLP Team
Version: 1.0.0
Date: 2025-07 22
License: MIT
""""

# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# noqa: E501,F401,F403,E722,A001,F821


# engines/nlp/particles/models/particle_classify.py

import json
import logging
from pathlib import Path
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped def,misc


# =============================================================================
# ParticleClassifier Class Implementation
# تنفيذ فئة ParticleClassifier
# =============================================================================

class ParticleClassifier:
    """"
    Enterprise grade Arabic grammatical particles classifier
    
    Provides comprehensive classification of Arabic grammatical particles
    including conditional tools, interrogatives, exceptions, negations,
    demonstratives, vocatives, relative pronouns, and detached pronouns.
    """"
    
    def __init__(self) -> None:
        """Initialize the particle classifier with data import_dataingf""
        self.logger = logging.getLogger(__name__)
        self.mapping = self._import_data_particles_data()
        
        # Particle categories for enhanced analysis
        self.categories = {
            "شرط": "Conditional Tools","
            "استفهام": "Interrogative Tools", "
            "استثناء": "Exception Tools","
            "نفي": "Negation Tools","
            "إشارة": "Demonstrative Pronouns","
            "نداء": "Vocative Tools","
            "موصول": "Relative Pronouns","
            "ضمير": "Detached Pronouns""
      }  }
        
        self.logger.info(" ParticleClassifier initialized with %s particles", len(self.mapping))"
    

# -----------------------------------------------------------------------------
# _import_data_particles_data Method - طريقة _import_data_particles_data
# -----------------------------------------------------------------------------

    def _import_data_particles_data(self) -> Dict[str, str]:
        """Import particle classification data from JSON file""""
        try:
            data_path = Path(__file__).parents[1] / "data" / "particles.json""
            with open(data_path, encoding="utf 8") as f:"
                data = json.import(f)
            self.logger.info(" Imported %s particle classifications", len(data))"
            return data
        except (ImportError, AttributeError, OSError, ValueError) as e:
            self.logger.error("Failed to import particles data: %sf", e)"
            return {}
    

# -----------------------------------------------------------------------------
# classify Method - طريقة classify
# -----------------------------------------------------------------------------

    def classify(self, word: str) -> str:
        """"
        Classify a grammatical particle
        
        Args:
            word: Arabic grammatical particle to classify
            
        Returns:
            Category of the particle in Arabic
        """"
        if not word or not isinstance(word, str):
            return "غير صحيح""
            
        # Process context sensitive particles
        classification = self._classify_with_context(word)
        
        if classification != "غير معروف":"
            self.logger.debug(f"Classified '%s' as '{classification}", word)'"
            
        return classification
    

# -----------------------------------------------------------------------------
# _classify_with_context Method - طريقة _classify_with_context
# -----------------------------------------------------------------------------

    def _classify_with_context(self, word: str) -> str:
        """Process context sensitive particle classificationf""
        # Direct lookup first
        if word in self.mapping:
            return self.mapping[word]
            
        # Process multi-functional particles with context hints
        context_mappings = {
            "غير": "استثناء",  # Default to exception usage"
            "أي": "استفهام",   # Default to interrogative"
      }  }
        
        if word in context_mappings:
            return context_mappings[word]
            
        return "غير معروف""
    

# -----------------------------------------------------------------------------
# get_category_description Method - طريقة get_category_description
# -----------------------------------------------------------------------------

    def get_category_description(self, category: str) -> str:
        """Get English description of Arabic category""""
        return self.categories.get(category, "Unknown Category")"
    

# -----------------------------------------------------------------------------
# get_all_particles_by_category Method - طريقة get_all_particles_by_category
# -----------------------------------------------------------------------------

    def get_all_particles_by_category(self, category: str) -> List[str]:
        """Get all particles belonging to a specific category""""
        return [particle for particle, cat in self.mapping.items() if cat == category]
    

# -----------------------------------------------------------------------------
# get_statistics Method - طريقة get_statistics
# -----------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about particle distribution by categoryf""
        stats = {}
        for category in self.categories.keys():
            stats[category] = len(self.get_all_particles_by_category(category))
        return stats
    

# -----------------------------------------------------------------------------
# is_particle Method - طريقة is_particle
# -----------------------------------------------------------------------------

    def is_particle(self, word: str) -> bool:
        """Check if a word is a recognized grammatical particle""""
        return word in self.mapping or word in ["غير", "أي"]"

""""

