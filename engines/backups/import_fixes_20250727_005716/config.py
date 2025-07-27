#!/usr/bin/env python3
"""
Config Module
وحدة config

Implementation of config functionality
تنفيذ وظائف config

Author: Arabic NLP Team
Version: 1.0.0
Date: 2025-07 22
License: MIT
"""

# Global suppressions for WinSurf IDE
# pylint: disable=broad-except,unused-variable,unused-argument,too-many-arguments
# pylint: disable=invalid-name,too-few-public-methods,missing-docstring
# pylint: disable=too-many-locals,too-many-branches,too-many statements
# noqa: E501,F401,F403,E722,A001,F821


"""
 إعدادات المحركات - Engine Configurations
==========================================

نماذج Pydantic لإعدادات جميع المحركات
"""

from pydantic import BaseModel, Field


# =============================================================================
# BaseEngineConfig Class Implementation
# تنفيذ فئة BaseEngineConfig
# =============================================================================


class BaseEngineConfig(BaseModel):
    """إعدادات المحرك الأساسي"""

    cache_enabled: bool = True
    shared_db: bool = False
    max_cache_size: int = 1000  # noqa: A001
    timeout_seconds: int = 30  # noqa: A001
    log_level: str = "INFO"  # noqa: A001


# =============================================================================
# PhonologyEngineConfig Class Implementation
# تنفيذ فئة PhonologyEngineConfig
# =============================================================================


class PhonologyEngineConfig(BaseEngineConfig):
    """إعدادات محرك التحليل الصوتي"""

    rules_file: str = "engines/nlp/phonology/data/phonology_rules.json"  # noqa: A001
    phoneme_patterns_file: str = (
    "engines/nlp/phonology/data/phoneme_patterns.json"  # noqa: A001
    )
    syllabic_unit_templates_file: str = (
    "engines/nlp/phonology/data/syllabic_unit_templates.json"  # noqa: A001
    )
    normalization_rules: str = (
    "engines/nlp/phonology/data/normalization.json"  # noqa: A001
    )


# =============================================================================
# MorphologyEngineConfig Class Implementation
# تنفيذ فئة MorphologyEngineConfig
# =============================================================================


class MorphologyEngineConfig(BaseEngineConfig):
    """إعدادات محرك التحليل الصرفي"""

    root_database_file: str = (
    "engines/nlp/morphology/data/roots_database.json"  # noqa: A001
    )
    pattern_templates_file: str = (
    "engines/nlp/morphology/data/pattern_templates.json"  # noqa: A001
    )
    affix_mappings_file: str = (
    "engines/nlp/morphology/data/affix_mappings.json"  # noqa: A001
    )
    feature_vectors_file: str = (
    "engines/nlp/morphology/data/feature_vectors.json"  # noqa: A001
    )
    segmentation_patterns_file: str = (
    "engines/nlp/morphology/data/segmentation_patterns.json"  # noqa: A001
    )


# =============================================================================
# WeightEngineConfig Class Implementation
# تنفيذ فئة WeightEngineConfig
# =============================================================================


class WeightEngineConfig(BaseEngineConfig):
    """إعدادات محرك الأوزان الصرفية"""

    patterns_file: str = "engines/nlp/weight/data/patterns.json"  # noqa: A001
    weight_templates_file: str = (
    "engines/nlp/weight/data/weight_templates.json"  # noqa: A001
    )
    morphological_classes_file: str = (
    "engines/nlp/weight/data/morphological_classes.json"  # noqa: A001
    )


# =============================================================================
# FrozenRootsEngineConfig Class Implementation
# تنفيذ فئة FrozenRootsEngineConfig
# =============================================================================


class FrozenRootsEngineConfig(BaseEngineConfig):
    """إعدادات محرك الجذور الجامدة"""

    frozen_list_file: str = (
    "engines/nlp/frozen_root/data/frozen_roots.json"  # noqa: A001
    )
    exceptions_file: str = "engines/nlp/frozen_root/data/exceptions.json"  # noqa: A001
    classification_rules_file: str = (
    "engines/nlp/frozen_root/data/classification_rules.json"  # noqa: A001
    )


# =============================================================================
# GrammaticalParticlesEngineConfig Class Implementation
# تنفيذ فئة GrammaticalParticlesEngineConfig
# =============================================================================


class GrammaticalParticlesEngineConfig(BaseEngineConfig):
    """إعدادات محرك الجسيمات النحوية"""

    particles_database_file: str = (
    "engines/nlp/grammatical_particles/data/particles.json"  # noqa: A001
    )
    categories_file: str = (
    "engines/nlp/grammatical_particles/data/categories.json"  # noqa: A001
    )
    rules_file: str = "engines/nlp/grammatical_particles/data/rules.json"  # noqa: A001


# =============================================================================
# FullPipelineEngineConfig Class Implementation
# تنفيذ فئة FullPipelineEngineConfig
# =============================================================================


class FullPipelineEngineConfig(BaseModel):
    """إعدادات المحرك الشامل"""

    phonology: PhonologyEngineConfig = Field(default_factory=PhonologyEngineConfig)
    morphology: MorphologyEngineConfig = Field(default_factory=MorphologyEngineConfig)
    weight: WeightEngineConfig = Field(default_factory=WeightEngineConfig)
    frozen_root: FrozenRootsEngineConfig = Field(
        default_factory=FrozenRootsEngineConfig
    )
    grammatical_particles: GrammaticalParticlesEngineConfig = Field(
        default_factory=GrammaticalParticlesEngineConfig
    )

    # إعدادات المعالجة العامة
    parallel_processing: bool = True
    max_workers: int = 4  # noqa: A001
    timeout_seconds: int = 60  # noqa: A001
    enable_caching: bool = True
    detailed_analysis: bool = True
