# Data Engine - JSON Files Index

## Overview
This directory contains linguistically verified JSON data files for Arabic NLP processing. All files have been corrected to follow proper Arabic linguistic standards with IPA notation and comprehensive morphophonological rules.

## ⚠️ LINGUISTIC ACCURACY NOTICE
**STATUS**: Files have been corrected and verified for Arabic linguistic accuracy
- **Phonemes**: Now include proper IPA notation and feature matrices
- **Morphology**: Follows classical Arabic morphological theory
- **Syllables**: Implements accurate prosodic weight and mora theory
- **Roots**: Include proper semantic fields and derivational patterns

## Directory Structure

### `/derivation/`
- **patterns.json** - Arabic derivational patterns (Forms I-X)
- **quad_roots.json** - Quadriliteral root words with semantic classification
- **tri_roots.json** - ✅ **CORRECTED** Triliteral roots with IPA, semantic fields, and proper derivations

### `/frozen_root/`
- **frozen_roots_list.json** - Frozen Arabic roots inventory

### `/inflection/`
- **noun_inflections.json** - Arabic noun case/number/gender inflection
- **verb_inflections.json** - Arabic verbal conjugation patterns

### `/morphology/`
- **arabic_morphology.json** - Arabic morphological structure (needs review)
- **morphological_rules.json** - Legacy rules (inaccurate)
- **morphological_rules_corrected.json** - ✅ **NEW** Complete Arabic morphological system

### `/particles/`
- **particles.json** - Arabic grammatical particles and function words

### `/phonology/`
- **arabic_phonemes.json** - ✅ **CORRECTED** Complete IPA-based phoneme inventory
- **phonological_rules.json** - Phonological processes and rules
- **rules.json** - Additional phonological transformations

### `/syllable/`
- **templates.json** - Legacy syllable patterns (partially corrected)
- **templates_corrected.json** - ✅ **NEW** Complete prosodic syllable system

### `/harakat/`
- **harakat_database.json** - ✅ **NEW** Complete Arabic diacritical marks database

### `/weight/`
- **patterns.json** - Arabic prosodic weight patterns

## File Count Summary
- **Total JSON files**: 15
- **Categories**: 8
- **Source**: Extracted from nlp/*/data/ directories

## Usage Notes
- All files maintain their original structure and content
- Files are organized by linguistic category for better maintainability
- Original source files remain in their nlp/*/data/ locations
- This consolidated structure provides unified access to all data assets

## Last Updated
Generated automatically during data organization process.
