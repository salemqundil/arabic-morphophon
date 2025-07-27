# 🔁 CIRCULAR IMPORT DEPENDENCY ANALYSIS

**Generated:** 2025-07-27T01:41:44.989301
**Files Analyzed:** 575
**Modules Found:** 575

## 📊 SUMMARY

- **Total Python Files:** 575
- **Local Modules:** 575
- **Circular Import Chains:** 0
- **High Coupling Modules:** 6

## 🔴 CIRCULAR IMPORT CHAINS

✅ **No circular imports detected!**

## 📈 MODULE COMPLEXITY ANALYSIS

| Module | Imports | Imported By | Total Coupling | Instability |
|--------|---------|-------------|----------------|-------------|
| `arabic_inflection_corrected` | 0 | 36 | 36 | 0.00 |
| `fix_logging_config` | 0 | 32 | 32 | 0.00 |
| `advanced_ast_syntax_fixer` | 0 | 21 | 21 | 0.00 |
| `advanced_arabic_phonology_system` | 0 | 15 | 15 | 0.00 |
| `phonology_core_unified` | 0 | 6 | 6 | 0.00 |
| `core` | 0 | 6 | 6 | 0.00 |
| `unified_phonemes` | 0 | 4 | 4 | 0.00 |
| `advanced_syntax_fixer` | 3 | 0 | 3 | 1.00 |
| `advanced_syntax_validator` | 3 | 0 | 3 | 1.00 |
| `controlled_string_fixer` | 3 | 0 | 3 | 1.00 |
| `final_comprehensive_syntax_fixer` | 3 | 0 | 3 | 1.00 |
| `precise_string_fixer` | 3 | 0 | 3 | 1.00 |
| `step2_indentation_fixer` | 3 | 0 | 3 | 1.00 |
| `surgical_indentation_fixer` | 3 | 0 | 3 | 1.00 |
| `ultra_precise_syntax_fixer` | 3 | 0 | 3 | 1.00 |
| `version_alignment_toolkit` | 3 | 0 | 3 | 1.00 |
| `backups.import_fixes_20250727_005716.arabic_pronouns_generator` | 3 | 0 | 3 | 1.00 |
| `backups.import_fixes_20250727_005716.arabic_relative_pronouns_analyzer` | 3 | 0 | 3 | 1.00 |
| `backups.import_fixes_20250727_005716.final_arrow_syntax_fixer` | 3 | 0 | 3 | 1.00 |
| `backups.import_fixes_20250727_005716.fix_empty_imports` | 3 | 0 | 3 | 1.00 |

## 🛠️ REFACTORING SUGGESTIONS

### 🟡 Reduce Module Coupling (MEDIUM)

Found 6 modules with high coupling

**Affected Modules:**
- `arabic_inflection_corrected`
- `fix_logging_config`
- `advanced_ast_syntax_fixer`
- `advanced_arabic_phonology_system`
- `phonology_core_unified`

**Recommended Actions:**
- Split large modules into smaller, focused modules
- Extract interfaces to reduce direct dependencies
- Use dependency inversion principle
- Consider facade pattern for complex subsystems

## 📦 DEPENDENCY GRAPH DATA

The following data can be used with graph visualization tools:

```json
{
  "nodes": [
    {
      "id": "advanced_arabic_function_words_generator",
      "label": "advanced_arabic_function_words_generator",
      "size": 12,
      "color": "blue",
      "title": "advanced_arabic_function_words_generator\\nImports: 0\\nImported by: 1"
    },
    {
      "id": "advanced_arabic_phonology_system",
      "label": "advanced_arabic_phonology_system",
      "size": 40,
      "color": "blue",
      "title": "advanced_arabic_phonology_system\\nImports: 0\\nImported by: 15"
    },
    {
      "id": "advanced_arabic_proper_names_generator",
      "label": "advanced_arabic_proper_names_generator",
      "size": 10,
      "color": "blue",
      "title": "advanced_arabic_proper_names_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "advanced_ast_syntax_fixer",
      "label": "advanced_ast_syntax_fixer",
      "size": 52,
      "color": "blue",
      "title": "advanced_ast_syntax_fixer\\nImports: 0\\nImported by: 21"
    },
    {
      "id": "advanced_complex_word_demo",
      "label": "advanced_complex_word_demo",
      "size": 10,
      "color": "blue",
      "title": "advanced_complex_word_demo\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "advanced_syntax_fixer",
      "label": "advanced_syntax_fixer",
      "size": 16,
      "color": "blue",
      "title": "advanced_syntax_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "advanced_syntax_validator",
      "label": "advanced_syntax_validator",
      "size": 16,
      "color": "blue",
      "title": "advanced_syntax_validator\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "arabic_demonstrative_evaluation",
      "label": "arabic_demonstrative_evaluation",
      "size": 10,
      "color": "blue",
      "title": "arabic_demonstrative_evaluation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_demonstrative_pronouns_deep_model",
      "label": "arabic_demonstrative_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "arabic_demonstrative_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_demonstrative_pronouns_generator",
      "label": "arabic_demonstrative_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "arabic_demonstrative_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_function_words_analyzer",
      "label": "arabic_function_words_analyzer",
      "size": 10,
      "color": "blue",
      "title": "arabic_function_words_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_function_words_generator",
      "label": "arabic_function_words_generator",
      "size": 10,
      "color": "blue",
      "title": "arabic_function_words_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_function_words_generator_clean",
      "label": "arabic_function_words_generator_clean",
      "size": 10,
      "color": "blue",
      "title": "arabic_function_words_generator_clean\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_inflection_corrected",
      "label": "arabic_inflection_corrected",
      "size": 82,
      "color": "blue",
      "title": "arabic_inflection_corrected\\nImports: 0\\nImported by: 36"
    },
    {
      "id": "arabic_inflection_rules_engine",
      "label": "arabic_inflection_rules_engine",
      "size": 10,
      "color": "blue",
      "title": "arabic_inflection_rules_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_inflection_ultimate",
      "label": "arabic_inflection_ultimate",
      "size": 10,
      "color": "blue",
      "title": "arabic_inflection_ultimate\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_inflection_ultimate_fixed",
      "label": "arabic_inflection_ultimate_fixed",
      "size": 10,
      "color": "blue",
      "title": "arabic_inflection_ultimate_fixed\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_interrogative_pronouns_deep_model",
      "label": "arabic_interrogative_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "arabic_interrogative_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_interrogative_pronouns_enhanced",
      "label": "arabic_interrogative_pronouns_enhanced",
      "size": 10,
      "color": "blue",
      "title": "arabic_interrogative_pronouns_enhanced\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_interrogative_pronouns_final",
      "label": "arabic_interrogative_pronouns_final",
      "size": 10,
      "color": "blue",
      "title": "arabic_interrogative_pronouns_final\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_interrogative_pronouns_generator",
      "label": "arabic_interrogative_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "arabic_interrogative_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_interrogative_pronouns_test_analysis",
      "label": "arabic_interrogative_pronouns_test_analysis",
      "size": 10,
      "color": "blue",
      "title": "arabic_interrogative_pronouns_test_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_mathematical_generator",
      "label": "arabic_mathematical_generator",
      "size": 10,
      "color": "blue",
      "title": "arabic_mathematical_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "ARABIC_MATH_GENERATOR_COMPLETION_REPORT",
      "label": "ARABIC_MATH_GENERATOR_COMPLETION_REPORT",
      "size": 10,
      "color": "blue",
      "title": "ARABIC_MATH_GENERATOR_COMPLETION_REPORT\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_morphological_weight_generator",
      "label": "arabic_morphological_weight_generator",
      "size": 12,
      "color": "blue",
      "title": "arabic_morphological_weight_generator\\nImports: 0\\nImported by: 1"
    },
    {
      "id": "arabic_nlp_status_report",
      "label": "arabic_nlp_status_report",
      "size": 14,
      "color": "blue",
      "title": "arabic_nlp_status_report\\nImports: 0\\nImported by: 2"
    },
    {
      "id": "arabic_normalizer",
      "label": "arabic_normalizer",
      "size": 10,
      "color": "blue",
      "title": "arabic_normalizer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_phoneme_word_decision_tree",
      "label": "arabic_phoneme_word_decision_tree",
      "size": 10,
      "color": "blue",
      "title": "arabic_phoneme_word_decision_tree\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_phonological_foundation",
      "label": "arabic_phonological_foundation",
      "size": 10,
      "color": "blue",
      "title": "arabic_phonological_foundation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_pronouns_analyzer",
      "label": "arabic_pronouns_analyzer",
      "size": 10,
      "color": "blue",
      "title": "arabic_pronouns_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_pronouns_deep_model",
      "label": "arabic_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "arabic_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_pronouns_generator",
      "label": "arabic_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "arabic_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_pronouns_generator_enhanced",
      "label": "arabic_pronouns_generator_enhanced",
      "size": 10,
      "color": "blue",
      "title": "arabic_pronouns_generator_enhanced\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_relative_pronouns_advanced_tester",
      "label": "arabic_relative_pronouns_advanced_tester",
      "size": 10,
      "color": "blue",
      "title": "arabic_relative_pronouns_advanced_tester\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_relative_pronouns_analyzer",
      "label": "arabic_relative_pronouns_analyzer",
      "size": 10,
      "color": "blue",
      "title": "arabic_relative_pronouns_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_relative_pronouns_deep_model",
      "label": "arabic_relative_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "arabic_relative_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_relative_pronouns_deep_model_simplified",
      "label": "arabic_relative_pronouns_deep_model_simplified",
      "size": 12,
      "color": "blue",
      "title": "arabic_relative_pronouns_deep_model_simplified\\nImports: 0\\nImported by: 1"
    },
    {
      "id": "arabic_relative_pronouns_generator",
      "label": "arabic_relative_pronouns_generator",
      "size": 14,
      "color": "blue",
      "title": "arabic_relative_pronouns_generator\\nImports: 0\\nImported by: 2"
    },
    {
      "id": "arabic_syllable_generator",
      "label": "arabic_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "arabic_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_test",
      "label": "arabic_test",
      "size": 10,
      "color": "blue",
      "title": "arabic_test\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_vector_engine",
      "label": "arabic_vector_engine",
      "size": 10,
      "color": "blue",
      "title": "arabic_vector_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "arabic_verb_conjugator",
      "label": "arabic_verb_conjugator",
      "size": 10,
      "color": "blue",
      "title": "arabic_verb_conjugator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "ast_validator",
      "label": "ast_validator",
      "size": 10,
      "color": "blue",
      "title": "ast_validator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "batch_syntax_fixer",
      "label": "batch_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "batch_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "cancel_phoneme_sound_coding",
      "label": "cancel_phoneme_sound_coding",
      "size": 10,
      "color": "blue",
      "title": "cancel_phoneme_sound_coding\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "check_violations",
      "label": "check_violations",
      "size": 10,
      "color": "blue",
      "title": "check_violations\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "check_violations_v2",
      "label": "check_violations_v2",
      "size": 10,
      "color": "blue",
      "title": "check_violations_v2\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "circular_import_analyzer",
      "label": "circular_import_analyzer",
      "size": 14,
      "color": "blue",
      "title": "circular_import_analyzer\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "citation_standardization_system",
      "label": "citation_standardization_system",
      "size": 10,
      "color": "blue",
      "title": "citation_standardization_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "complete_all_13_engines",
      "label": "complete_all_13_engines",
      "size": 10,
      "color": "blue",
      "title": "complete_all_13_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "complete_arabic_phonological_coverage",
      "label": "complete_arabic_phonological_coverage",
      "size": 10,
      "color": "blue",
      "title": "complete_arabic_phonological_coverage\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "complete_arabic_phonological_foundation",
      "label": "complete_arabic_phonological_foundation",
      "size": 12,
      "color": "blue",
      "title": "complete_arabic_phonological_foundation\\nImports: 0\\nImported by: 1"
    },
    {
      "id": "complete_arabic_tracer",
      "label": "complete_arabic_tracer",
      "size": 10,
      "color": "blue",
      "title": "complete_arabic_tracer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "complex_word_analysis_demo",
      "label": "complex_word_analysis_demo",
      "size": 10,
      "color": "blue",
      "title": "complex_word_analysis_demo\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "comprehensive_arabic_phonological_system",
      "label": "comprehensive_arabic_phonological_system",
      "size": 10,
      "color": "blue",
      "title": "comprehensive_arabic_phonological_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "comprehensive_arabic_verb_syllable_generator",
      "label": "comprehensive_arabic_verb_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "comprehensive_arabic_verb_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "comprehensive_encoding_fix",
      "label": "comprehensive_encoding_fix",
      "size": 10,
      "color": "blue",
      "title": "comprehensive_encoding_fix\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "comprehensive_engine_analysis",
      "label": "comprehensive_engine_analysis",
      "size": 10,
      "color": "blue",
      "title": "comprehensive_engine_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "comprehensive_file_analysis",
      "label": "comprehensive_file_analysis",
      "size": 10,
      "color": "blue",
      "title": "comprehensive_file_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "comprehensive_phoneme_engine",
      "label": "comprehensive_phoneme_engine",
      "size": 10,
      "color": "blue",
      "title": "comprehensive_phoneme_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "comprehensive_progressive_system",
      "label": "comprehensive_progressive_system",
      "size": 10,
      "color": "blue",
      "title": "comprehensive_progressive_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "comprehensive_syntax_batch_fixer",
      "label": "comprehensive_syntax_batch_fixer",
      "size": 10,
      "color": "blue",
      "title": "comprehensive_syntax_batch_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "controlled_repair_analysis",
      "label": "controlled_repair_analysis",
      "size": 12,
      "color": "blue",
      "title": "controlled_repair_analysis\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "controlled_string_fixer",
      "label": "controlled_string_fixer",
      "size": 16,
      "color": "blue",
      "title": "controlled_string_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "controlled_syntax_scanner",
      "label": "controlled_syntax_scanner",
      "size": 12,
      "color": "blue",
      "title": "controlled_syntax_scanner\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "create_working_engines",
      "label": "create_working_engines",
      "size": 10,
      "color": "blue",
      "title": "create_working_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "critical_syntax_fixer",
      "label": "critical_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "critical_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "debug_output",
      "label": "debug_output",
      "size": 10,
      "color": "blue",
      "title": "debug_output\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "debug_test",
      "label": "debug_test",
      "size": 10,
      "color": "blue",
      "title": "debug_test\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "demo_all_engines",
      "label": "demo_all_engines",
      "size": 10,
      "color": "blue",
      "title": "demo_all_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "disable_error_squiggles",
      "label": "disable_error_squiggles",
      "size": 10,
      "color": "blue",
      "title": "disable_error_squiggles\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "documentation_standards",
      "label": "documentation_standards",
      "size": 10,
      "color": "blue",
      "title": "documentation_standards\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "emergency_fstring_fixer",
      "label": "emergency_fstring_fixer",
      "size": 10,
      "color": "blue",
      "title": "emergency_fstring_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "emergency_reorganization_system",
      "label": "emergency_reorganization_system",
      "size": 10,
      "color": "blue",
      "title": "emergency_reorganization_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "emergency_syntax_fixer",
      "label": "emergency_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "emergency_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "engine_fix_guide",
      "label": "engine_fix_guide",
      "size": 10,
      "color": "blue",
      "title": "engine_fix_guide\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "enhanced_citation_validator",
      "label": "enhanced_citation_validator",
      "size": 10,
      "color": "blue",
      "title": "enhanced_citation_validator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "enhanced_demo_all_engines",
      "label": "enhanced_demo_all_engines",
      "size": 10,
      "color": "blue",
      "title": "enhanced_demo_all_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "enhanced_final_analysis",
      "label": "enhanced_final_analysis",
      "size": 10,
      "color": "blue",
      "title": "enhanced_final_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "enhanced_system_architecture",
      "label": "enhanced_system_architecture",
      "size": 10,
      "color": "blue",
      "title": "enhanced_system_architecture\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "final_arrow_syntax_fixer",
      "label": "final_arrow_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "final_arrow_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "final_comprehensive_cleanup",
      "label": "final_comprehensive_cleanup",
      "size": 10,
      "color": "blue",
      "title": "final_comprehensive_cleanup\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "final_comprehensive_comparison",
      "label": "final_comprehensive_comparison",
      "size": 10,
      "color": "blue",
      "title": "final_comprehensive_comparison\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "final_comprehensive_report",
      "label": "final_comprehensive_report",
      "size": 10,
      "color": "blue",
      "title": "final_comprehensive_report\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "final_comprehensive_report_generator",
      "label": "final_comprehensive_report_generator",
      "size": 10,
      "color": "blue",
      "title": "final_comprehensive_report_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "final_comprehensive_syntax_fixer",
      "label": "final_comprehensive_syntax_fixer",
      "size": 16,
      "color": "blue",
      "title": "final_comprehensive_syntax_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "fixed_demo_all_engines",
      "label": "fixed_demo_all_engines",
      "size": 10,
      "color": "blue",
      "title": "fixed_demo_all_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_arrow_comparisons",
      "label": "fix_arrow_comparisons",
      "size": 10,
      "color": "blue",
      "title": "fix_arrow_comparisons\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_broken_comparisons",
      "label": "fix_broken_comparisons",
      "size": 10,
      "color": "blue",
      "title": "fix_broken_comparisons\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_common_fstring_errors",
      "label": "fix_common_fstring_errors",
      "size": 10,
      "color": "blue",
      "title": "fix_common_fstring_errors\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_comparison_operators",
      "label": "fix_comparison_operators",
      "size": 10,
      "color": "blue",
      "title": "fix_comparison_operators\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_empty_imports",
      "label": "fix_empty_imports",
      "size": 10,
      "color": "blue",
      "title": "fix_empty_imports\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_fstring_issues",
      "label": "fix_fstring_issues",
      "size": 10,
      "color": "blue",
      "title": "fix_fstring_issues\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_fstring_syntax",
      "label": "fix_fstring_syntax",
      "size": 10,
      "color": "blue",
      "title": "fix_fstring_syntax\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_import_data",
      "label": "fix_import_data",
      "size": 10,
      "color": "blue",
      "title": "fix_import_data\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_logging_config",
      "label": "fix_logging_config",
      "size": 74,
      "color": "blue",
      "title": "fix_logging_config\\nImports: 0\\nImported by: 32"
    },
    {
      "id": "fix_numeric_comparisons",
      "label": "fix_numeric_comparisons",
      "size": 10,
      "color": "blue",
      "title": "fix_numeric_comparisons\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_return_type_arrow",
      "label": "fix_return_type_arrow",
      "size": 10,
      "color": "blue",
      "title": "fix_return_type_arrow\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_syntax_errors",
      "label": "fix_syntax_errors",
      "size": 10,
      "color": "blue",
      "title": "fix_syntax_errors\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "fix_winsurf_issues",
      "label": "fix_winsurf_issues",
      "size": 10,
      "color": "blue",
      "title": "fix_winsurf_issues\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "forensic_audit",
      "label": "forensic_audit",
      "size": 10,
      "color": "blue",
      "title": "forensic_audit\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "hafez_morphology_reactor",
      "label": "hafez_morphology_reactor",
      "size": 10,
      "color": "blue",
      "title": "hafez_morphology_reactor\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "hafez_phonemes",
      "label": "hafez_phonemes",
      "size": 10,
      "color": "blue",
      "title": "hafez_phonemes\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "hafez_syllables",
      "label": "hafez_syllables",
      "size": 10,
      "color": "blue",
      "title": "hafez_syllables\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "harakat_engine",
      "label": "harakat_engine",
      "size": 10,
      "color": "blue",
      "title": "harakat_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "hierarchical_demo",
      "label": "hierarchical_demo",
      "size": 10,
      "color": "blue",
      "title": "hierarchical_demo\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "integrated_engine",
      "label": "integrated_engine",
      "size": 10,
      "color": "blue",
      "title": "integrated_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "integrated_progressive_system",
      "label": "integrated_progressive_system",
      "size": 10,
      "color": "blue",
      "title": "integrated_progressive_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "interactive_progressive_analyzer",
      "label": "interactive_progressive_analyzer",
      "size": 10,
      "color": "blue",
      "title": "interactive_progressive_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "manual_test_fix",
      "label": "manual_test_fix",
      "size": 10,
      "color": "blue",
      "title": "manual_test_fix\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "master_integration_system",
      "label": "master_integration_system",
      "size": 10,
      "color": "blue",
      "title": "master_integration_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "morphology_model",
      "label": "morphology_model",
      "size": 10,
      "color": "blue",
      "title": "morphology_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "phase3b_comprehensive_fixes",
      "label": "phase3b_comprehensive_fixes",
      "size": 10,
      "color": "blue",
      "title": "phase3b_comprehensive_fixes\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "phase3_fstring_bracket_fixes",
      "label": "phase3_fstring_bracket_fixes",
      "size": 14,
      "color": "blue",
      "title": "phase3_fstring_bracket_fixes\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "phonemes_harakat_analyzer",
      "label": "phonemes_harakat_analyzer",
      "size": 10,
      "color": "blue",
      "title": "phonemes_harakat_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "phoneme_model",
      "label": "phoneme_model",
      "size": 10,
      "color": "blue",
      "title": "phoneme_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "phoneme_weight_map",
      "label": "phoneme_weight_map",
      "size": 10,
      "color": "blue",
      "title": "phoneme_weight_map\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "phonology_core_unified",
      "label": "phonology_core_unified",
      "size": 22,
      "color": "blue",
      "title": "phonology_core_unified\\nImports: 0\\nImported by: 6"
    },
    {
      "id": "precise_string_fixer",
      "label": "precise_string_fixer",
      "size": 16,
      "color": "blue",
      "title": "precise_string_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "precision_violation_fixer",
      "label": "precision_violation_fixer",
      "size": 10,
      "color": "blue",
      "title": "precision_violation_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "progressive_integration_analysis",
      "label": "progressive_integration_analysis",
      "size": 10,
      "color": "blue",
      "title": "progressive_integration_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "progressive_vector_tracker",
      "label": "progressive_vector_tracker",
      "size": 10,
      "color": "blue",
      "title": "progressive_vector_tracker\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "PROJECT_COMPLETION_REPORT",
      "label": "PROJECT_COMPLETION_REPORT",
      "size": 10,
      "color": "blue",
      "title": "PROJECT_COMPLETION_REPORT\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "quality_status_report",
      "label": "quality_status_report",
      "size": 10,
      "color": "blue",
      "title": "quality_status_report\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "quick_status",
      "label": "quick_status",
      "size": 10,
      "color": "blue",
      "title": "quick_status\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "remove_arabic_forever",
      "label": "remove_arabic_forever",
      "size": 10,
      "color": "blue",
      "title": "remove_arabic_forever\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "run_all_nlp_engines",
      "label": "run_all_nlp_engines",
      "size": 10,
      "color": "blue",
      "title": "run_all_nlp_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "run_syntax_fixes",
      "label": "run_syntax_fixes",
      "size": 10,
      "color": "blue",
      "title": "run_syntax_fixes\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "simple_arrow_syntax_fixer",
      "label": "simple_arrow_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "simple_arrow_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "simple_phoneme_cancellation",
      "label": "simple_phoneme_cancellation",
      "size": 10,
      "color": "blue",
      "title": "simple_phoneme_cancellation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "simple_test",
      "label": "simple_test",
      "size": 10,
      "color": "blue",
      "title": "simple_test\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "simple_text_cleaner",
      "label": "simple_text_cleaner",
      "size": 10,
      "color": "blue",
      "title": "simple_text_cleaner\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "step1_easy_syntax_fixes",
      "label": "step1_easy_syntax_fixes",
      "size": 10,
      "color": "blue",
      "title": "step1_easy_syntax_fixes\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "step2b_simple_indentation_fixer",
      "label": "step2b_simple_indentation_fixer",
      "size": 10,
      "color": "blue",
      "title": "step2b_simple_indentation_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "step2_indentation_fixer",
      "label": "step2_indentation_fixer",
      "size": 16,
      "color": "blue",
      "title": "step2_indentation_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "strategic_action_plan",
      "label": "strategic_action_plan",
      "size": 10,
      "color": "blue",
      "title": "strategic_action_plan\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "surgical_indentation_fixer",
      "label": "surgical_indentation_fixer",
      "size": 16,
      "color": "blue",
      "title": "surgical_indentation_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "surgical_syntax_fixer_v2",
      "label": "surgical_syntax_fixer_v2",
      "size": 10,
      "color": "blue",
      "title": "surgical_syntax_fixer_v2\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "surgical_syntax_fixer_v3",
      "label": "surgical_syntax_fixer_v3",
      "size": 10,
      "color": "blue",
      "title": "surgical_syntax_fixer_v3\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "surgical_syntax_repair",
      "label": "surgical_syntax_repair",
      "size": 10,
      "color": "blue",
      "title": "surgical_syntax_repair\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "syllable_accomplishments",
      "label": "syllable_accomplishments",
      "size": 10,
      "color": "blue",
      "title": "syllable_accomplishments\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "syllable_encoder",
      "label": "syllable_encoder",
      "size": 10,
      "color": "blue",
      "title": "syllable_encoder\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "syllable_mission_accomplished",
      "label": "syllable_mission_accomplished",
      "size": 10,
      "color": "blue",
      "title": "syllable_mission_accomplished\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "syllable_model",
      "label": "syllable_model",
      "size": 10,
      "color": "blue",
      "title": "syllable_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "syllable_phonological_engine",
      "label": "syllable_phonological_engine",
      "size": 12,
      "color": "blue",
      "title": "syllable_phonological_engine\\nImports: 0\\nImported by: 1"
    },
    {
      "id": "targeted_critical_syntax_fixer",
      "label": "targeted_critical_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "targeted_critical_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "targeted_syntax_scanner",
      "label": "targeted_syntax_scanner",
      "size": 10,
      "color": "blue",
      "title": "targeted_syntax_scanner\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_all_fixed_engines",
      "label": "test_all_fixed_engines",
      "size": 10,
      "color": "blue",
      "title": "test_all_fixed_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_arabic_math_concepts",
      "label": "test_arabic_math_concepts",
      "size": 10,
      "color": "blue",
      "title": "test_arabic_math_concepts\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_morphological_weights",
      "label": "test_morphological_weights",
      "size": 10,
      "color": "blue",
      "title": "test_morphological_weights\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_phoneme_integration",
      "label": "test_phoneme_integration",
      "size": 10,
      "color": "blue",
      "title": "test_phoneme_integration\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_phoneme_simple",
      "label": "test_phoneme_simple",
      "size": 10,
      "color": "blue",
      "title": "test_phoneme_simple\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_proper_names_examples",
      "label": "test_proper_names_examples",
      "size": 10,
      "color": "blue",
      "title": "test_proper_names_examples\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_syllable_engine",
      "label": "test_syllable_engine",
      "size": 10,
      "color": "blue",
      "title": "test_syllable_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_syllable_generator",
      "label": "test_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "test_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_syllable_readiness",
      "label": "test_syllable_readiness",
      "size": 10,
      "color": "blue",
      "title": "test_syllable_readiness\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_waw_hamza",
      "label": "test_waw_hamza",
      "size": 10,
      "color": "blue",
      "title": "test_waw_hamza\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_word",
      "label": "test_word",
      "size": 10,
      "color": "blue",
      "title": "test_word\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "test_zero_layer_comprehensive",
      "label": "test_zero_layer_comprehensive",
      "size": 10,
      "color": "blue",
      "title": "test_zero_layer_comprehensive\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "text_heatmap_generator",
      "label": "text_heatmap_generator",
      "size": 10,
      "color": "blue",
      "title": "text_heatmap_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "toggle_error_squiggles",
      "label": "toggle_error_squiggles",
      "size": 10,
      "color": "blue",
      "title": "toggle_error_squiggles\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "ultimate_syntax_fix",
      "label": "ultimate_syntax_fix",
      "size": 10,
      "color": "blue",
      "title": "ultimate_syntax_fix\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "ultimate_violation_eliminator",
      "label": "ultimate_violation_eliminator",
      "size": 10,
      "color": "blue",
      "title": "ultimate_violation_eliminator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "ultimate_winsurf_eliminator",
      "label": "ultimate_winsurf_eliminator",
      "size": 10,
      "color": "blue",
      "title": "ultimate_winsurf_eliminator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "ultra_precise_syntax_fixer",
      "label": "ultra_precise_syntax_fixer",
      "size": 16,
      "color": "blue",
      "title": "ultra_precise_syntax_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "unified_arabic_engine",
      "label": "unified_arabic_engine",
      "size": 10,
      "color": "blue",
      "title": "unified_arabic_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "unified_phonemes",
      "label": "unified_phonemes",
      "size": 18,
      "color": "blue",
      "title": "unified_phonemes\\nImports: 0\\nImported by: 4"
    },
    {
      "id": "utf8_encoding_fixer",
      "label": "utf8_encoding_fixer",
      "size": 10,
      "color": "blue",
      "title": "utf8_encoding_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "utf8_terminal_cleaner",
      "label": "utf8_terminal_cleaner",
      "size": 10,
      "color": "blue",
      "title": "utf8_terminal_cleaner\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "validate_broken_files",
      "label": "validate_broken_files",
      "size": 10,
      "color": "blue",
      "title": "validate_broken_files\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "validate_citations",
      "label": "validate_citations",
      "size": 10,
      "color": "blue",
      "title": "validate_citations\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "verify_error_suppression",
      "label": "verify_error_suppression",
      "size": 10,
      "color": "blue",
      "title": "verify_error_suppression\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "verify_string_fixes",
      "label": "verify_string_fixes",
      "size": 10,
      "color": "blue",
      "title": "verify_string_fixes\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "version_alignment_toolkit",
      "label": "version_alignment_toolkit",
      "size": 16,
      "color": "blue",
      "title": "version_alignment_toolkit\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "violation_elimination_system",
      "label": "violation_elimination_system",
      "size": 10,
      "color": "blue",
      "title": "violation_elimination_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "visual_heatmap_generator",
      "label": "visual_heatmap_generator",
      "size": 10,
      "color": "blue",
      "title": "visual_heatmap_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "winsurf_standards_library",
      "label": "winsurf_standards_library",
      "size": 10,
      "color": "blue",
      "title": "winsurf_standards_library\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "winsurf_verification_system",
      "label": "winsurf_verification_system",
      "size": 10,
      "color": "blue",
      "title": "winsurf_verification_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "xampp_arabic_phonology",
      "label": "xampp_arabic_phonology",
      "size": 10,
      "color": "blue",
      "title": "xampp_arabic_phonology\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "xampp_test_phonemes_matrix",
      "label": "xampp_test_phonemes_matrix",
      "size": 10,
      "color": "blue",
      "title": "xampp_test_phonemes_matrix\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "yellow_line_eliminator",
      "label": "yellow_line_eliminator",
      "size": 10,
      "color": "blue",
      "title": "yellow_line_eliminator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "zero_layer_phonology",
      "label": "zero_layer_phonology",
      "size": 12,
      "color": "blue",
      "title": "zero_layer_phonology\\nImports: 0\\nImported by: 1"
    },
    {
      "id": "__init__",
      "label": "__init__",
      "size": 10,
      "color": "blue",
      "title": "__init__\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "archive.backup_unified_plugin_manager",
      "label": "backup_unified_plugin_manager",
      "size": 10,
      "color": "blue",
      "title": "archive.backup_unified_plugin_manager\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.advanced_arabic_engine",
      "label": "advanced_arabic_engine",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.advanced_arabic_engine\\nImports: 0\\nImported by: 2"
    },
    {
      "id": "backups.import_fixes_20250727_005716.advanced_arabic_function_words_generator",
      "label": "advanced_arabic_function_words_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.advanced_arabic_function_words_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.advanced_arabic_phonology_system",
      "label": "advanced_arabic_phonology_system",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.advanced_arabic_phonology_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.advanced_arabic_proper_names_generator",
      "label": "advanced_arabic_proper_names_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.advanced_arabic_proper_names_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.advanced_arabic_vector_generator",
      "label": "advanced_arabic_vector_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.advanced_arabic_vector_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.advanced_ast_syntax_fixer",
      "label": "advanced_ast_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.advanced_ast_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.advanced_complex_word_demo",
      "label": "advanced_complex_word_demo",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.advanced_complex_word_demo\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.analyzer",
      "label": "analyzer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.api",
      "label": "api",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.api\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_demonstrative_evaluation",
      "label": "arabic_demonstrative_evaluation",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_demonstrative_evaluation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_demonstrative_pronouns_deep_model",
      "label": "arabic_demonstrative_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_demonstrative_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_demonstrative_pronouns_generator",
      "label": "arabic_demonstrative_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_demonstrative_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_function_words_analyzer",
      "label": "arabic_function_words_analyzer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_function_words_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_function_words_generator",
      "label": "arabic_function_words_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_function_words_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_function_words_generator_clean",
      "label": "arabic_function_words_generator_clean",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_function_words_generator_clean\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_inflection_corrected",
      "label": "arabic_inflection_corrected",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_inflection_corrected\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_inflection_rules_engine",
      "label": "arabic_inflection_rules_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_inflection_rules_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_inflection_ultimate",
      "label": "arabic_inflection_ultimate",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_inflection_ultimate\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_inflection_ultimate_fixed",
      "label": "arabic_inflection_ultimate_fixed",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_inflection_ultimate_fixed\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_deep_model",
      "label": "arabic_interrogative_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_enhanced",
      "label": "arabic_interrogative_pronouns_enhanced",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_enhanced\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_final",
      "label": "arabic_interrogative_pronouns_final",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_final\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_generator",
      "label": "arabic_interrogative_pronouns_generator",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_generator\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_test_analysis",
      "label": "arabic_interrogative_pronouns_test_analysis",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_test_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_mathematical_generator",
      "label": "arabic_mathematical_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_mathematical_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.ARABIC_MATH_GENERATOR_COMPLETION_REPORT",
      "label": "ARABIC_MATH_GENERATOR_COMPLETION_REPORT",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.ARABIC_MATH_GENERATOR_COMPLETION_REPORT\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_morphological_weight_generator",
      "label": "arabic_morphological_weight_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_morphological_weight_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_nlp_status_report",
      "label": "arabic_nlp_status_report",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_nlp_status_report\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_normalizer",
      "label": "arabic_normalizer",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_normalizer\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_phoneme_word_decision_tree",
      "label": "arabic_phoneme_word_decision_tree",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_phoneme_word_decision_tree\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_phonological_foundation",
      "label": "arabic_phonological_foundation",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_phonological_foundation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_pronouns_analyzer",
      "label": "arabic_pronouns_analyzer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_pronouns_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_pronouns_deep_model",
      "label": "arabic_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_pronouns_generator",
      "label": "arabic_pronouns_generator",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_pronouns_generator\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_pronouns_generator_enhanced",
      "label": "arabic_pronouns_generator_enhanced",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_pronouns_generator_enhanced\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_advanced_tester",
      "label": "arabic_relative_pronouns_advanced_tester",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_advanced_tester\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_analyzer",
      "label": "arabic_relative_pronouns_analyzer",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_analyzer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_deep_model",
      "label": "arabic_relative_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_deep_model_simplified",
      "label": "arabic_relative_pronouns_deep_model_simplified",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_deep_model_simplified\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_generator",
      "label": "arabic_relative_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_syllable_generator",
      "label": "arabic_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_test",
      "label": "arabic_test",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_test\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_vector_engine",
      "label": "arabic_vector_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_vector_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.arabic_verb_conjugator",
      "label": "arabic_verb_conjugator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.arabic_verb_conjugator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.assimilation",
      "label": "assimilation",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.assimilation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.ast_validator",
      "label": "ast_validator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.ast_validator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.backup_unified_plugin_manager",
      "label": "backup_unified_plugin_manager",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.backup_unified_plugin_manager\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.base_engine",
      "label": "base_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.base_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.batch_syntax_fixer",
      "label": "batch_syntax_fixer",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.batch_syntax_fixer\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.check_violations",
      "label": "check_violations",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.check_violations\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.citation_standardization_system",
      "label": "citation_standardization_system",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.citation_standardization_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.classifier",
      "label": "classifier",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.classifier\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.cli_text_processor",
      "label": "cli_text_processor",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.cli_text_processor\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comparative",
      "label": "comparative",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comparative\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.complete_all_13_engines",
      "label": "complete_all_13_engines",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.complete_all_13_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.complete_arabic_phonological_coverage",
      "label": "complete_arabic_phonological_coverage",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.complete_arabic_phonological_coverage\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.complete_arabic_phonological_foundation",
      "label": "complete_arabic_phonological_foundation",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.complete_arabic_phonological_foundation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.complete_arabic_tracer",
      "label": "complete_arabic_tracer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.complete_arabic_tracer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.complex_word_analysis_demo",
      "label": "complex_word_analysis_demo",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.complex_word_analysis_demo\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comprehensive_arabic_phonological_system",
      "label": "comprehensive_arabic_phonological_system",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comprehensive_arabic_phonological_system\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comprehensive_arabic_verb_syllable_generator",
      "label": "comprehensive_arabic_verb_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comprehensive_arabic_verb_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comprehensive_encoding_fix",
      "label": "comprehensive_encoding_fix",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comprehensive_encoding_fix\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comprehensive_engine_analysis",
      "label": "comprehensive_engine_analysis",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comprehensive_engine_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comprehensive_file_analysis",
      "label": "comprehensive_file_analysis",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comprehensive_file_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comprehensive_phoneme_engine",
      "label": "comprehensive_phoneme_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comprehensive_phoneme_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comprehensive_progressive_system",
      "label": "comprehensive_progressive_system",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comprehensive_progressive_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.comprehensive_syntax_batch_fixer",
      "label": "comprehensive_syntax_batch_fixer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.comprehensive_syntax_batch_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.config",
      "label": "config",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.config\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.conftest",
      "label": "conftest",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.conftest\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.create_working_engines",
      "label": "create_working_engines",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.create_working_engines\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.critical_syntax_fixer",
      "label": "critical_syntax_fixer",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.critical_syntax_fixer\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.debug_output",
      "label": "debug_output",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.debug_output\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.debug_test",
      "label": "debug_test",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.debug_test\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.deletion",
      "label": "deletion",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.deletion\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.derive",
      "label": "derive",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.derive\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.emergency_fstring_fixer",
      "label": "emergency_fstring_fixer",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.emergency_fstring_fixer\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.emergency_syntax_fixer",
      "label": "emergency_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.emergency_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine",
      "label": "engine",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine_advanced",
      "label": "engine_advanced",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine_advanced\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine_batch_runner",
      "label": "engine_batch_runner",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine_batch_runner\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine_clean",
      "label": "engine_clean",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine_clean\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine_fix_guide",
      "label": "engine_fix_guide",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine_fix_guide\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine_health_monitor",
      "label": "engine_health_monitor",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine_health_monitor\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine_new",
      "label": "engine_new",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine_new\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine_old",
      "label": "engine_old",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine_old\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.engine_professional",
      "label": "engine_professional",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.engine_professional\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.enhanced_final_analysis",
      "label": "enhanced_final_analysis",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.enhanced_final_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.enhanced_system_architecture",
      "label": "enhanced_system_architecture",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.enhanced_system_architecture\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.feature_space",
      "label": "feature_space",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.feature_space\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.final_arrow_syntax_fixer",
      "label": "final_arrow_syntax_fixer",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.final_arrow_syntax_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.final_comprehensive_comparison",
      "label": "final_comprehensive_comparison",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.final_comprehensive_comparison\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.final_comprehensive_report",
      "label": "final_comprehensive_report",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.final_comprehensive_report\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.final_comprehensive_report_generator",
      "label": "final_comprehensive_report_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.final_comprehensive_report_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_arrow_comparisons",
      "label": "fix_arrow_comparisons",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_arrow_comparisons\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_broken_comparisons",
      "label": "fix_broken_comparisons",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_broken_comparisons\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_common_fstring_errors",
      "label": "fix_common_fstring_errors",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_common_fstring_errors\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_comparison_operators",
      "label": "fix_comparison_operators",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_comparison_operators\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_empty_imports",
      "label": "fix_empty_imports",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_empty_imports\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_fstring_issues",
      "label": "fix_fstring_issues",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_fstring_issues\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_fstring_syntax",
      "label": "fix_fstring_syntax",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_fstring_syntax\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_import_data",
      "label": "fix_import_data",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_import_data\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_logging_config",
      "label": "fix_logging_config",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_logging_config\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_numeric_comparisons",
      "label": "fix_numeric_comparisons",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_numeric_comparisons\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_return_type_arrow",
      "label": "fix_return_type_arrow",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_return_type_arrow\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_syntax_errors",
      "label": "fix_syntax_errors",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_syntax_errors\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.fix_winsurf_issues",
      "label": "fix_winsurf_issues",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.fix_winsurf_issues\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.forensic_audit",
      "label": "forensic_audit",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.forensic_audit\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.hafez_morphology_reactor",
      "label": "hafez_morphology_reactor",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.hafez_morphology_reactor\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.hafez_phonemes",
      "label": "hafez_phonemes",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.hafez_phonemes\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.hafez_syllables",
      "label": "hafez_syllables",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.hafez_syllables\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.harakat_engine",
      "label": "harakat_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.harakat_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.hierarchical_demo",
      "label": "hierarchical_demo",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.hierarchical_demo\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.hierarchical_graph_engine",
      "label": "hierarchical_graph_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.hierarchical_graph_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.inflect",
      "label": "inflect",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.inflect\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.inflection",
      "label": "inflection",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.inflection\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.integrated_engine",
      "label": "integrated_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.integrated_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.integrated_progressive_system",
      "label": "integrated_progressive_system",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.integrated_progressive_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.interactive_progressive_analyzer",
      "label": "interactive_progressive_analyzer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.interactive_progressive_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.inversion",
      "label": "inversion",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.inversion\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.machine_learning_engine",
      "label": "machine_learning_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.machine_learning_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.manual_test_fix",
      "label": "manual_test_fix",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.manual_test_fix\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.morphological_models",
      "label": "morphological_models",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.morphological_models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.morphology",
      "label": "morphology",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.morphology\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.morphology_model",
      "label": "morphology_model",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.morphology_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.particle_classify",
      "label": "particle_classify",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.particle_classify\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.particle_segment",
      "label": "particle_segment",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.particle_segment\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.pattern_embed",
      "label": "pattern_embed",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.pattern_embed\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.phonemes_harakat_analyzer",
      "label": "phonemes_harakat_analyzer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.phonemes_harakat_analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.phoneme_model",
      "label": "phoneme_model",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.phoneme_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.phoneme_weight_map",
      "label": "phoneme_weight_map",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.phoneme_weight_map\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.phonology",
      "label": "phonology",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.phonology\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.phonology_core_unified",
      "label": "phonology_core_unified",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.phonology_core_unified\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.precision_violation_fixer",
      "label": "precision_violation_fixer",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.precision_violation_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.progressive_integration_analysis",
      "label": "progressive_integration_analysis",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.progressive_integration_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.progressive_vector_tracker",
      "label": "progressive_vector_tracker",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.progressive_vector_tracker\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.PROJECT_COMPLETION_REPORT",
      "label": "PROJECT_COMPLETION_REPORT",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.PROJECT_COMPLETION_REPORT\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.project_status_checker",
      "label": "project_status_checker",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.project_status_checker\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.quality_status_report",
      "label": "quality_status_report",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.quality_status_report\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.quick_status",
      "label": "quick_status",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.quick_status\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.remove_arabic_forever",
      "label": "remove_arabic_forever",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.remove_arabic_forever\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.root_embed",
      "label": "root_embed",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.root_embed\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.rule_base",
      "label": "rule_base",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.rule_base\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.run_all_nlp_engines",
      "label": "run_all_nlp_engines",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.run_all_nlp_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.run_syntax_fixes",
      "label": "run_syntax_fixes",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.run_syntax_fixes\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.segmenter",
      "label": "segmenter",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.segmenter\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.settings",
      "label": "settings",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.settings\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.simple_arrow_syntax_fixer",
      "label": "simple_arrow_syntax_fixer",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.simple_arrow_syntax_fixer\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.simple_test",
      "label": "simple_test",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.simple_test\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.simple_text_cleaner",
      "label": "simple_text_cleaner",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.simple_text_cleaner\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.strategic_action_plan",
      "label": "strategic_action_plan",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.strategic_action_plan\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.surgical_syntax_fixer_v2",
      "label": "surgical_syntax_fixer_v2",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.surgical_syntax_fixer_v2\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.surgical_syntax_fixer_v3",
      "label": "surgical_syntax_fixer_v3",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.surgical_syntax_fixer_v3\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.surgical_syntax_repair",
      "label": "surgical_syntax_repair",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.surgical_syntax_repair\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.syllable_accomplishments",
      "label": "syllable_accomplishments",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.syllable_accomplishments\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.syllable_check",
      "label": "syllable_check",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.syllable_check\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.syllable_encoder",
      "label": "syllable_encoder",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.syllable_encoder\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.syllable_mission_accomplished",
      "label": "syllable_mission_accomplished",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.syllable_mission_accomplished\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.syllable_model",
      "label": "syllable_model",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.syllable_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.syllable_phonological_engine",
      "label": "syllable_phonological_engine",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.syllable_phonological_engine\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.targeted_critical_syntax_fixer",
      "label": "targeted_critical_syntax_fixer",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.targeted_critical_syntax_fixer\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.targeted_syntax_scanner",
      "label": "targeted_syntax_scanner",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.targeted_syntax_scanner\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.templates",
      "label": "templates",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.templates\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_all_fixed_engines",
      "label": "test_all_fixed_engines",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_all_fixed_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_arabic_math_concepts",
      "label": "test_arabic_math_concepts",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_arabic_math_concepts\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_arabic_system",
      "label": "test_arabic_system",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_arabic_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_base_engine",
      "label": "test_base_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_base_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_comprehensive_engines",
      "label": "test_comprehensive_engines",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_comprehensive_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_config",
      "label": "test_config",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_config\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_core_integration",
      "label": "test_core_integration",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_core_integration\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_hello_world",
      "label": "test_hello_world",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_hello_world\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_morphological_weights",
      "label": "test_morphological_weights",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_morphological_weights\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_phoneme_integration",
      "label": "test_phoneme_integration",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_phoneme_integration\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_phoneme_processing",
      "label": "test_phoneme_processing",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_phoneme_processing\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_phoneme_simple",
      "label": "test_phoneme_simple",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_phoneme_simple\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_proper_names_examples",
      "label": "test_proper_names_examples",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_proper_names_examples\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_syllable_engine",
      "label": "test_syllable_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_syllable_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_syllable_generator",
      "label": "test_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_syllable_processing",
      "label": "test_syllable_processing",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_syllable_processing\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_syllable_readiness",
      "label": "test_syllable_readiness",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_syllable_readiness\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_waw_hamza",
      "label": "test_waw_hamza",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_waw_hamza\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_word",
      "label": "test_word",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_word\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.test_zero_layer_comprehensive",
      "label": "test_zero_layer_comprehensive",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.test_zero_layer_comprehensive\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.text_heatmap_generator",
      "label": "text_heatmap_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.text_heatmap_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.ultimate_syntax_fix",
      "label": "ultimate_syntax_fix",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.ultimate_syntax_fix\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.ultimate_violation_eliminator",
      "label": "ultimate_violation_eliminator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.ultimate_violation_eliminator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.ultimate_winsurf_eliminator",
      "label": "ultimate_winsurf_eliminator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.ultimate_winsurf_eliminator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.unified_arabic_engine",
      "label": "unified_arabic_engine",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.unified_arabic_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.unified_phonemes",
      "label": "unified_phonemes",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.unified_phonemes\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.utf8_encoding_fixer",
      "label": "utf8_encoding_fixer",
      "size": 16,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.utf8_encoding_fixer\\nImports: 3\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.utf8_terminal_cleaner",
      "label": "utf8_terminal_cleaner",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.utf8_terminal_cleaner\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.validate_broken_files",
      "label": "validate_broken_files",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.validate_broken_files\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.validate_citations",
      "label": "validate_citations",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.validate_citations\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.validate_tools_ast",
      "label": "validate_tools_ast",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.validate_tools_ast\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.verb_check",
      "label": "verb_check",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.verb_check\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.verify_string_fixes",
      "label": "verify_string_fixes",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.verify_string_fixes\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.violation_elimination_system",
      "label": "violation_elimination_system",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.violation_elimination_system\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.visual_heatmap_generator",
      "label": "visual_heatmap_generator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.visual_heatmap_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.winsurf_standards_library",
      "label": "winsurf_standards_library",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.winsurf_standards_library\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.winsurf_verification_system",
      "label": "winsurf_verification_system",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.winsurf_verification_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.xampp_arabic_phonology",
      "label": "xampp_arabic_phonology",
      "size": 14,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.xampp_arabic_phonology\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.xampp_test_phonemes_matrix",
      "label": "xampp_test_phonemes_matrix",
      "size": 12,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.xampp_test_phonemes_matrix\\nImports: 1\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.yellow_line_eliminator",
      "label": "yellow_line_eliminator",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.yellow_line_eliminator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716.zero_layer_phonology",
      "label": "zero_layer_phonology",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716.zero_layer_phonology\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.import_fixes_20250727_005716",
      "label": "import_fixes_20250727_005716",
      "size": 10,
      "color": "blue",
      "title": "backups.import_fixes_20250727_005716\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "backups.indentation_fixes_20250727_010252",
      "label": "indentation_fixes_20250727_010252",
      "size": 10,
      "color": "blue",
      "title": "backups.indentation_fixes_20250727_010252\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.base_engine",
      "label": "base_engine",
      "size": 10,
      "color": "blue",
      "title": "core.base_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.config",
      "label": "config",
      "size": 10,
      "color": "blue",
      "title": "core.config\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.hierarchical_graph_engine",
      "label": "hierarchical_graph_engine",
      "size": 10,
      "color": "blue",
      "title": "core.hierarchical_graph_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.inflection",
      "label": "inflection",
      "size": 10,
      "color": "blue",
      "title": "core.inflection\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.morphology",
      "label": "morphology",
      "size": 10,
      "color": "blue",
      "title": "core.morphology\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.phonology",
      "label": "phonology",
      "size": 10,
      "color": "blue",
      "title": "core.phonology\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.unified_arabic_engine",
      "label": "unified_arabic_engine",
      "size": 10,
      "color": "blue",
      "title": "core.unified_arabic_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core",
      "label": "core",
      "size": 22,
      "color": "blue",
      "title": "core\\nImports: 0\\nImported by: 6"
    },
    {
      "id": "core.nlp.advanced_arabic_engine",
      "label": "advanced_arabic_engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.advanced_arabic_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.base_engine",
      "label": "base_engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.base_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.machine_learning_engine",
      "label": "machine_learning_engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.machine_learning_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp",
      "label": "nlp",
      "size": 10,
      "color": "blue",
      "title": "core.nlp\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.derivation.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.derivation.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.derivation",
      "label": "derivation",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.derivation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.derivation.models.comparative",
      "label": "comparative",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.derivation.models.comparative\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.derivation.models.derive",
      "label": "derive",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.derivation.models.derive\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.derivation.models.pattern_embed",
      "label": "pattern_embed",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.derivation.models.pattern_embed\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.derivation.models.root_embed",
      "label": "root_embed",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.derivation.models.root_embed\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.derivation.models",
      "label": "models",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.derivation.models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.frozen_root.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.frozen_root.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.frozen_root.engine_new",
      "label": "engine_new",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.frozen_root.engine_new\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.frozen_root",
      "label": "frozen_root",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.frozen_root\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.frozen_root.models.classifier",
      "label": "classifier",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.frozen_root.models.classifier\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.frozen_root.models.syllable_check",
      "label": "syllable_check",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.frozen_root.models.syllable_check\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.frozen_root.models.verb_check",
      "label": "verb_check",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.frozen_root.models.verb_check\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.frozen_root.models",
      "label": "models",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.frozen_root.models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.full_pipeline.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.full_pipeline.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.full_pipeline",
      "label": "full_pipeline",
      "size": 14,
      "color": "blue",
      "title": "core.nlp.full_pipeline\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "core.nlp.grammatical_particles.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.grammatical_particles.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.grammatical_particles",
      "label": "grammatical_particles",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.grammatical_particles\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.inflection.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.inflection.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.inflection",
      "label": "inflection",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.inflection\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.inflection.models.feature_space",
      "label": "feature_space",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.inflection.models.feature_space\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.inflection.models.inflect",
      "label": "inflect",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.inflection.models.inflect\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.inflection.models",
      "label": "models",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.inflection.models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.morphology.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.morphology.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.morphology.config.settings",
      "label": "settings",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.morphology.config.settings\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.morphology.models.morphological_models",
      "label": "morphological_models",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.morphology.models.morphological_models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.particles.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.particles.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.particles.models.particle_classify",
      "label": "particle_classify",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.particles.models.particle_classify\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.particles.models.particle_segment",
      "label": "particle_segment",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.particles.models.particle_segment\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phoneme.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phoneme.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phoneme",
      "label": "phoneme",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phoneme\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phoneme_advanced.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phoneme_advanced.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.api",
      "label": "api",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.api\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.engine_old",
      "label": "engine_old",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.engine_old\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.engine_professional",
      "label": "engine_professional",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.engine_professional\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological",
      "label": "phonological",
      "size": 14,
      "color": "blue",
      "title": "core.nlp.phonological\\nImports: 2\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.models.assimilation",
      "label": "assimilation",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.models.assimilation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.models.deletion",
      "label": "deletion",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.models.deletion\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.models.inversion",
      "label": "inversion",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.models.inversion\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.models.rule_base",
      "label": "rule_base",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.models.rule_base\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonological.models",
      "label": "models",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonological.models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonology.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonology.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonology.engine_clean",
      "label": "engine_clean",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonology.engine_clean\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.phonology.config.settings",
      "label": "settings",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.phonology.config.settings\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.syllable.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.syllable.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.syllable.engine_advanced",
      "label": "engine_advanced",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.syllable.engine_advanced\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.syllable",
      "label": "syllable",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.syllable\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.syllable.models.segmenter",
      "label": "segmenter",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.syllable.models.segmenter\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.syllable.models.templates",
      "label": "templates",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.syllable.models.templates\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.syllable.models",
      "label": "models",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.syllable.models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.weight.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.weight.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "core.nlp.weight.models.analyzer",
      "label": "analyzer",
      "size": 10,
      "color": "blue",
      "title": "core.nlp.weight.models.analyzer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.advanced_arabic_function_words_generator",
      "label": "advanced_arabic_function_words_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.advanced_arabic_function_words_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.advanced_arabic_phonology_system",
      "label": "advanced_arabic_phonology_system",
      "size": 10,
      "color": "blue",
      "title": "experimental.advanced_arabic_phonology_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.advanced_arabic_proper_names_generator",
      "label": "advanced_arabic_proper_names_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.advanced_arabic_proper_names_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.advanced_arabic_vector_generator",
      "label": "advanced_arabic_vector_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.advanced_arabic_vector_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.advanced_ast_syntax_fixer",
      "label": "advanced_ast_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "experimental.advanced_ast_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.advanced_complex_word_demo",
      "label": "advanced_complex_word_demo",
      "size": 10,
      "color": "blue",
      "title": "experimental.advanced_complex_word_demo\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_demonstrative_evaluation",
      "label": "arabic_demonstrative_evaluation",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_demonstrative_evaluation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_demonstrative_pronouns_deep_model",
      "label": "arabic_demonstrative_pronouns_deep_model",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_demonstrative_pronouns_deep_model\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_demonstrative_pronouns_generator",
      "label": "arabic_demonstrative_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_demonstrative_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_function_words_generator",
      "label": "arabic_function_words_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_function_words_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_function_words_generator_clean",
      "label": "arabic_function_words_generator_clean",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_function_words_generator_clean\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_interrogative_pronouns_generator",
      "label": "arabic_interrogative_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_interrogative_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_mathematical_generator",
      "label": "arabic_mathematical_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_mathematical_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.ARABIC_MATH_GENERATOR_COMPLETION_REPORT",
      "label": "ARABIC_MATH_GENERATOR_COMPLETION_REPORT",
      "size": 10,
      "color": "blue",
      "title": "experimental.ARABIC_MATH_GENERATOR_COMPLETION_REPORT\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_morphological_weight_generator",
      "label": "arabic_morphological_weight_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_morphological_weight_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_pronouns_generator",
      "label": "arabic_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_pronouns_generator_enhanced",
      "label": "arabic_pronouns_generator_enhanced",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_pronouns_generator_enhanced\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_relative_pronouns_advanced_tester",
      "label": "arabic_relative_pronouns_advanced_tester",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_relative_pronouns_advanced_tester\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_relative_pronouns_generator",
      "label": "arabic_relative_pronouns_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_relative_pronouns_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.arabic_syllable_generator",
      "label": "arabic_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.arabic_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.comprehensive_arabic_verb_syllable_generator",
      "label": "comprehensive_arabic_verb_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.comprehensive_arabic_verb_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.final_comprehensive_report_generator",
      "label": "final_comprehensive_report_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.final_comprehensive_report_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "experimental.visual_heatmap_generator",
      "label": "visual_heatmap_generator",
      "size": 10,
      "color": "blue",
      "title": "experimental.visual_heatmap_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.base_engine",
      "label": "base_engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.base_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp",
      "label": "nlp",
      "size": 14,
      "color": "blue",
      "title": "nlp\\nImports: 0\\nImported by: 2"
    },
    {
      "id": "nlp.derivation.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.derivation.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.derivation",
      "label": "derivation",
      "size": 10,
      "color": "blue",
      "title": "nlp.derivation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.derivation.models.comparative",
      "label": "comparative",
      "size": 10,
      "color": "blue",
      "title": "nlp.derivation.models.comparative\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.derivation.models.derive",
      "label": "derive",
      "size": 10,
      "color": "blue",
      "title": "nlp.derivation.models.derive\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.derivation.models.pattern_embed",
      "label": "pattern_embed",
      "size": 10,
      "color": "blue",
      "title": "nlp.derivation.models.pattern_embed\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.derivation.models.root_embed",
      "label": "root_embed",
      "size": 10,
      "color": "blue",
      "title": "nlp.derivation.models.root_embed\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.derivation.models",
      "label": "models",
      "size": 10,
      "color": "blue",
      "title": "nlp.derivation.models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.frozen_root.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.frozen_root.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.frozen_root.engine_new",
      "label": "engine_new",
      "size": 10,
      "color": "blue",
      "title": "nlp.frozen_root.engine_new\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.frozen_root",
      "label": "frozen_root",
      "size": 10,
      "color": "blue",
      "title": "nlp.frozen_root\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.frozen_root.models.classifier",
      "label": "classifier",
      "size": 10,
      "color": "blue",
      "title": "nlp.frozen_root.models.classifier\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.frozen_root.models.verb_check",
      "label": "verb_check",
      "size": 10,
      "color": "blue",
      "title": "nlp.frozen_root.models.verb_check\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.frozen_root.models",
      "label": "models",
      "size": 10,
      "color": "blue",
      "title": "nlp.frozen_root.models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.full_pipeline.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.full_pipeline.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.grammatical_particles.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.grammatical_particles.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.inflection.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.inflection.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.inflection.models",
      "label": "models",
      "size": 10,
      "color": "blue",
      "title": "nlp.inflection.models\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.particles.models.particle_segment",
      "label": "particle_segment",
      "size": 10,
      "color": "blue",
      "title": "nlp.particles.models.particle_segment\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.phoneme.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.phoneme.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.phoneme",
      "label": "phoneme",
      "size": 10,
      "color": "blue",
      "title": "nlp.phoneme\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.phonological.api",
      "label": "api",
      "size": 10,
      "color": "blue",
      "title": "nlp.phonological.api\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.phonological.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.phonological.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.phonological.models.assimilation",
      "label": "assimilation",
      "size": 10,
      "color": "blue",
      "title": "nlp.phonological.models.assimilation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.phonological.models.deletion",
      "label": "deletion",
      "size": 10,
      "color": "blue",
      "title": "nlp.phonological.models.deletion\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.phonological.models.rule_base",
      "label": "rule_base",
      "size": 10,
      "color": "blue",
      "title": "nlp.phonological.models.rule_base\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.syllable.engine",
      "label": "engine",
      "size": 10,
      "color": "blue",
      "title": "nlp.syllable.engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.syllable.engine_advanced",
      "label": "engine_advanced",
      "size": 10,
      "color": "blue",
      "title": "nlp.syllable.engine_advanced\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.syllable.models.segmenter",
      "label": "segmenter",
      "size": 10,
      "color": "blue",
      "title": "nlp.syllable.models.segmenter\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "nlp.syllable.models.templates",
      "label": "templates",
      "size": 10,
      "color": "blue",
      "title": "nlp.syllable.models.templates\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.conftest",
      "label": "conftest",
      "size": 10,
      "color": "blue",
      "title": "tests.conftest\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.engine_batch_runner",
      "label": "engine_batch_runner",
      "size": 10,
      "color": "blue",
      "title": "tests.engine_batch_runner\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.engine_health_monitor",
      "label": "engine_health_monitor",
      "size": 10,
      "color": "blue",
      "title": "tests.engine_health_monitor\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_all_fixed_engines",
      "label": "test_all_fixed_engines",
      "size": 10,
      "color": "blue",
      "title": "tests.test_all_fixed_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_arabic_math_concepts",
      "label": "test_arabic_math_concepts",
      "size": 10,
      "color": "blue",
      "title": "tests.test_arabic_math_concepts\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_arabic_system",
      "label": "test_arabic_system",
      "size": 10,
      "color": "blue",
      "title": "tests.test_arabic_system\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_comprehensive_engines",
      "label": "test_comprehensive_engines",
      "size": 10,
      "color": "blue",
      "title": "tests.test_comprehensive_engines\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_core_integration",
      "label": "test_core_integration",
      "size": 10,
      "color": "blue",
      "title": "tests.test_core_integration\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_hello_world",
      "label": "test_hello_world",
      "size": 10,
      "color": "blue",
      "title": "tests.test_hello_world\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_morphological_weights",
      "label": "test_morphological_weights",
      "size": 10,
      "color": "blue",
      "title": "tests.test_morphological_weights\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_phoneme_integration",
      "label": "test_phoneme_integration",
      "size": 10,
      "color": "blue",
      "title": "tests.test_phoneme_integration\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_phoneme_processing",
      "label": "test_phoneme_processing",
      "size": 10,
      "color": "blue",
      "title": "tests.test_phoneme_processing\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_phoneme_simple",
      "label": "test_phoneme_simple",
      "size": 10,
      "color": "blue",
      "title": "tests.test_phoneme_simple\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_proper_names_examples",
      "label": "test_proper_names_examples",
      "size": 10,
      "color": "blue",
      "title": "tests.test_proper_names_examples\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_syllable_engine",
      "label": "test_syllable_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_syllable_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_syllable_generator",
      "label": "test_syllable_generator",
      "size": 10,
      "color": "blue",
      "title": "tests.test_syllable_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_syllable_processing",
      "label": "test_syllable_processing",
      "size": 10,
      "color": "blue",
      "title": "tests.test_syllable_processing\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_syllable_readiness",
      "label": "test_syllable_readiness",
      "size": 10,
      "color": "blue",
      "title": "tests.test_syllable_readiness\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_waw_hamza",
      "label": "test_waw_hamza",
      "size": 10,
      "color": "blue",
      "title": "tests.test_waw_hamza\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_word",
      "label": "test_word",
      "size": 10,
      "color": "blue",
      "title": "tests.test_word\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_zero_layer_comprehensive",
      "label": "test_zero_layer_comprehensive",
      "size": 10,
      "color": "blue",
      "title": "tests.test_zero_layer_comprehensive\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests",
      "label": "tests",
      "size": 10,
      "color": "blue",
      "title": "tests\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_core.test_base_engine",
      "label": "test_base_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_core.test_base_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_core.test_config",
      "label": "test_config",
      "size": 10,
      "color": "blue",
      "title": "tests.test_core.test_config\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_core",
      "label": "test_core",
      "size": 10,
      "color": "blue",
      "title": "tests.test_core\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp",
      "label": "test_nlp",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_derivation.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_derivation.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_derivation",
      "label": "test_derivation",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_derivation\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_frozen_root.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_frozen_root.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_frozen_root",
      "label": "test_frozen_root",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_frozen_root\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_inflection.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_inflection.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_inflection",
      "label": "test_inflection",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_inflection\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_morphology.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_morphology.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_morphology",
      "label": "test_morphology",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_morphology\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_phoneme.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_phoneme.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_phoneme",
      "label": "test_phoneme",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_phoneme\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_phonological.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_phonological.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_phonological",
      "label": "test_phonological",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_phonological\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_syllable.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_syllable.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_syllable",
      "label": "test_syllable",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_syllable\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_weight.test_engine",
      "label": "test_engine",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_weight.test_engine\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tests.test_nlp.test_weight",
      "label": "test_weight",
      "size": 10,
      "color": "blue",
      "title": "tests.test_nlp.test_weight\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.arabic_nlp_status_report",
      "label": "arabic_nlp_status_report",
      "size": 10,
      "color": "blue",
      "title": "tools.arabic_nlp_status_report\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.cli_text_processor",
      "label": "cli_text_processor",
      "size": 10,
      "color": "blue",
      "title": "tools.cli_text_processor\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.comprehensive_file_analysis",
      "label": "comprehensive_file_analysis",
      "size": 10,
      "color": "blue",
      "title": "tools.comprehensive_file_analysis\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.comprehensive_syntax_batch_fixer",
      "label": "comprehensive_syntax_batch_fixer",
      "size": 10,
      "color": "blue",
      "title": "tools.comprehensive_syntax_batch_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.critical_syntax_fixer",
      "label": "critical_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "tools.critical_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.emergency_syntax_fixer",
      "label": "emergency_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "tools.emergency_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.final_arrow_syntax_fixer",
      "label": "final_arrow_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "tools.final_arrow_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.final_comprehensive_report",
      "label": "final_comprehensive_report",
      "size": 10,
      "color": "blue",
      "title": "tools.final_comprehensive_report\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.precision_violation_fixer",
      "label": "precision_violation_fixer",
      "size": 10,
      "color": "blue",
      "title": "tools.precision_violation_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.PROJECT_COMPLETION_REPORT",
      "label": "PROJECT_COMPLETION_REPORT",
      "size": 10,
      "color": "blue",
      "title": "tools.PROJECT_COMPLETION_REPORT\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.project_status_checker",
      "label": "project_status_checker",
      "size": 10,
      "color": "blue",
      "title": "tools.project_status_checker\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.quality_status_report",
      "label": "quality_status_report",
      "size": 10,
      "color": "blue",
      "title": "tools.quality_status_report\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.quick_status",
      "label": "quick_status",
      "size": 10,
      "color": "blue",
      "title": "tools.quick_status\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.simple_arrow_syntax_fixer",
      "label": "simple_arrow_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "tools.simple_arrow_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.simple_text_cleaner",
      "label": "simple_text_cleaner",
      "size": 10,
      "color": "blue",
      "title": "tools.simple_text_cleaner\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.strategic_action_plan",
      "label": "strategic_action_plan",
      "size": 10,
      "color": "blue",
      "title": "tools.strategic_action_plan\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.surgical_syntax_fixer_v2",
      "label": "surgical_syntax_fixer_v2",
      "size": 10,
      "color": "blue",
      "title": "tools.surgical_syntax_fixer_v2\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.surgical_syntax_fixer_v3",
      "label": "surgical_syntax_fixer_v3",
      "size": 10,
      "color": "blue",
      "title": "tools.surgical_syntax_fixer_v3\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.targeted_critical_syntax_fixer",
      "label": "targeted_critical_syntax_fixer",
      "size": 10,
      "color": "blue",
      "title": "tools.targeted_critical_syntax_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.text_heatmap_generator",
      "label": "text_heatmap_generator",
      "size": 10,
      "color": "blue",
      "title": "tools.text_heatmap_generator\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.utf8_encoding_fixer",
      "label": "utf8_encoding_fixer",
      "size": 10,
      "color": "blue",
      "title": "tools.utf8_encoding_fixer\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.utf8_terminal_cleaner",
      "label": "utf8_terminal_cleaner",
      "size": 10,
      "color": "blue",
      "title": "tools.utf8_terminal_cleaner\\nImports: 0\\nImported by: 0"
    },
    {
      "id": "tools.validate_tools_ast",
      "label": "validate_tools_ast",
      "size": 10,
      "color": "blue",
      "title": "tools.validate_tools_ast\\nImports: 0\\nImported by: 0"
    }
  ],
  "edges": [
    {
      "from": "advanced_syntax_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "advanced_syntax_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "advanced_syntax_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "advanced_syntax_validator",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "advanced_syntax_validator",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "advanced_syntax_validator",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "circular_import_analyzer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "circular_import_analyzer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "controlled_repair_analysis",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "controlled_string_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "controlled_string_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "controlled_string_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "controlled_syntax_scanner",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "final_comprehensive_syntax_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "final_comprehensive_syntax_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "final_comprehensive_syntax_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "phase3_fstring_bracket_fixes",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "phase3_fstring_bracket_fixes",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "precise_string_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "precise_string_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "precise_string_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "step2_indentation_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "step2_indentation_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "step2_indentation_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "surgical_indentation_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "surgical_indentation_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "surgical_indentation_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "ultra_precise_syntax_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "ultra_precise_syntax_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "ultra_precise_syntax_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "version_alignment_toolkit",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "version_alignment_toolkit",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "version_alignment_toolkit",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_generator",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_interrogative_pronouns_generator",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_normalizer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_phoneme_word_decision_tree",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_pronouns_generator",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_pronouns_generator",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_pronouns_generator",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_advanced_tester",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_advanced_tester",
      "to": "arabic_relative_pronouns_generator",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_analyzer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_analyzer",
      "to": "arabic_relative_pronouns_generator",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.arabic_relative_pronouns_analyzer",
      "to": "arabic_relative_pronouns_deep_model_simplified",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.batch_syntax_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.batch_syntax_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.comprehensive_arabic_phonological_system",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.comprehensive_arabic_phonological_system",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.conftest",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.create_working_engines",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.critical_syntax_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.debug_output",
      "to": "nlp",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.debug_output",
      "to": "arabic_nlp_status_report",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.deletion",
      "to": "backups.import_fixes_20250727_005716.advanced_arabic_engine",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.emergency_fstring_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.emergency_fstring_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.engine",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.engine_new",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.engine_new",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.final_arrow_syntax_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.final_arrow_syntax_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.final_arrow_syntax_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_arrow_comparisons",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_broken_comparisons",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_broken_comparisons",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_common_fstring_errors",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_common_fstring_errors",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_empty_imports",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_empty_imports",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_empty_imports",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_fstring_issues",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_logging_config",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.fix_logging_config",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.hafez_phonemes",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.hafez_syllables",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.inflection",
      "to": "phonology_core_unified",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.inflection",
      "to": "core",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.morphology",
      "to": "phonology_core_unified",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.morphology",
      "to": "core",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.pattern_embed",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.pattern_embed",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.phonology",
      "to": "phonology_core_unified",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.phonology",
      "to": "core",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.project_status_checker",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.quick_status",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.remove_arabic_forever",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.remove_arabic_forever",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.rule_base",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.run_syntax_fixes",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.run_syntax_fixes",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.segmenter",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.segmenter",
      "to": "backups.import_fixes_20250727_005716.advanced_arabic_engine",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.segmenter",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.simple_arrow_syntax_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.simple_arrow_syntax_fixer",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.simple_test",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.simple_test",
      "to": "arabic_nlp_status_report",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.simple_test",
      "to": "nlp",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.simple_text_cleaner",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.syllable_phonological_engine",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.syllable_phonological_engine",
      "to": "complete_arabic_phonological_foundation",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.syllable_phonological_engine",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.targeted_critical_syntax_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.targeted_syntax_scanner",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_core_integration",
      "to": "phonology_core_unified",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_core_integration",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_core_integration",
      "to": "core",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_morphological_weights",
      "to": "arabic_morphological_weight_generator",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_phoneme_integration",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_phoneme_integration",
      "to": "unified_phonemes",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_phoneme_processing",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_phoneme_simple",
      "to": "unified_phonemes",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_syllable_processing",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_syllable_readiness",
      "to": "syllable_phonological_engine",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_word",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_word",
      "to": "unified_phonemes",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_zero_layer_comprehensive",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_zero_layer_comprehensive",
      "to": "zero_layer_phonology",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.test_zero_layer_comprehensive",
      "to": "unified_phonemes",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.utf8_encoding_fixer",
      "to": "fix_logging_config",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.utf8_encoding_fixer",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.utf8_encoding_fixer",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.utf8_terminal_cleaner",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.validate_broken_files",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.validate_tools_ast",
      "to": "advanced_ast_syntax_fixer",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.violation_elimination_system",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.xampp_arabic_phonology",
      "to": "advanced_arabic_function_words_generator",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.xampp_arabic_phonology",
      "to": "arabic_inflection_corrected",
      "arrows": "to"
    },
    {
      "from": "backups.import_fixes_20250727_005716.xampp_test_phonemes_matrix",
      "to": "advanced_arabic_phonology_system",
      "arrows": "to"
    },
    {
      "from": "core.nlp.full_pipeline",
      "to": "phonology_core_unified",
      "arrows": "to"
    },
    {
      "from": "core.nlp.full_pipeline",
      "to": "core",
      "arrows": "to"
    },
    {
      "from": "core.nlp.phonological",
      "to": "phonology_core_unified",
      "arrows": "to"
    },
    {
      "from": "core.nlp.phonological",
      "to": "core",
      "arrows": "to"
    }
  ]
}
```

## 🔧 TOOLS AND TECHNIQUES

### Breaking Circular Imports:

1. **Dependency Injection:**
   ```python
   # Instead of direct import
   def process_data(processor=None):
       if processor is None:
           from .processor import DataProcessor
           processor = DataProcessor()
   ```

2. **Lazy Imports:**
   ```python
   def get_processor():
       from .processor import DataProcessor
       return DataProcessor()
   ```

3. **Interface Extraction:**
   ```python
   # Create abstract base class
   from abc import ABC, abstractmethod

   class ProcessorInterface(ABC):
       @abstractmethod
       def process(self, data): pass
   ```

4. **Module Restructuring:**
   - Extract common functionality into separate modules
   - Move shared constants/types to dedicated modules
   - Use configuration objects instead of direct imports

## ✅ NEXT STEPS

1. **Address circular imports** (if any) as highest priority
2. **Reduce coupling** in highly connected modules
3. **Extract interfaces** for stable module boundaries
4. **Consider architectural patterns** like dependency injection
5. **Use tools** like `import-linter` for ongoing monitoring

---

*Analysis completed successfully. Use this report to guide module refactoring efforts.*
