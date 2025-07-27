
-- =============================================================================
-- Arabic Morphophonological Project - Database Schema
-- Generated on: 2025-07-20
-- Version: 2.0.0 (Redesigned Architecture)
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =============================================================================
-- 1. ARABIC ROOTS SCHEMA
-- =============================================================================

-- Main roots table
CREATE TABLE arabic_roots (
    root_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    root_text VARCHAR(10) NOT NULL UNIQUE,
    root_type VARCHAR(20) CHECK (root_type IN ('trilateral', 'quadrilateral', 'quinqueliteral')) NOT NULL,
    semantic_field VARCHAR(100),
    frequency_score INTEGER DEFAULT 0,
    weakness_type VARCHAR(20) CHECK (weakness_type IN ('sound', 'assimilated', 'hollow', 'defective', 'hamzated')) DEFAULT 'sound',
    phonological_class VARCHAR(50),
    semantic_vector REAL[] DEFAULT '{}',
    phonetic_features JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    
    -- Constraints
    CONSTRAINT chk_root_text_length CHECK (length(root_text) >= 2 AND length(root_text) <= 5),
    CONSTRAINT chk_frequency_score CHECK (frequency_score >= 0)
);

-- Root radicals (normalized)
CREATE TABLE root_radicals (
    radical_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    root_id UUID REFERENCES arabic_roots(root_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 1 AND position <= 5),
    radical_char CHAR(1) NOT NULL,
    phonetic_class VARCHAR(20) CHECK (phonetic_class IN ('consonant', 'weak', 'emphatic', 'hamza')) NOT NULL,
    articulatory_features JSONB DEFAULT '{}',
    is_geminated BOOLEAN DEFAULT FALSE,
    
    UNIQUE(root_id, position)
);

-- Indexes for roots
CREATE INDEX idx_roots_text ON arabic_roots USING btree(root_text);
CREATE INDEX idx_roots_type ON arabic_roots USING btree(root_type);
CREATE INDEX idx_roots_semantic ON arabic_roots USING btree(semantic_field);
CREATE INDEX idx_roots_frequency ON arabic_roots USING btree(frequency_score DESC);
CREATE INDEX idx_roots_weakness ON arabic_roots USING btree(weakness_type);
CREATE INDEX idx_roots_updated ON arabic_roots USING btree(updated_at DESC);

CREATE INDEX idx_radicals_root_position ON root_radicals USING btree(root_id, position);
CREATE INDEX idx_radicals_char ON root_radicals USING btree(radical_char);
CREATE INDEX idx_radicals_class ON root_radicals USING btree(phonetic_class);

-- =============================================================================
-- 2. MORPHOLOGICAL PATTERNS SCHEMA
-- =============================================================================

-- Pattern templates
CREATE TABLE morphological_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_name VARCHAR(50) NOT NULL,
    pattern_template VARCHAR(20) NOT NULL,
    cv_structure VARCHAR(20) NOT NULL,
    pattern_type VARCHAR(20) CHECK (pattern_type IN ('verb', 'noun', 'adjective', 'masdar', 'participle')) NOT NULL,
    verb_form INTEGER CHECK (verb_form >= 1 AND verb_form <= 15),
    frequency_rank INTEGER DEFAULT 0,
    semantic_function TEXT,
    morphological_category VARCHAR(50),
    transformation_rules JSONB DEFAULT '{}',
    compatibility_matrix JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_verb_form_type CHECK (
        (pattern_type = 'verb' AND verb_form IS NOT NULL) OR 
        (pattern_type != 'verb' AND verb_form IS NULL)
    )
);

-- Pattern morphemes (decomposed)
CREATE TABLE pattern_morphemes (
    morpheme_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID REFERENCES morphological_patterns(pattern_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 1),
    morpheme_type VARCHAR(20) CHECK (morpheme_type IN ('root_consonant', 'vowel', 'prefix', 'suffix', 'infix')) NOT NULL,
    morpheme_value VARCHAR(5) NOT NULL,
    is_templatic BOOLEAN DEFAULT FALSE,
    phonological_weight REAL DEFAULT 1.0,
    
    UNIQUE(pattern_id, position)
);

-- Indexes for patterns
CREATE INDEX idx_patterns_type ON morphological_patterns USING btree(pattern_type);
CREATE INDEX idx_patterns_verb_form ON morphological_patterns USING btree(verb_form);
CREATE INDEX idx_patterns_frequency ON morphological_patterns USING btree(frequency_rank DESC);
CREATE INDEX idx_patterns_template ON morphological_patterns USING btree(pattern_template);
CREATE INDEX idx_patterns_cv ON morphological_patterns USING btree(cv_structure);

CREATE INDEX idx_morphemes_pattern_position ON pattern_morphemes USING btree(pattern_id, position);
CREATE INDEX idx_morphemes_type ON pattern_morphemes USING btree(morpheme_type);

-- =============================================================================
-- 3. PHONOLOGICAL RULES SCHEMA
-- =============================================================================

-- Phonological rules engine
CREATE TABLE phonological_rules (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_name VARCHAR(100) NOT NULL,
    rule_category VARCHAR(20) CHECK (rule_category IN ('assimilation', 'deletion', 'insertion', 'metathesis', 'gemination', 'vowel_harmony')) NOT NULL,
    rule_scope VARCHAR(20) CHECK (rule_scope IN ('local', 'global', 'morpheme_boundary', 'word_boundary', 'syllable_boundary')) NOT NULL,
    input_pattern VARCHAR(100) NOT NULL,
    output_pattern VARCHAR(100) NOT NULL,
    conditioning_environment VARCHAR(200),
    rule_priority INTEGER DEFAULT 0,
    is_optional BOOLEAN DEFAULT FALSE,
    dialectal_variation VARCHAR(20) DEFAULT 'standard',
    rule_formula TEXT,
    examples JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_rule_priority CHECK (rule_priority >= 0 AND rule_priority <= 100)
);

-- Phoneme inventory
CREATE TABLE phonemes (
    phoneme_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phoneme_symbol VARCHAR(5) NOT NULL UNIQUE,
    phoneme_class VARCHAR(20) CHECK (phoneme_class IN ('consonant', 'vowel', 'diphthong', 'geminate')) NOT NULL,
    articulatory_features JSONB DEFAULT '{}',
    acoustic_features JSONB DEFAULT '{}',
    frequency_score REAL DEFAULT 0.0,
    distribution_contexts JSONB DEFAULT '{}',
    allophonic_variants JSONB DEFAULT '[]',
    
    -- Constraints
    CONSTRAINT chk_frequency_positive CHECK (frequency_score >= 0.0)
);

-- Indexes for phonological data
CREATE INDEX idx_rules_category ON phonological_rules USING btree(rule_category);
CREATE INDEX idx_rules_scope ON phonological_rules USING btree(rule_scope);
CREATE INDEX idx_rules_priority ON phonological_rules USING btree(rule_priority DESC);
CREATE INDEX idx_rules_dialect ON phonological_rules USING btree(dialectal_variation);

CREATE INDEX idx_phonemes_class ON phonemes USING btree(phoneme_class);
CREATE INDEX idx_phonemes_frequency ON phonemes USING btree(frequency_score DESC);
CREATE INDEX idx_phonemes_symbol ON phonemes USING btree(phoneme_symbol);

-- =============================================================================
-- 4. SYLLABLE STRUCTURE SCHEMA
-- =============================================================================

-- Syllable templates
CREATE TABLE syllable_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    syllable_type VARCHAR(10) CHECK (syllable_type IN ('CV', 'CVC', 'CVV', 'CVVC', 'CVCC', 'CVCCC')) NOT NULL,
    position VARCHAR(15) CHECK (position IN ('initial', 'medial', 'final', 'monosyllabic')) NOT NULL,
    stress_pattern VARCHAR(15) CHECK (stress_pattern IN ('primary', 'secondary', 'unstressed')) DEFAULT 'unstressed',
    frequency_weight REAL DEFAULT 1.0,
    phonotactic_constraints JSONB DEFAULT '{}',
    is_permitted BOOLEAN DEFAULT TRUE,
    
    UNIQUE(syllable_type, position)
);

-- Word syllabification
CREATE TABLE word_syllables (
    syllable_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    word_id UUID NOT NULL, -- references generated_words
    syllable_position INTEGER NOT NULL CHECK (syllable_position >= 1),
    syllable_content VARCHAR(10) NOT NULL,
    template_id UUID REFERENCES syllable_templates(template_id),
    stress_level VARCHAR(15) CHECK (stress_level IN ('primary', 'secondary', 'unstressed')) DEFAULT 'unstressed',
    onset VARCHAR(5) DEFAULT '',
    nucleus VARCHAR(5) NOT NULL,
    coda VARCHAR(5) DEFAULT '',
    
    UNIQUE(word_id, syllable_position)
);

-- Indexes for syllables
CREATE INDEX idx_syllable_templates_type ON syllable_templates USING btree(syllable_type);
CREATE INDEX idx_syllable_templates_position ON syllable_templates USING btree(position);
CREATE INDEX idx_syllable_templates_weight ON syllable_templates USING btree(frequency_weight DESC);

CREATE INDEX idx_word_syllables_word_position ON word_syllables USING btree(word_id, syllable_position);
CREATE INDEX idx_word_syllables_stress ON word_syllables USING btree(stress_level);

-- =============================================================================
-- 5. GENERATED WORDS SCHEMA
-- =============================================================================

-- Generated morphological words
CREATE TABLE generated_words (
    word_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    root_id UUID REFERENCES arabic_roots(root_id) ON DELETE CASCADE,
    pattern_id UUID REFERENCES morphological_patterns(pattern_id) ON DELETE CASCADE,
    surface_form VARCHAR(50) NOT NULL,
    underlying_form VARCHAR(50) NOT NULL,
    phonetic_transcription VARCHAR(100),
    syllable_structure VARCHAR(30),
    stress_pattern VARCHAR(20),
    morphological_analysis JSONB DEFAULT '{}',
    semantic_analysis JSONB DEFAULT '{}',
    phonological_derivation JSONB DEFAULT '[]',
    generation_metadata JSONB DEFAULT '{}',
    confidence_score REAL DEFAULT 0.0 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified TIMESTAMP,
    verification_status VARCHAR(20) DEFAULT 'unverified' CHECK (verification_status IN ('verified', 'unverified', 'disputed')),
    
    -- Constraints
    CONSTRAINT chk_forms_not_empty CHECK (length(surface_form) > 0 AND length(underlying_form) > 0)
);

-- Applied phonological rules (audit trail)
CREATE TABLE applied_rules_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    word_id UUID REFERENCES generated_words(word_id) ON DELETE CASCADE,
    rule_id UUID REFERENCES phonological_rules(rule_id) ON DELETE CASCADE,
    application_order INTEGER NOT NULL CHECK (application_order >= 1),
    input_form VARCHAR(50) NOT NULL,
    output_form VARCHAR(50) NOT NULL,
    rule_context VARCHAR(100),
    environment_matched VARCHAR(100),
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(word_id, application_order)
);

-- Indexes for generated words
CREATE INDEX idx_generated_words_root_pattern ON generated_words USING btree(root_id, pattern_id);
CREATE INDEX idx_generated_words_surface ON generated_words USING gin(surface_form gin_trgm_ops);
CREATE INDEX idx_generated_words_confidence ON generated_words USING btree(confidence_score DESC);
CREATE INDEX idx_generated_words_timestamp ON generated_words USING btree(generation_timestamp DESC);
CREATE INDEX idx_generated_words_verification ON generated_words USING btree(verification_status);

CREATE INDEX idx_applied_rules_word_order ON applied_rules_log USING btree(word_id, application_order);
CREATE INDEX idx_applied_rules_rule ON applied_rules_log USING btree(rule_id);
CREATE INDEX idx_applied_rules_timestamp ON applied_rules_log USING btree(applied_at DESC);

-- =============================================================================
-- 6. ANALYTICS & CACHING SCHEMA
-- =============================================================================

-- Usage analytics
CREATE TABLE usage_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(100),
    user_id VARCHAR(100),
    endpoint VARCHAR(100) NOT NULL,
    request_data JSONB DEFAULT '{}',
    response_data JSONB DEFAULT '{}',
    processing_time_ms INTEGER DEFAULT 0,
    status_code INTEGER DEFAULT 200,
    error_message TEXT,
    client_ip INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Partitioning by month
    PARTITION BY RANGE (timestamp)
);

-- Request cache
CREATE TABLE request_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    cache_value JSONB NOT NULL,
    cache_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    hit_count INTEGER DEFAULT 0,
    
    -- Constraint
    CONSTRAINT chk_expires_future CHECK (expires_at > created_at)
);

-- Performance metrics
CREATE TABLE performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit VARCHAR(20),
    service_name VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    additional_data JSONB DEFAULT '{}'
);

-- Indexes for analytics
CREATE INDEX idx_usage_analytics_timestamp ON usage_analytics USING btree(timestamp DESC);
CREATE INDEX idx_usage_analytics_endpoint ON usage_analytics USING btree(endpoint);
CREATE INDEX idx_usage_analytics_status ON usage_analytics USING btree(status_code);
CREATE INDEX idx_usage_analytics_user ON usage_analytics USING btree(user_id) WHERE user_id IS NOT NULL;

CREATE INDEX idx_request_cache_expires ON request_cache USING btree(expires_at);
CREATE INDEX idx_request_cache_created ON request_cache USING btree(created_at DESC);

CREATE INDEX idx_performance_metrics_name_time ON performance_metrics USING btree(metric_name, timestamp DESC);
CREATE INDEX idx_performance_metrics_service ON performance_metrics USING btree(service_name);

-- =============================================================================
-- 7. TRIGGERS AND FUNCTIONS
-- =============================================================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_arabic_roots_updated_at 
    BEFORE UPDATE ON arabic_roots 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Cache cleanup function
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM request_cache WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ language 'plpgsql';

-- Analytics partition maintenance
CREATE OR REPLACE FUNCTION create_monthly_analytics_partition(start_date DATE)
RETURNS void AS $$
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := 'usage_analytics_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF usage_analytics 
                    FOR VALUES FROM (%L) TO (%L)', 
                   partition_name, start_date, end_date);
                   
    EXECUTE format('CREATE INDEX %I ON %I (timestamp DESC)', 
                   'idx_' || partition_name || '_timestamp', partition_name);
END;
$$ language 'plpgsql';

-- =============================================================================
-- 8. INITIAL DATA SETUP
-- =============================================================================

-- Insert basic syllable templates
INSERT INTO syllable_templates (syllable_type, position, frequency_weight) VALUES
('CV', 'initial', 3.0),
('CV', 'medial', 4.0),
('CV', 'final', 2.0),
('CVC', 'initial', 2.5),
('CVC', 'medial', 3.5),
('CVC', 'final', 4.5),
('CVV', 'final', 1.5),
('CVVC', 'final', 1.0);

-- Insert basic phonemes
INSERT INTO phonemes (phoneme_symbol, phoneme_class, articulatory_features, frequency_score) VALUES
('ب', 'consonant', '{"manner": "stop", "place": "bilabial", "voice": "voiced"}', 8.5),
('ت', 'consonant', '{"manner": "stop", "place": "dental", "voice": "voiceless"}', 7.2),
('ث', 'consonant', '{"manner": "fricative", "place": "dental", "voice": "voiceless"}', 3.1),
('ج', 'consonant', '{"manner": "affricate", "place": "alveopalatal", "voice": "voiced"}', 5.8),
('ح', 'consonant', '{"manner": "fricative", "place": "pharyngeal", "voice": "voiceless"}', 6.4),
('خ', 'consonant', '{"manner": "fricative", "place": "velar", "voice": "voiceless"}', 4.2),
('د', 'consonant', '{"manner": "stop", "place": "dental", "voice": "voiced"}', 7.8),
('ذ', 'consonant', '{"manner": "fricative", "place": "dental", "voice": "voiced"}', 3.6),
('ر', 'consonant', '{"manner": "trill", "place": "alveolar", "voice": "voiced"}', 9.1),
('ز', 'consonant', '{"manner": "fricative", "place": "alveolar", "voice": "voiced"}', 4.7),
('س', 'consonant', '{"manner": "fricative", "place": "alveolar", "voice": "voiceless"}', 8.3),
('ش', 'consonant', '{"manner": "fricative", "place": "alveopalatal", "voice": "voiceless"}', 5.9),
('ص', 'consonant', '{"manner": "fricative", "place": "alveolar", "voice": "voiceless", "emphatic": true}', 6.7),
('ض', 'consonant', '{"manner": "stop", "place": "dental", "voice": "voiced", "emphatic": true}', 4.1),
('ط', 'consonant', '{"manner": "stop", "place": "dental", "voice": "voiceless", "emphatic": true}', 5.3),
('ظ', 'consonant', '{"manner": "fricative", "place": "dental", "voice": "voiced", "emphatic": true}', 2.8),
('ع', 'consonant', '{"manner": "fricative", "place": "pharyngeal", "voice": "voiced"}', 7.6),
('غ', 'consonant', '{"manner": "fricative", "place": "velar", "voice": "voiced"}', 4.9),
('ف', 'consonant', '{"manner": "fricative", "place": "labiodental", "voice": "voiceless"}', 8.7),
('ق', 'consonant', '{"manner": "stop", "place": "uvular", "voice": "voiceless"}', 6.2),
('ك', 'consonant', '{"manner": "stop", "place": "velar", "voice": "voiceless"}', 9.4),
('ل', 'consonant', '{"manner": "lateral", "place": "alveolar", "voice": "voiced"}', 9.8),
('م', 'consonant', '{"manner": "nasal", "place": "bilabial", "voice": "voiced"}', 9.2),
('ن', 'consonant', '{"manner": "nasal", "place": "alveolar", "voice": "voiced"}', 9.6),
('ه', 'consonant', '{"manner": "fricative", "place": "glottal", "voice": "voiceless"}', 7.3),
('و', 'consonant', '{"manner": "approximant", "place": "labial", "voice": "voiced"}', 8.1),
('ي', 'consonant', '{"manner": "approximant", "place": "palatal", "voice": "voiced"}', 8.9),
('ء', 'consonant', '{"manner": "stop", "place": "glottal", "voice": "voiceless"}', 5.4),
('َ', 'vowel', '{"height": "low", "backness": "central", "length": "short"}', 15.2),
('ُ', 'vowel', '{"height": "high", "backness": "back", "length": "short"}', 12.8),
('ِ', 'vowel', '{"height": "high", "backness": "front", "length": "short"}', 14.1),
('ا', 'vowel', '{"height": "low", "backness": "central", "length": "long"}', 18.7),
('و', 'vowel', '{"height": "high", "backness": "back", "length": "long"}', 11.3),
('ي', 'vowel', '{"height": "high", "backness": "front", "length": "long"}', 13.6);

-- Create current month partition
SELECT create_monthly_analytics_partition(date_trunc('month', CURRENT_DATE));

-- =============================================================================
-- 9. VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Comprehensive root view
CREATE VIEW v_roots_complete AS
SELECT 
    ar.root_id,
    ar.root_text,
    ar.root_type,
    ar.semantic_field,
    ar.weakness_type,
    ar.frequency_score,
    array_agg(rr.radical_char ORDER BY rr.position) as radicals,
    array_agg(rr.phonetic_class ORDER BY rr.position) as radical_classes,
    COUNT(gw.word_id) as generated_words_count,
    AVG(gw.confidence_score) as avg_confidence
FROM arabic_roots ar
LEFT JOIN root_radicals rr ON ar.root_id = rr.root_id
LEFT JOIN generated_words gw ON ar.root_id = gw.root_id
GROUP BY ar.root_id, ar.root_text, ar.root_type, ar.semantic_field, 
         ar.weakness_type, ar.frequency_score;

-- Pattern usage statistics
CREATE VIEW v_pattern_statistics AS
SELECT 
    mp.pattern_id,
    mp.pattern_name,
    mp.pattern_template,
    mp.pattern_type,
    mp.verb_form,
    COUNT(gw.word_id) as usage_count,
    AVG(gw.confidence_score) as avg_confidence,
    MAX(gw.generation_timestamp) as last_used
FROM morphological_patterns mp
LEFT JOIN generated_words gw ON mp.pattern_id = gw.pattern_id
GROUP BY mp.pattern_id, mp.pattern_name, mp.pattern_template, 
         mp.pattern_type, mp.verb_form;

-- Phonological rule effectiveness
CREATE VIEW v_rule_effectiveness AS
SELECT 
    pr.rule_id,
    pr.rule_name,
    pr.rule_category,
    COUNT(arl.log_id) as application_count,
    COUNT(DISTINCT arl.word_id) as words_affected,
    AVG(CASE WHEN arl.input_form != arl.output_form THEN 1 ELSE 0 END) as effectiveness_ratio,
    MAX(arl.applied_at) as last_applied
FROM phonological_rules pr
LEFT JOIN applied_rules_log arl ON pr.rule_id = arl.rule_id
GROUP BY pr.rule_id, pr.rule_name, pr.rule_category;

-- =============================================================================
-- 10. PERFORMANCE OPTIMIZATION
-- =============================================================================

-- Analyze tables for query optimization
ANALYZE arabic_roots;
ANALYZE root_radicals;
ANALYZE morphological_patterns;
ANALYZE pattern_morphemes;
ANALYZE phonological_rules;
ANALYZE phonemes;
ANALYZE syllable_templates;
ANALYZE word_syllables;
ANALYZE generated_words;
ANALYZE applied_rules_log;

COMMIT;

-- =============================================================================
-- END OF SCHEMA CREATION
-- =============================================================================
