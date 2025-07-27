-- Database initialization script for local development

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create roles
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user WITH LOGIN PASSWORD 'app_password';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_admin') THEN
        CREATE ROLE app_admin WITH LOGIN PASSWORD 'admin_password' SUPERUSER;
    END IF;
END
$$;

-- Create database schema
CREATE SCHEMA IF NOT EXISTS arabic_nlp;

-- Set default permissions
ALTER DEFAULT PRIVILEGES IN SCHEMA arabic_nlp GRANT ALL ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA arabic_nlp GRANT ALL ON SEQUENCES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA arabic_nlp GRANT ALL ON FUNCTIONS TO app_user;

-- Grant privileges to existing objects
GRANT USAGE ON SCHEMA arabic_nlp TO app_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA arabic_nlp TO app_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA arabic_nlp TO app_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA arabic_nlp TO app_user;

-- Create tables
SET search_path TO arabic_nlp, public;

-- User management tables
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    key_prefix VARCHAR(8) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, key_name)
);

-- Arabic NLP Engine tables
CREATE TABLE IF NOT EXISTS processed_texts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    input_text TEXT NOT NULL,
    processing_type VARCHAR(50) NOT NULL,
    result JSONB NOT NULL,
    processing_time FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS roots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    root_text VARCHAR(10) NOT NULL UNIQUE,
    is_quad BOOLEAN NOT NULL DEFAULT FALSE,
    frequency INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_text VARCHAR(20) NOT NULL UNIQUE,
    pattern_type VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS derived_words (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    word_text VARCHAR(50) NOT NULL,
    root_id UUID REFERENCES roots(id) ON DELETE CASCADE,
    pattern_id UUID REFERENCES patterns(id) ON DELETE SET NULL,
    pos VARCHAR(20) NOT NULL,
    meaning TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(word_text, root_id)
);

-- Usage statistics tables
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    processing_time FLOAT NOT NULL,
    request_size INTEGER,
    response_size INTEGER,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_processed_texts_user_id ON processed_texts(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON api_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_derived_words_root_id ON derived_words(root_id);
CREATE INDEX IF NOT EXISTS idx_derived_words_pattern_id ON derived_words(pattern_id);
CREATE INDEX IF NOT EXISTS idx_derived_words_word_text ON derived_words(word_text);

-- Create a GIN index for trigram search on roots
CREATE INDEX IF NOT EXISTS idx_root_text_trgm ON roots USING gin (root_text gin_trgm_ops);

-- Create or replace function to update timestamps
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updating timestamps
DO $$
DECLARE
    table_names TEXT[] := ARRAY['users', 'api_keys', 'roots', 'patterns', 'derived_words'];
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY table_names LOOP
        EXECUTE format('
            DROP TRIGGER IF EXISTS update_%s_timestamp ON %s;
            CREATE TRIGGER update_%s_timestamp
            BEFORE UPDATE ON %s
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
        ', table_name, table_name, table_name, table_name);
    END LOOP;
END;
$$;

-- Insert a default admin user (username: admin, password: admin123)
INSERT INTO users (username, email, password_hash, is_admin)
VALUES ('admin', 'admin@example.com', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', TRUE)
ON CONFLICT (username) DO NOTHING;

-- Insert some sample roots for testing
INSERT INTO roots (root_text, is_quad, frequency) VALUES
('كتب', FALSE, 1000),
('قرأ', FALSE, 950),
('علم', FALSE, 900),
('دخل', FALSE, 850),
('خرج', FALSE, 800),
('فهم', FALSE, 750),
('دعو', FALSE, 700),
('جلس', FALSE, 650),
('قوم', FALSE, 600),
('زرع', FALSE, 550)
ON CONFLICT (root_text) DO NOTHING;

-- Insert some sample patterns
INSERT INTO patterns (pattern_text, pattern_type, description) VALUES
('فَاعِل', 'اسم فاعل', 'Active participle'),
('مَفْعُول', 'اسم مفعول', 'Passive participle'),
('فَعَّال', 'صيغة مبالغة', 'Intensive active participle'),
('مِفْعَال', 'اسم آلة', 'Instrumental noun'),
('مَفْعَل', 'اسم مكان', 'Noun of place')
ON CONFLICT (pattern_text) DO NOTHING;

-- Create a function to check database health
CREATE OR REPLACE FUNCTION check_database_health()
RETURNS TABLE (status TEXT, details JSONB) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'healthy'::TEXT AS status,
        jsonb_build_object(
            'database_size', pg_size_pretty(pg_database_size(current_database())),
            'table_count', (SELECT COUNT(*)::TEXT FROM information_schema.tables WHERE table_schema = 'arabic_nlp'),
            'user_count', (SELECT COUNT(*)::TEXT FROM users),
            'version', version()
        ) AS details;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the health check function
GRANT EXECUTE ON FUNCTION check_database_health() TO app_user;
