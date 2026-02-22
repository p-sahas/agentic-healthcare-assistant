"""
Dynamic Supabase schema generator - reads dimensions from config.

This ensures the database schema always matches config.EMBEDDING_DIM
without any hardcoded values in SQL files.
"""

from infrastructure.config import EMBEDDING_DIM, EMBEDDING_MODEL


def generate_supabase_schema() -> str:
    """
    Generate Supabase schema DDL dynamically from config.
    
    Returns:
        SQL DDL string with vector dimensions from config.EMBEDDING_DIM
    """
    
    return f"""-- ============================================================================
-- Supabase Schema: Memory System + CRM
-- PostgreSQL 15+ with pgvector extension
-- ============================================================================
-- 
--  DYNAMICALLY GENERATED FROM CONFIG
-- Embedding Model: {EMBEDDING_MODEL}
-- Vector Dimensions: {EMBEDDING_DIM}
-- 
-- This schema is generated programmatically to ensure dimensions
-- always match config.EMBEDDING_DIM (single source of truth).
-- 
-- ============================================================================

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SHORT-TERM MEMORY (Supabase backend — ring buffer with TTL)
-- ============================================================================

CREATE TABLE IF NOT EXISTS st_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ttl_at TIMESTAMPTZ  -- Auto-cleanup after this time (default 24h from created_at)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_st_turns_user_session ON st_turns (user_id, session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_st_turns_ttl ON st_turns (ttl_at) WHERE ttl_at IS NOT NULL;

COMMENT ON TABLE st_turns IS 'Short-term conversation memory — ring buffer with TTL';

-- ============================================================================
-- LONG-TERM SEMANTIC MEMORY (pgvector facts)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mem_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}),
    score REAL NOT NULL CHECK (score >= 0 AND score <= 1),
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ DEFAULT NOW(),
    ttl_at TIMESTAMPTZ,
    pin BOOLEAN DEFAULT FALSE,
    deleted BOOLEAN DEFAULT FALSE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_mem_facts_user_id ON mem_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_facts_score ON mem_facts(score DESC);
CREATE INDEX IF NOT EXISTS idx_mem_facts_deleted ON mem_facts(deleted) WHERE deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mem_facts_ttl ON mem_facts(ttl_at) WHERE ttl_at IS NOT NULL;

-- pgvector index (IVFFlat supports higher dimensions, HNSW limited to 2000)
CREATE INDEX IF NOT EXISTS idx_mem_facts_embedding 
ON mem_facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Helper function for semantic search
CREATE OR REPLACE FUNCTION search_mem_facts(
    query_embedding vector({EMBEDDING_DIM}),
    query_user_id TEXT,
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    user_id TEXT,
    text TEXT,
    score REAL,
    tags JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.id,
        f.user_id,
        f.text,
        f.score,
        f.tags,
        1 - (f.embedding <=> query_embedding) AS similarity
    FROM mem_facts f
    WHERE f.user_id = query_user_id
        AND f.deleted = FALSE
        AND (f.ttl_at IS NULL OR f.ttl_at > NOW())
        AND 1 - (f.embedding <=> query_embedding) >= match_threshold
    ORDER BY f.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- LONG-TERM EPISODIC MEMORY (pgvector episodes)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mem_episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    summary_embedding vector({EMBEDDING_DIM}),
    topic_tags JSONB DEFAULT '[]'::jsonb,
    start_at TIMESTAMPTZ NOT NULL,
    end_at TIMESTAMPTZ NOT NULL,
    turn_count INTEGER NOT NULL CHECK (turn_count > 0),
    turns JSONB NOT NULL,  -- Full conversation as JSON array
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_mem_episodes_user_id ON mem_episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_episodes_session_id ON mem_episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_mem_episodes_start_at ON mem_episodes(start_at DESC);
CREATE INDEX IF NOT EXISTS idx_mem_episodes_created_at ON mem_episodes(created_at DESC);

-- pgvector index
CREATE INDEX IF NOT EXISTS idx_mem_episodes_embedding 
ON mem_episodes USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);

-- Helper function for semantic search
CREATE OR REPLACE FUNCTION search_mem_episodes(
    query_embedding vector({EMBEDDING_DIM}),
    query_user_id TEXT,
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    user_id TEXT,
    session_id TEXT,
    summary TEXT,
    topic_tags JSONB,
    start_at TIMESTAMPTZ,
    end_at TIMESTAMPTZ,
    turn_count INTEGER,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.user_id,
        e.session_id,
        e.summary,
        e.topic_tags,
        e.start_at,
        e.end_at,
        e.turn_count,
        1 - (e.summary_embedding <=> query_embedding) AS similarity
    FROM mem_episodes e
    WHERE e.user_id = query_user_id
        AND 1 - (e.summary_embedding <=> query_embedding) >= match_threshold
    ORDER BY e.summary_embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- NOTE: RAG document chunks are stored in Qdrant Cloud (not Supabase).
-- ============================================================================

-- ============================================================================
-- PROCEDURAL MEMORY (How-to knowledge - workflows and procedures)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mem_procedures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    context_when TEXT,  -- When to use this procedure
    steps JSONB NOT NULL,  -- Array of ordered steps
    conditions JSONB,  -- Preconditions, constraints
    examples JSONB,  -- Example usage scenarios
    embedding vector({EMBEDDING_DIM}),  -- For semantic retrieval
    category TEXT,  -- e.g., 'booking', 'patient_care', 'administrative'
    active BOOLEAN DEFAULT TRUE,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_mem_procedures_name ON mem_procedures(name);
CREATE INDEX IF NOT EXISTS idx_mem_procedures_category ON mem_procedures(category);
CREATE INDEX IF NOT EXISTS idx_mem_procedures_active ON mem_procedures(active) WHERE active = TRUE;

-- pgvector index for semantic retrieval
CREATE INDEX IF NOT EXISTS idx_mem_procedures_embedding 
ON mem_procedures USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- Helper function for semantic procedure search
CREATE OR REPLACE FUNCTION search_mem_procedures(
    query_embedding vector({EMBEDDING_DIM}),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    name TEXT,
    description TEXT,
    steps JSONB,
    category TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id,
        p.name,
        p.description,
        p.steps,
        p.category,
        1 - (p.embedding <=> query_embedding) AS similarity
    FROM mem_procedures p
    WHERE p.active = TRUE
        AND 1 - (p.embedding <=> query_embedding) >= match_threshold
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE mem_procedures IS 'Procedural memory - workflows and step-by-step procedures for task execution';

-- ============================================================================
-- CRM: LOCATIONS (matches ORM: Location.__tablename__ = "locations")
-- ============================================================================

CREATE TABLE IF NOT EXISTS locations (
    location_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('HOSPITAL','OPD','LAB','CLINIC')),
    address TEXT,
    tz TEXT NOT NULL,  -- IANA timezone
    lat REAL,
    lng REAL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- ============================================================================
-- CRM: SPECIALTIES (matches ORM: Specialty.__tablename__ = "specialties")
-- ============================================================================

CREATE TABLE IF NOT EXISTS specialties (
    specialty_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- ============================================================================
-- CRM: DOCTORS (matches ORM: Doctor.__tablename__ = "doctors")
-- ============================================================================

CREATE TABLE IF NOT EXISTS doctors (
    doctor_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    specialty_id TEXT REFERENCES specialties(specialty_id) ON DELETE SET NULL,
    license_no TEXT UNIQUE,
    phone TEXT,
    email TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doctors_specialty ON doctors(specialty_id);

-- ============================================================================
-- CRM: PATIENTS (matches ORM: Patient.__tablename__ = "patients")
-- ============================================================================

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    external_user_id TEXT NOT NULL UNIQUE,  -- Phone number without '+'
    full_name TEXT NOT NULL,
    dob TEXT,  -- ISO format YYYY-MM-DD
    gender TEXT CHECK (gender IN ('M','F','X')),
    phone TEXT,
    email TEXT,
    notes TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_patients_external_user_id ON patients(external_user_id);
CREATE INDEX IF NOT EXISTS idx_patients_phone ON patients(phone);

-- ============================================================================
-- CRM: BOOKINGS (matches ORM: Booking.__tablename__ = "bookings")
-- ============================================================================

CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id) ON DELETE CASCADE,
    location_id TEXT NOT NULL REFERENCES locations(location_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    reason TEXT,
    start_at INTEGER NOT NULL,  -- Epoch seconds UTC
    end_at INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','CONFIRMED','RESCHEDULED','CANCELLED','NO_SHOW','COMPLETED')),
    source TEXT NOT NULL CHECK (source IN ('CRM','MEMORY','MIGRATED','SEED')),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bookings_patient ON bookings(patient_id);
CREATE INDEX IF NOT EXISTS idx_bookings_doctor ON bookings(doctor_id);
CREATE INDEX IF NOT EXISTS idx_bookings_location ON bookings(location_id);
CREATE INDEX IF NOT EXISTS idx_bookings_start_at ON bookings(start_at DESC);
CREATE INDEX IF NOT EXISTS idx_bookings_status ON bookings(status);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) - Production Ready
-- ============================================================================

-- Enable RLS on memory tables
ALTER TABLE mem_facts ENABLE ROW LEVEL SECURITY;
ALTER TABLE mem_episodes ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotent re-runs)
DROP POLICY IF EXISTS "Users can view their own facts" ON mem_facts;
DROP POLICY IF EXISTS "Users can manage their own facts" ON mem_facts;
DROP POLICY IF EXISTS "Users can view their own episodes" ON mem_episodes;
DROP POLICY IF EXISTS "Users can manage their own episodes" ON mem_episodes;

-- Policies: Users can only access their own memory
CREATE POLICY "Users can view their own facts"
    ON mem_facts FOR SELECT
    USING (user_id = current_setting('app.user_id', TRUE));

CREATE POLICY "Users can manage their own facts"
    ON mem_facts FOR ALL
    USING (user_id = current_setting('app.user_id', TRUE));

CREATE POLICY "Users can view their own episodes"
    ON mem_episodes FOR SELECT
    USING (user_id = current_setting('app.user_id', TRUE));

CREATE POLICY "Users can manage their own episodes"
    ON mem_episodes FOR ALL
    USING (user_id = current_setting('app.user_id', TRUE));

-- CRM tables: No RLS for now (can be added based on requirements)

-- ============================================================================
-- VIEWS FOR ANALYTICS (Optional but useful)
-- ============================================================================

-- Active memory facts per user
CREATE OR REPLACE VIEW v_active_facts AS
SELECT 
    user_id,
    COUNT(*) AS total_facts,
    AVG(score) AS avg_score,
    COUNT(*) FILTER (WHERE pin = TRUE) AS pinned_facts
FROM mem_facts
WHERE deleted = FALSE 
    AND (ttl_at IS NULL OR ttl_at > NOW())
GROUP BY user_id;

-- Episode statistics
CREATE OR REPLACE VIEW v_episode_stats AS
SELECT 
    user_id,
    COUNT(*) AS total_episodes,
    SUM(turn_count) AS total_turns,
    AVG(turn_count) AS avg_turns_per_episode,
    MAX(created_at) AS last_episode_at
FROM mem_episodes
GROUP BY user_id;

-- Upcoming appointments (CRM)
CREATE OR REPLACE VIEW v_upcoming_appointments AS
SELECT 
    b.booking_id,
    p.full_name AS patient_name,
    p.phone AS patient_phone,
    d.full_name AS doctor_name,
    s.name AS specialty,
    l.name AS location,
    b.start_at,
    b.end_at,
    b.status,
    b.reason
FROM bookings b
JOIN patients p ON b.patient_id = p.patient_id
JOIN doctors d ON b.doctor_id = d.doctor_id
JOIN locations l ON b.location_id = l.location_id
LEFT JOIN specialties s ON d.specialty_id = s.specialty_id
WHERE b.start_at >= EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::INTEGER
    AND b.status IN ('CONFIRMED', 'PENDING')
ORDER BY b.start_at;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE mem_facts IS 'Long-term semantic memory facts with pgvector embeddings';
COMMENT ON TABLE mem_episodes IS 'Long-term episodic memory — full conversations with pgvector summaries';
COMMENT ON TABLE patients IS 'CRM patient records';
COMMENT ON TABLE doctors IS 'CRM doctor records';
COMMENT ON TABLE bookings IS 'CRM appointment bookings';
COMMENT ON TABLE locations IS 'CRM healthcare locations';
COMMENT ON TABLE specialties IS 'Medical specialties';

COMMENT ON FUNCTION search_mem_facts IS 'Semantic search over memory facts using cosine similarity';
COMMENT ON FUNCTION search_mem_episodes IS 'Semantic search over episode summaries using cosine similarity';

-- ============================================================================
-- COMPLETION
-- ============================================================================

-- Verify installation
DO $$
BEGIN
    RAISE NOTICE ' Supabase schema created successfully!';
    RAISE NOTICE ' Tables created: st_turns, mem_facts, mem_episodes, mem_procedures, crm_*';
    RAISE NOTICE ' pgvector indexes created with IVFFlat (cosine similarity)';
    RAISE NOTICE ' Model: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dims)';
    RAISE NOTICE ' Row Level Security (RLS) enabled for memory tables';
    RAISE NOTICE ' Short-term memory: Supabase (st_turns table)';
    RAISE NOTICE ' Memory types: Short-term, Semantic, Episodic, Procedural';
    RAISE NOTICE ' Ready for production use!';
END $$;
"""
