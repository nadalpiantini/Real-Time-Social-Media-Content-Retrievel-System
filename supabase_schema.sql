-- SMContent Database Schema for Supabase
-- Prefix: smc_ para evitar conflictos en base compartida

-- Tabla para almacenar fuentes/cuentas de donde se scrapea
CREATE TABLE smc_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    platform VARCHAR(50) NOT NULL DEFAULT 'Linkedin',
    url TEXT,
    last_scraped TIMESTAMP,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tabla principal para posts
CREATE TABLE smc_posts (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(255) NOT NULL,
    source_id INTEGER REFERENCES smc_sources(id) ON DELETE CASCADE,
    source_name VARCHAR(255) NOT NULL, -- Redundante pero útil para queries
    platform VARCHAR(50) NOT NULL DEFAULT 'Linkedin',
    text TEXT NOT NULL,
    full_raw_text TEXT, -- Para compatibilidad con sistema actual
    post_owner VARCHAR(255) NOT NULL,
    scraped_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Índices únicos para evitar duplicados
    UNIQUE(post_id, source_name)
);

-- Tabla para historial de búsquedas (analytics)
CREATE TABLE smc_searches (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    results_count INTEGER DEFAULT 0,
    executed_at TIMESTAMP DEFAULT NOW(),
    execution_time_ms INTEGER,
    user_ip VARCHAR(45), -- Para tracking básico
    created_at TIMESTAMP DEFAULT NOW()
);

-- Índices para optimizar búsquedas
CREATE INDEX idx_smc_posts_source_name ON smc_posts(source_name);
CREATE INDEX idx_smc_posts_platform ON smc_posts(platform);
CREATE INDEX idx_smc_posts_text ON smc_posts USING gin(to_tsvector('english', text));
CREATE INDEX idx_smc_searches_query ON smc_searches(query);
CREATE INDEX idx_smc_searches_executed_at ON smc_searches(executed_at);

-- Row Level Security (RLS) - Por ahora abierto, después se puede restringir
ALTER TABLE smc_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE smc_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE smc_searches ENABLE ROW LEVEL SECURITY;

-- Políticas básicas (permitir todo por ahora)
CREATE POLICY "Allow all operations" ON smc_sources FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON smc_posts FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON smc_searches FOR ALL USING (true);

-- Función helper para obtener posts por fuente (compatibilidad con JSON original)
CREATE OR REPLACE FUNCTION get_posts_by_source(source_name_param VARCHAR)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'Name', source_name_param,
        'Posts', json_object_agg(
            post_id,
            json_build_object(
                'text', text,
                'post_owner', post_owner,
                'source', platform
            )
        )
    )
    INTO result
    FROM smc_posts 
    WHERE source_name = source_name_param;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Función para insertar posts desde el scraper
CREATE OR REPLACE FUNCTION insert_scraped_posts(
    source_name_param VARCHAR,
    posts_data JSON
) RETURNS INTEGER AS $$
DECLARE
    source_id_var INTEGER;
    post_key TEXT;
    post_data JSON;
    inserted_count INTEGER := 0;
BEGIN
    -- Crear o obtener source
    INSERT INTO smc_sources (name, platform, last_scraped)
    VALUES (source_name_param, 'Linkedin', NOW())
    ON CONFLICT (name) 
    DO UPDATE SET last_scraped = NOW()
    RETURNING id INTO source_id_var;
    
    -- Si no se obtuvo ID, buscarlo
    IF source_id_var IS NULL THEN
        SELECT id INTO source_id_var FROM smc_sources WHERE name = source_name_param;
    END IF;
    
    -- Insertar posts
    FOR post_key, post_data IN SELECT * FROM json_each(posts_data->'Posts')
    LOOP
        INSERT INTO smc_posts (
            post_id, 
            source_id, 
            source_name, 
            platform,
            text, 
            full_raw_text,
            post_owner
        )
        VALUES (
            post_key,
            source_id_var,
            source_name_param,
            COALESCE(post_data->>'source', 'Linkedin'),
            post_data->>'text',
            post_data->>'text', -- Por ahora igual, después se puede expandir
            COALESCE(post_data->>'post_owner', post_data->>'name', source_name_param)
        )
        ON CONFLICT (post_id, source_name) DO NOTHING;
        
        GET DIAGNOSTICS inserted_count = ROW_COUNT;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Insertar datos de ejemplo (de manthanbhikadiya_data.json)
-- Se ejecutará después de crear las tablas