-- API Keys table with all required fields
-- Note: WAL mode is set via connection options, not in migration
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    secret_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    permissions TEXT NOT NULL, -- JSON array of Permission enum values
    allowed_spaces TEXT NOT NULL, -- JSON array of MemorySpaceId values
    rate_limit_rps INTEGER NOT NULL,
    rate_limit_burst INTEGER NOT NULL,
    created_at TEXT NOT NULL, -- ISO 8601 timestamp
    expires_at TEXT, -- ISO 8601 timestamp (nullable)
    last_used TEXT, -- ISO 8601 timestamp (nullable)
    revoked_at TEXT, -- ISO 8601 timestamp (nullable)
    revocation_reason TEXT -- Audit trail for revocations (nullable)
);

-- Index for filtering by expiration date
CREATE INDEX idx_expires_at ON api_keys(expires_at)
    WHERE expires_at IS NOT NULL;

-- Index for filtering revoked keys
CREATE INDEX idx_revoked_at ON api_keys(revoked_at)
    WHERE revoked_at IS NOT NULL;

-- Index for last_used queries (cleanup, monitoring)
CREATE INDEX idx_last_used ON api_keys(last_used);
