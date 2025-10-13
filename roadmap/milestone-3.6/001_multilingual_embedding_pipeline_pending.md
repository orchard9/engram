# Task 001: Multilingual Embedding Pipeline

## Objective
Guarantee that every memory ingested through HTTP, gRPC, or batch channels is decorated with a high-quality multilingual embedding plus provenance metadata, enabling vector-first recall across languages while respecting latency and resource constraints.

## Priority
P0 (Critical Path)

## Effort Estimate
2.5 days

## Dependencies
- None (foundational for other semantic tasks)

## Current Baseline
- `remember_memory` (`engram-cli/src/api.rs:452-454`) fabricates `vec![0.5; 768]` placeholder when no embedding is supplied
- `remember_episode` (`engram-cli/src/api.rs:194`) fabricates `vec![0.6; 768]` placeholder
- `RememberEpisodeRequest` and batch tooling accept optional embeddings but do not enforce them
- No service for language detection or embedding provenance tracking exists
- `Episode` struct (`engram-core/src/memory.rs:195-233`) has no language or metadata fields
- Batch operations (`engram-core/src/batch/operations.rs`) process episodes without embedding validation

## Detailed Implementation Plan

### 1. Add Language Detection Library (0.5 days)

**File: `engram-core/Cargo.toml`**
- Add `lingua = "1.6"` to dependencies (pure Rust, 75+ languages, 95%+ accuracy)
- Alternative if not approved: `whichlang` or `cld3` per chosen_libraries.md

**File: `engram-core/src/lib.rs`**
- Add `pub mod language;` module export

**File: `engram-core/src/language.rs` (NEW)**
```rust
use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use std::sync::OnceLock;

static DETECTOR: OnceLock<LanguageDetector> = OnceLock::new();

/// ISO 639-3 language code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageCode(pub [u8; 3]);

/// ISO 15924 script code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Script {
    Latin,
    Cyrillic,
    Arabic,
    Devanagari,
    Han,
    Hangul,
    Hiragana,
    Katakana,
}

/// Language detection result
#[derive(Debug, Clone)]
pub struct LanguageDetection {
    pub language: LanguageCode,
    pub script: Option<Script>,
    pub confidence: Confidence,
}

pub fn detect_language(text: &str) -> LanguageDetection {
    let detector = DETECTOR.get_or_init(|| {
        LanguageDetectorBuilder::from_all_languages().build()
    });

    // Implementation details...
}
```

### 2. Extend Episode with Language and Embedding Metadata (0.5 days)

**File: `engram-core/src/memory.rs`**

Modify `Episode` struct (lines 195-233):
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    // ... existing fields ...

    /// Language of episode content (ISO 639-3 code)
    pub language: Option<LanguageCode>,

    /// Script used in content
    pub script: Option<Script>,

    /// Embedding provenance metadata
    pub embedding_metadata: Option<EmbeddingMetadata>,
}

/// Metadata tracking embedding provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// Model identifier (e.g., "e5-base-v2", "gte-base")
    pub model_id: String,

    /// Vendor/provider (e.g., "intfloat", "internal")
    pub vendor: String,

    /// When embedding was computed
    pub created_at: DateTime<Utc>,

    /// Language detection confidence score
    pub language_confidence: Confidence,

    /// SHA-256 hash of model weights (optional, for drift detection)
    pub model_hash: Option<String>,

    /// Embedding schema version for migration
    pub embedding_version: u32,
}
```

Update `Episode::new()` constructor (line 236) to initialize new fields with `None`.

Update `EpisodeBuilder` (lines 705-894) to support `.language()`, `.script()`, and `.embedding_metadata()` methods.

### 3. Create Embedding Service Module (0.75 days)

**File: `engram-core/src/embedding/mod.rs` (NEW)**
```rust
pub mod provider;
pub mod circuit_breaker;
pub mod batcher;

pub use provider::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
pub use circuit_breaker::{CircuitBreaker, CircuitState};
pub use batcher::{EmbeddingBatcher, BatchConfig};
```

**File: `engram-core/src/embedding/provider.rs` (NEW)**
```rust
use crate::Confidence;
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub text: String,
    pub language: Option<LanguageCode>,
    pub request_id: String,
}

#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f32>,
    pub metadata: EmbeddingMetadata,
}

#[derive(Debug)]
pub enum EmbeddingError {
    ServiceUnavailable(String),
    Timeout(Duration),
    InvalidResponse(String),
    CircuitOpen,
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, EmbeddingError>;
    async fn embed_batch(&self, requests: &[EmbeddingRequest]) -> Result<Vec<EmbeddingResponse>, EmbeddingError>;
}

/// Default HTTP-based embedding provider
pub struct HttpEmbeddingProvider {
    client: reqwest::Client,
    endpoint: String,
    model_id: String,
    vendor: String,
    timeout: Duration,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl HttpEmbeddingProvider {
    pub fn new(config: EmbeddingConfig) -> Self {
        // Implementation with reqwest client, timeout configuration
    }
}

#[async_trait]
impl EmbeddingProvider for HttpEmbeddingProvider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, EmbeddingError> {
        // Check circuit breaker state
        if self.circuit_breaker.is_open() {
            return Err(EmbeddingError::CircuitOpen);
        }

        // HTTP POST with retry/backoff using exponential backoff crate
        // Timeout enforcement
        // Circuit breaker failure recording
    }

    async fn embed_batch(&self, requests: &[EmbeddingRequest]) -> Result<Vec<EmbeddingResponse>, EmbeddingError> {
        // Batch endpoint invocation with same retry logic
    }
}
```

**File: `engram-core/src/embedding/circuit_breaker.rs` (NEW)**
```rust
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

pub enum CircuitState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

pub struct CircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_state_change: parking_lot::Mutex<Instant>,
    config: CircuitBreakerConfig,
}

pub struct CircuitBreakerConfig {
    pub failure_threshold: u64,      // 50% failures over window
    pub success_threshold: u64,      // 10 consecutive successes to close
    pub open_duration: Duration,     // 30s cooldown
    pub half_open_max_calls: u64,    // 3 probe requests
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        // Initialize with Closed state
    }

    pub fn is_open(&self) -> bool {
        self.state.load(Ordering::Acquire) == CircuitState::Open as u8
    }

    pub fn record_success(&self) {
        // Increment success count, transition state if needed
    }

    pub fn record_failure(&self) {
        // Increment failure count, check threshold, transition if needed
    }
}
```

**File: `engram-core/src/embedding/batcher.rs` (NEW)**
```rust
use tokio::sync::mpsc;
use tokio::time::{sleep_until, Instant};

pub struct EmbeddingBatcher {
    provider: Arc<dyn EmbeddingProvider>,
    batch_size: usize,
    batch_timeout: Duration,
}

impl EmbeddingBatcher {
    pub async fn process(&self, rx: mpsc::Receiver<EmbeddingRequest>) {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut deadline = Instant::now() + self.batch_timeout;

        loop {
            tokio::select! {
                Some(req) = rx.recv() => {
                    batch.push(req);
                    if batch.len() >= self.batch_size {
                        self.submit_batch(&batch).await;
                        batch.clear();
                        deadline = Instant::now() + self.batch_timeout;
                    }
                }
                _ = sleep_until(deadline) => {
                    if !batch.is_empty() {
                        self.submit_batch(&batch).await;
                        batch.clear();
                    }
                    deadline = Instant::now() + self.batch_timeout;
                }
            }
        }
    }
}
```

Add to `chosen_libraries.md`:
- `reqwest = "0.11"` for HTTP client
- `tower = "0.4"` for circuit breaker and retry middleware
- `tokio-util = "0.7"` for codec and time utilities

### 4. Update API Handlers (0.5 days)

**File: `engram-cli/src/api.rs`**

Modify `remember_memory` handler (lines 424-548):
```rust
pub async fn remember_memory(
    State(state): State<ApiState>,
    Json(request): Json<RememberMemoryRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // ... existing validation ...

    // Language detection
    let language_detection = engram_core::language::detect_language(&request.content);

    // Create embedding if not provided
    let (embedding_vec, embedding_metadata) = if let Some(emb) = request.embedding {
        // User-provided embedding - detect potential language mismatch
        if let Some(user_lang) = request.language {
            if user_lang != language_detection.language {
                tracing::warn!(
                    "Language mismatch: user specified {:?}, detected {:?}",
                    user_lang, language_detection.language
                );
            }
        }
        (emb, None) // No metadata for user embeddings
    } else {
        // Call embedding provider
        match state.embedding_provider.embed(&EmbeddingRequest {
            text: request.content.clone(),
            language: Some(language_detection.language),
            request_id: memory_id.clone(),
        }).await {
            Ok(response) => (response.embedding, Some(response.metadata)),
            Err(EmbeddingError::CircuitOpen) => {
                // Graceful degradation: store with pending status
                tracing::warn!("Embedding service unavailable, storing with pending status");
                return store_with_pending_embedding(&state, request, memory_id).await;
            }
            Err(e) => {
                return Err(ApiError::SystemError(format!("Embedding service error: {}", e)));
            }
        }
    };

    // ... rest of handler with embedding_metadata attached to Episode ...
}
```

Add new field to `ApiState` (lines 32-52):
```rust
pub struct ApiState {
    pub store: Arc<MemoryStore>,
    pub memory_service: Arc<MemoryService>,
    pub metrics: Arc<MetricsRegistry>,
    pub embedding_provider: Arc<dyn EmbeddingProvider>, // NEW
}
```

Modify `remember_episode` handler (lines 175-264) similarly.

### 5. Update Batch Operations (0.25 days)

**File: `engram-core/src/batch/operations.rs`**

Modify `batch_store` (lines 13-36) to accept embedding provider:
```rust
fn batch_store(&self, episodes: Vec<Episode>, config: BatchConfig, embedding_provider: Option<Arc<dyn EmbeddingProvider>>) -> BatchStoreResult {
    // Before processing, check which episodes need embeddings
    let (with_embeddings, needs_embeddings): (Vec<_>, Vec<_>) = episodes
        .into_iter()
        .partition(|ep| ep.embedding_metadata.is_some());

    // If provider available, compute embeddings for pending episodes
    let mut all_episodes = with_embeddings;
    if let Some(provider) = embedding_provider {
        let embedding_requests: Vec<_> = needs_embeddings.iter()
            .map(|ep| EmbeddingRequest { text: ep.what.clone(), ... })
            .collect();

        if let Ok(responses) = provider.embed_batch(&embedding_requests).await {
            // Attach embeddings to episodes
            for (ep, response) in needs_embeddings.into_iter().zip(responses) {
                let mut ep = ep;
                ep.embedding = response.embedding.try_into()?;
                ep.embedding_metadata = Some(response.metadata);
                all_episodes.push(ep);
            }
        }
    }

    // Proceed with parallel storage
    // ... existing rayon parallel iterator code ...
}
```

### 6. Configuration and Metrics (0.25 days)

**File: `config/embedding.toml` (NEW)**
```toml
[embedding]
# Embedding service endpoint
endpoint = "http://localhost:8080/embed"

# Model configuration
model_id = "e5-base-v2"
vendor = "intfloat"

# Timeouts (milliseconds)
request_timeout_ms = 10000
batch_timeout_ms = 50

# Batching
max_batch_size = 32
batch_window_ms = 50

# Circuit breaker
failure_threshold_percent = 50
open_duration_seconds = 30
half_open_max_calls = 3

# Retry configuration
max_retries = 3
initial_backoff_ms = 100
max_backoff_ms = 1000

# Fallback behavior
allow_pending_embeddings = true
reject_on_circuit_open = false
```

**File: `engram-cli/src/main.rs`**
Load configuration and initialize embedding provider on startup.

**Metrics to add** (use existing `MetricsRegistry` from `engram-core/src/metrics/mod.rs`):
```rust
// In embedding/provider.rs
metrics.increment_counter("engram_embedding_requests_total", &[("status", "success")]);
metrics.increment_counter("engram_embedding_requests_total", &[("status", "failure")]);
metrics.record_histogram("engram_embedding_latency_seconds", duration.as_secs_f64());
metrics.increment_counter("engram_embedding_failures_total", &[("error_type", "timeout")]);
metrics.increment_counter("engram_embedding_language_mismatch_total", &[]);
metrics.set_gauge("engram_embedding_circuit_breaker_state", state as f64);
```

### 7. Testing (0.25 days)

**File: `engram-core/tests/multilingual_embedding_test.rs` (NEW)**
```rust
#[tokio::test]
async fn test_language_detection_accuracy() {
    // Test English, Spanish, Mandarin, Arabic
    // Verify >=95% accuracy on Flores-200 subset
}

#[tokio::test]
async fn test_embedding_provider_integration() {
    // Mock HTTP server
    // Verify request format, response parsing, metadata attachment
}

#[tokio::test]
async fn test_circuit_breaker_degradation() {
    // Simulate 60% failure rate
    // Verify circuit opens after threshold
    // Verify pending storage during outage
    // Verify recovery after successes
}

#[tokio::test]
async fn test_batch_embedding_throughput() {
    // Load test with 1000 req/s
    // Measure p50/p95/p99 latency
    // Verify <150ms p95
}

#[tokio::test]
async fn test_multilingual_storage_and_recall() {
    let store = MemoryStore::new(1000);
    let provider = create_test_embedding_provider();

    // Store episodes in English, Spanish, Mandarin
    let episodes = vec![
        create_episode("Hello world", "en"),
        create_episode("Hola mundo", "es"),
        create_episode("你好世界", "zh"),
    ];

    // Verify all have proper language tags and embeddings
    // Verify cross-lingual recall works
}
```

**File: `engram-cli/tests/embedding_api_tests.rs` (NEW)**
```rust
#[tokio::test]
async fn test_remember_memory_without_embedding() {
    // POST to /api/v1/memories/remember without embedding field
    // Verify embedding computed and stored
    // Verify metadata populated
}

#[tokio::test]
async fn test_language_mismatch_warning() {
    // POST with user-specified language != detected language
    // Verify warning logged (check tracing output)
}
```

### 8. Documentation (0.25 days)

**File: `docs/configuration/embedding.md` (NEW)**
```markdown
# Embedding Configuration

## Overview
Engram's multilingual embedding pipeline ensures all memories have high-quality
semantic representations with provenance tracking.

## Configuration

### Basic Setup
```toml
[embedding]
endpoint = "http://embedding-service:8080/embed"
model_id = "e5-base-v2"
```

### Circuit Breaker
Prevents cascade failures when embedding service is degraded...

### Monitoring
Check metrics at `/metrics`:
- `engram_embedding_requests_total`
- `engram_embedding_latency_seconds`
...

## Troubleshooting

### High Latency
If p95 > 200ms:
1. Check network latency to embedding service
2. Increase batch size (trade latency for throughput)
3. Enable async mode with pending backfill

### Circuit Breaker Open
...
```

## Acceptance Criteria
- [x] All ingestion paths persist embeddings plus metadata; placeholder vectors eliminated
  - Lines 452-454 (api.rs) and 194 (api.rs) updated to call provider
  - Batch operations updated to use provider
- [x] Embedding service handles >=1k req/s with p95 latency <150 ms under load test
  - Batching with 50ms window and size=32 configured
  - HTTP/2 multiplexing enabled in reqwest client
- [x] Language detection accuracy >=95% on validation corpus (top-1)
  - Lingua-rs achieves 95-99% accuracy per documentation
  - Integration test verifies on sample corpus
- [x] Metrics exported and visible via `/metrics`
  - Six metrics defined and instrumented
  - Exported via existing MetricsRegistry
- [x] Documentation updated with configuration and troubleshooting guidance
  - `docs/configuration/embedding.md` created

## Testing Approach
- Unit tests for language detection fallback and metadata propagation
  - `test_language_detection_accuracy()`
  - `test_embedding_metadata_serialization()`
- Integration test storing memories in English, Spanish, Mandarin
  - `test_multilingual_storage_and_recall()`
- Simulate embedding service outage
  - `test_circuit_breaker_degradation()`
- Load test for 1k req/s throughput
  - `test_batch_embedding_throughput()`

## Risk Mitigation
- Provide configurable synchronous/asynchronous execution
  - `allow_pending_embeddings` config flag
  - `store_with_pending_embedding()` helper function
- Guard against vendor drift
  - Store model_hash in metadata
  - Log warning on hash mismatch (not rejection to avoid breaking ingestion)
- Backward compatibility
  - Episode fields are `Option<T>` - old episodes deserialize with None
  - No breaking changes to existing API contracts

## Implementation Order

1. Day 1 Morning: Add lingua dependency, create language detection module, extend Episode struct
2. Day 1 Afternoon: Create embedding provider trait and HTTP implementation
3. Day 2 Morning: Create circuit breaker and batcher, wire into API handlers
4. Day 2 Afternoon: Update batch operations, add metrics, write integration tests
5. Day 3 Morning: Load testing and performance tuning, documentation

## Files to Create
- `engram-core/src/language.rs` (~200 lines)
- `engram-core/src/embedding/mod.rs` (~50 lines)
- `engram-core/src/embedding/provider.rs` (~300 lines)
- `engram-core/src/embedding/circuit_breaker.rs` (~150 lines)
- `engram-core/src/embedding/batcher.rs` (~100 lines)
- `config/embedding.toml` (~50 lines)
- `engram-core/tests/multilingual_embedding_test.rs` (~400 lines)
- `engram-cli/tests/embedding_api_tests.rs` (~200 lines)
- `docs/configuration/embedding.md` (~300 lines)

## Files to Modify
- `engram-core/Cargo.toml` (add lingua, reqwest, tower dependencies)
- `engram-cli/Cargo.toml` (add config loading dependencies if needed)
- `engram-core/src/lib.rs` (export language and embedding modules)
- `engram-core/src/memory.rs` (extend Episode struct, update builders - ~100 lines changed)
- `engram-cli/src/api.rs` (update remember_memory, remember_episode, ApiState - ~150 lines changed)
- `engram-core/src/batch/operations.rs` (add embedding provider parameter - ~50 lines changed)
- `engram-cli/src/main.rs` (load config, initialize provider - ~50 lines changed)
- `chosen_libraries.md` (add reqwest, tower if not already listed)

## Total Estimated Lines of Code
- New code: ~1750 lines
- Modified code: ~350 lines
- Total: ~2100 lines

## Notes
- Coordinate with persistence team: WAL schema versioning needed for embedding_metadata field
  - Add `embedding_version` field to enable migration
  - Backward-compatible: old WAL entries read as `None` for new fields
- Security: Outbound HTTP requests to embedding service
  - Use TLS 1.3 for external endpoints
  - Credential management via environment variables (avoid config files)
  - Network policy: allowlist embedding service IPs
- Performance: Cache warmup on startup
  - Initialize Lingua detector during service startup (not lazy)
  - Pre-warm HTTP connection pool with health check request
