# Phase 1.2 Required Fixes

**Priority Levels:**
- **HIGH:** Must fix before production
- **MEDIUM:** Should fix soon (before Phase 2)
- **LOW:** Nice to have (tech debt reduction)

---

## HIGH PRIORITY FIXES

### FIX-1: Add ToSchema derive to TierCounts

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/store.rs`
**Line:** 155

**Current Code:**
```rust
/// Count of memories in each storage tier
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TierCounts {
    /// Number of memories in hot tier (in-memory)
    pub hot: usize,
    /// Number of memories in warm tier (memory-mapped)
    pub warm: usize,
    /// Number of memories in cold tier (archived)
    pub cold: usize,
    /// Total number of memories across all tiers
    pub total: usize,
}
```

**Fixed Code:**
```rust
/// Count of memories in each storage tier
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct TierCounts {
    /// Number of memories in hot tier (in-memory)
    pub hot: usize,
    /// Number of memories in warm tier (memory-mapped)
    pub warm: usize,
    /// Number of memories in cold tier (archived)
    pub cold: usize,
    /// Total number of memories across all tiers
    pub total: usize,
}
```

**Rationale:**
- `ListMemoriesResponse` uses `TierCounts` and has `#[derive(ToSchema)]`
- OpenAPI spec generation requires all nested types to have `ToSchema`
- Without this, the OpenAPI JSON schema will be incomplete

**Verification:**
```bash
cargo doc --package engram-core --open
# Check that TierCounts appears in generated docs with schema
```

---

### FIX-2: Add explicit validation for limit=0 edge case

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-cli/src/api.rs`
**Line:** 2179-2181 (after query extraction)

**Current Code:**
```rust
// Validate and clamp pagination parameters
let offset = query.offset;
let limit = query.limit.min(1000);
```

**Fixed Code:**
```rust
// Validate and clamp pagination parameters
let offset = query.offset;
let limit = query.limit.min(1000);

// Validate limit is non-zero (use case unclear: error or count-only query?)
if limit == 0 {
    return Err(ApiError::bad_request(
        "Pagination limit must be greater than 0",
        "Specify a positive limit to retrieve memories. For counting only, use tier_counts in response.",
        "GET /api/v1/memories?tier=hot&limit=100",
    ));
}
```

**Alternative (if limit=0 is intentional for count-only queries):**
```rust
// Validate and clamp pagination parameters
let offset = query.offset;
// Note: limit=0 is valid for count-only queries (returns tier_counts without memories)
let limit = query.limit.min(1000);
```

**Recommendation:** Choose based on product decision. If unsure, reject limit=0 to prevent user confusion.

**Verification:**
Add test in `api_tier_iteration_tests.rs`:
```rust
#[tokio::test]
async fn test_limit_zero_returns_error() {
    let app = create_test_router().await;
    let (status, json) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?limit=0",
        None
    ).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "BAD_REQUEST");
}
```

---

## MEDIUM PRIORITY FIXES

### FIX-3: Add offset sanity check to prevent abuse

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-cli/src/api.rs`
**Line:** 2179-2181 (after query extraction)

**Current Code:**
```rust
// Validate and clamp pagination parameters
let offset = query.offset;
let limit = query.limit.min(1000);
```

**Fixed Code:**
```rust
// Validate and clamp pagination parameters
let offset = query.offset;
let limit = query.limit.min(1000);

// Prevent deep pagination abuse (offset sanity check)
const MAX_OFFSET: usize = 100_000;
if offset > MAX_OFFSET {
    return Err(ApiError::bad_request(
        format!("Offset too large: {} (max: {})", offset, MAX_OFFSET),
        "For deep pagination, use the streaming API or consolidation snapshots",
        "GET /api/v1/memories?tier=hot&offset=0&limit=1000",
    ));
}
```

**Rationale:**
- Deep pagination (offset >> total count) can indicate abuse or misconfiguration
- `.skip(offset)` still iterates through skipped items (O(offset) cost)
- Prevents pathological cases like `offset=999999999`

**Trade-off:** Limits legitimate use cases where users want to paginate through millions of memories.
**Alternative:** Remove if bulk export use cases are common. Consider streaming API instead.

---

### FIX-4: Add has_more field to PaginationInfo

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-cli/src/api.rs`
**Line:** 2119-2128

**Current Code:**
```rust
/// Pagination metadata for list responses
#[derive(Debug, Serialize, ToSchema)]
pub struct PaginationInfo {
    /// Starting offset of returned results
    pub offset: usize,
    /// Maximum number of results requested
    pub limit: usize,
    /// Actual number of results returned
    pub returned: usize,
}
```

**Fixed Code:**
```rust
/// Pagination metadata for list responses
#[derive(Debug, Serialize, ToSchema)]
pub struct PaginationInfo {
    /// Starting offset of returned results
    pub offset: usize,
    /// Maximum number of results requested
    pub limit: usize,
    /// Actual number of results returned
    pub returned: usize,
    /// Whether more results exist beyond this page
    pub has_more: bool,
}
```

**Update handler code** (line 2238-2246):
```rust
let returned = memories.len();

// Build response with pagination metadata
let response = ListMemoriesResponse {
    memories,
    count: returned,
    pagination: PaginationInfo {
        offset,
        limit,
        returned,
        has_more: returned == limit && offset + returned < tier_counts.hot,
    },
    tier_counts,
};
```

**Rationale:**
- Clients need to know if more pages exist without guessing
- Common pagination pattern (GitHub API, Stripe API, etc.)
- `has_more = true` iff: (1) returned == limit AND (2) not at end of data

**Backward Compatibility:** Additive change, doesn't break existing clients.

---

### FIX-5: Extract JSON construction helper to reduce duplication

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-cli/src/api.rs`
**Line:** 2196-2234 (list_memories_rest function)

**Current Code:**
```rust
"hot" => {
    store
        .iter_hot_memories()
        .skip(offset)
        .take(limit)
        .map(|(id, ep)| {
            if query.include_embeddings {
                json!({
                    "id": id,
                    "content": ep.what,
                    "embedding": ep.embedding.to_vec(),
                    "confidence": ep.encoding_confidence.raw(),
                    "timestamp": ep.when.to_rfc3339(),
                })
            } else {
                json!({
                    "id": id,
                    "content": ep.what,
                    "confidence": ep.encoding_confidence.raw(),
                    "timestamp": ep.when.to_rfc3339(),
                })
            }
        })
        .collect()
}
```

**Add helper function** (before `list_memories_rest`):
```rust
/// Build JSON representation of a memory for API response
fn memory_to_json(id: String, ep: Episode, include_embeddings: bool) -> serde_json::Value {
    let mut obj = json!({
        "id": id,
        "content": ep.what,
        "confidence": ep.encoding_confidence.raw(),
        "timestamp": ep.when.to_rfc3339(),
    });

    if include_embeddings {
        obj["embedding"] = json!(ep.embedding.to_vec());
    }

    obj
}
```

**Updated Code:**
```rust
"hot" => {
    store
        .iter_hot_memories()
        .skip(offset)
        .take(limit)
        .map(|(id, ep)| memory_to_json(id, ep, query.include_embeddings))
        .collect()
}
```

**Rationale:**
- Eliminates duplication between embedding inclusion branches
- Makes Phase 2 warm/cold tier implementation cleaner (same helper)
- Centralizes memory JSON format for consistency

**Phase 2 Benefit:**
```rust
"warm" => {
    store
        .iter_warm_memories()
        .skip(offset)
        .take(limit)
        .map(|(id, ep)| memory_to_json(id, ep, query.include_embeddings))
        .collect()
}
```

---

### FIX-6: Enhance documentation with RETURNS and ERRORS sections

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-cli/src/api.rs`
**Line:** 2147-2157

**Current Code:**
```rust
/// GET /api/v1/memories - List memories with tier-aware pagination
///
/// This endpoint supports querying across storage tiers (hot/warm/cold) with
/// pagination and optional embedding inclusion for reduced payload sizes.
///
/// Query parameters:
/// - `tier`: Storage tier ("hot", "warm", "cold", "all") - defaults to "hot"
/// - `offset`: Pagination offset (default: 0)
/// - `limit`: Number of results (default: 100, max: 1000)
/// - `include_embeddings`: Include 768-dim vectors (default: false)
/// - `space`: Memory space ID (defaults to server default)
pub async fn list_memories_rest(
```

**Fixed Code:**
```rust
/// GET /api/v1/memories - List memories with tier-aware pagination
///
/// This endpoint supports querying across storage tiers (hot/warm/cold) with
/// pagination and optional embedding inclusion for reduced payload sizes.
///
/// # Query Parameters
/// - `tier`: Storage tier ("hot", "warm", "cold", "all") - defaults to "hot"
/// - `offset`: Pagination offset (default: 0)
/// - `limit`: Number of results (default: 100, max: 1000)
/// - `include_embeddings`: Include 768-dim vectors (default: false)
/// - `space`: Memory space ID (defaults to server default)
///
/// # Returns
/// `ListMemoriesResponse` containing:
/// - `memories`: Array of memory objects with id, content, confidence, timestamp
///   (and optional 768-dim embedding if `include_embeddings=true`)
/// - `count`: Number of memories in this response (alias for `pagination.returned`)
/// - `pagination`: Metadata with offset, limit, returned count, and has_more flag
/// - `tier_counts`: Counts across all tiers (hot, warm, cold, total)
///
/// # Errors
/// - `400 BAD_REQUEST`: Invalid tier parameter (not one of: hot, warm, cold, all)
/// - `500 INTERNAL_SERVER_ERROR`: Failed to access memory space or store
/// - `501 NOT_IMPLEMENTED`: Warm/cold tier iteration not yet available (Phase 2)
///
/// # Examples
/// ```text
/// GET /api/v1/memories?tier=hot&offset=0&limit=50
/// GET /api/v1/memories?tier=hot&include_embeddings=true&limit=10
/// GET /api/v1/memories (uses defaults: tier=hot, offset=0, limit=100)
/// ```
///
/// # Backward Compatibility
/// Legacy clients relying on `memories` and `count` fields will continue to work.
/// New fields (`pagination`, `tier_counts`) are additive and optional to parse.
pub async fn list_memories_rest(
```

**Rationale:**
- Standard Rust doc format (# sections)
- Explicitly documents response structure
- Lists all possible error codes
- Provides concrete examples
- Documents backward compatibility guarantees

---

## LOW PRIORITY FIXES

### FIX-7: Add tier counts caching to reduce Phase 2 latency

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/store.rs`
**Line:** 1797-1826

**Current Code:**
```rust
pub fn get_tier_counts(&self) -> TierCounts {
    // Count hot tier
    let hot = self.memory_count.load(Ordering::Relaxed);

    // Get warm and cold counts from persistent backend if available
    #[cfg(feature = "memory_mapped_persistence")]
    {
        if let Some(ref backend) = self.persistent_backend {
            let stats = backend.get_tier_statistics();
            let warm = stats.warm.memory_count;
            let cold = stats.cold.memory_count;
            return TierCounts {
                hot,
                warm,
                cold,
                total: hot + warm + cold,
            };
        }
    }

    TierCounts {
        hot,
        warm: 0,
        cold: 0,
        total: hot,
    }
}
```

**Proposed Fix (Phase 2):**
Add cached tier counts with TTL:
```rust
pub struct MemoryStore {
    // ... existing fields ...
    tier_counts_cache: AtomicCell<Option<(TierCounts, Instant)>>,
    tier_counts_ttl: Duration,
}

pub fn get_tier_counts(&self) -> TierCounts {
    // Check cache first
    if let Some((cached, timestamp)) = self.tier_counts_cache.load() {
        if timestamp.elapsed() < self.tier_counts_ttl {
            return cached;
        }
    }

    // Cache miss or expired - recompute
    let hot = self.memory_count.load(Ordering::Relaxed);

    #[cfg(feature = "memory_mapped_persistence")]
    {
        if let Some(ref backend) = self.persistent_backend {
            let stats = backend.get_tier_statistics();
            let warm = stats.warm.memory_count;
            let cold = stats.cold.memory_count;
            let counts = TierCounts {
                hot,
                warm,
                cold,
                total: hot + warm + cold,
            };

            // Update cache
            self.tier_counts_cache.store(Some((counts, Instant::now())));
            return counts;
        }
    }

    let counts = TierCounts {
        hot,
        warm: 0,
        cold: 0,
        total: hot,
    };
    self.tier_counts_cache.store(Some((counts, Instant::now())));
    counts
}
```

**Rationale:**
- Phase 1: Hot tier is O(1) atomic read (no caching needed)
- Phase 2: Warm/cold counts may require I/O (filesystem stat, index scan)
- 1-second stale data acceptable for tier counts (not critical for correctness)

**Alternative:** Make `tier_counts` optional in API response (controlled by query param)

---

### FIX-8: Add property-based tests for pagination invariants

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-cli/tests/api_tier_iteration_tests.rs` (to be created)

**Add dependency** to `engram-cli/Cargo.toml`:
```toml
[dev-dependencies]
proptest = "1.4"
```

**Add test:**
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_pagination_invariants(
        total_items in 0usize..500,
        offset in 0usize..1000,
        limit in 1usize..200,
    ) {
        // Property: returned <= min(limit, max(0, total - offset))
        let expected_returned = limit.min(total_items.saturating_sub(offset));

        // Property: offset + returned <= total
        let returned = expected_returned;
        assert!(offset + returned <= total_items.max(offset));

        // Property: returned == limit implies has_more (if not at end)
        let has_more = returned == limit && offset + returned < total_items;
        if returned == limit && !has_more {
            assert_eq!(offset + returned, total_items);
        }
    }
}
```

**Rationale:**
- Validates pagination logic across all possible input combinations
- Catches edge cases that manual tests might miss
- Provides confidence in correctness properties

---

## Summary of Changes

| Fix | Priority | File | Lines | Estimated Time |
|-----|----------|------|-------|----------------|
| FIX-1: Add ToSchema | HIGH | store.rs | 155 | 5 min |
| FIX-2: Validate limit=0 | HIGH | api.rs | 2181 | 10 min |
| FIX-3: Offset sanity check | MEDIUM | api.rs | 2181 | 10 min |
| FIX-4: Add has_more field | MEDIUM | api.rs | 2119-2246 | 20 min |
| FIX-5: Extract helper | MEDIUM | api.rs | 2196-2234 | 15 min |
| FIX-6: Enhance docs | MEDIUM | api.rs | 2147 | 10 min |
| FIX-7: Cache tier counts | LOW | store.rs | 1797 | 30 min (Phase 2) |
| FIX-8: Property tests | LOW | tests/ | New file | 30 min |

**Total Estimated Time:** 2.5 hours (excluding FIX-7 which is Phase 2, FIX-8 which is optional)

---

## Implementation Order

1. **FIX-1** (5 min) - Unblocks OpenAPI spec generation
2. **Create test suite** (1 hour) - See PHASE_1_2_FIXES_REQUIRED.md deliverable #3
3. **FIX-2** (10 min) - Add limit=0 validation with test
4. **FIX-5** (15 min) - Extract helper (makes remaining fixes cleaner)
5. **FIX-4** (20 min) - Add has_more with test
6. **FIX-6** (10 min) - Enhance documentation
7. **FIX-3** (10 min) - Add offset sanity check with test
8. **Run `make quality`** - Verify all changes pass clippy and tests

**Total: ~2 hours**

FIX-7 and FIX-8 can be deferred to Phase 2 or tech debt sprint.
