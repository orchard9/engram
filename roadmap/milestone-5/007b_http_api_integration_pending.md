# Task 007b: HTTP API Integration

## Status
PENDING

## Priority
P1 (High Priority - Blocks user-facing probabilistic queries)

## Effort Estimate
1 day

## Dependencies
- Task 007 ✅ (Testing infrastructure complete)

## Objective
Integrate probabilistic query system into HTTP API and CLI, enabling end-users to access probabilistic queries through production endpoints.

## Technical Approach

### 1. Implement MemoryStore Integration

Add `recall_probabilistic()` method to `MemoryStore`:

```rust
// engram-core/src/store.rs

impl MemoryStore {
    /// Recall memories with full probabilistic query support
    pub fn recall_probabilistic(&self, cue: &Cue) -> ProbabilisticQueryResult {
        // Step 1: Perform standard recall
        let recall_result = self.recall(cue);

        // Step 2: Extract activation paths if spreading was used
        let activation_paths = self.extract_activation_paths(&recall_result);

        // Step 3: Gather uncertainty sources from system metrics
        let uncertainty_sources = self.gather_uncertainty_sources();

        // Step 4: Execute probabilistic query
        let executor = ProbabilisticQueryExecutor::default();
        executor.execute(recall_result.results, &activation_paths, uncertainty_sources)
    }

    /// Extract activation paths from recall context
    fn extract_activation_paths(&self, recall_result: &RecallResult) -> Vec<ActivationPath> {
        // Extract from spreading activation metadata if available
        vec![] // TODO: Implement based on recall context
    }

    /// Gather uncertainty sources from system state
    fn gather_uncertainty_sources(&self) -> Vec<UncertaintySource> {
        let mut sources = vec![];

        // Add system pressure uncertainty
        if let Some(pressure) = self.get_system_pressure() {
            sources.push(UncertaintySource::SystemPressure {
                pressure_level: pressure,
                effect_on_confidence: pressure * 0.1,
            });
        }

        // Add spreading activation uncertainty if enabled
        if self.config.spreading_enabled {
            sources.push(UncertaintySource::SpreadingActivationNoise {
                activation_variance: 0.05,
                path_diversity: 0.1,
            });
        }

        sources
    }
}
```

### 2. Add HTTP API Endpoint

Create new endpoint in `engram-cli/src/api.rs`:

```rust
/// Probabilistic query endpoint
#[utoipa::path(
    post,
    path = "/api/v1/query/probabilistic",
    request_body = ProbabilisticQueryRequest,
    responses(
        (status = 200, description = "Query successful", body = ProbabilisticQueryResponse),
        (status = 400, description = "Invalid request"),
        (status = 500, description = "Internal server error")
    ),
    tag = "query"
)]
async fn recall_probabilistic(
    State(store): State<Arc<MemoryStore>>,
    Json(request): Json<ProbabilisticQueryRequest>,
) -> Result<Json<ProbabilisticQueryResponse>, (StatusCode, String)> {
    // Parse cue from request
    let cue = parse_cue_from_request(&request)?;

    // Execute probabilistic recall
    let result = store.recall_probabilistic(&cue);

    // Convert to response
    Ok(Json(ProbabilisticQueryResponse {
        episodes: result.episodes.into_iter().map(|(ep, conf)| {
            EpisodeWithConfidence {
                episode: ep,
                confidence: conf.raw(),
            }
        }).collect(),
        confidence_interval: ConfidenceIntervalResponse {
            point: result.confidence_interval.point.raw(),
            lower: result.confidence_interval.lower.raw(),
            upper: result.confidence_interval.upper.raw(),
            width: result.confidence_interval.width,
        },
        evidence_chain: result.evidence_chain.into_iter().map(Into::into).collect(),
        uncertainty_sources: result.uncertainty_sources.into_iter().map(Into::into).collect(),
        is_successful: result.is_successful(),
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
struct ProbabilisticQueryRequest {
    /// Query text for semantic search
    #[serde(default)]
    query: Option<String>,

    /// Embedding vector for similarity search
    #[serde(default)]
    embedding: Option<Vec<f32>>,

    /// Maximum results to return
    #[serde(default = "default_limit")]
    limit: usize,
}

#[derive(Debug, Serialize, ToSchema)]
struct ProbabilisticQueryResponse {
    /// Retrieved episodes with confidence scores
    episodes: Vec<EpisodeWithConfidence>,

    /// Aggregated confidence interval
    confidence_interval: ConfidenceIntervalResponse,

    /// Evidence chain showing sources
    evidence_chain: Vec<EvidenceResponse>,

    /// Uncertainty sources contributing to interval width
    uncertainty_sources: Vec<UncertaintySourceResponse>,

    /// Whether query was successful (confidence > threshold)
    is_successful: bool,
}
```

### 3. Add CLI Commands

Extend CLI in `engram-cli/src/main.rs`:

```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...

    /// Query with probabilistic confidence intervals
    QueryProbabilistic {
        /// Query text
        #[arg(short, long)]
        query: String,

        /// Maximum results
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Output format (json, table, compact)
        #[arg(short, long, default_value = "table")]
        format: OutputFormat,
    },
}

async fn handle_query_probabilistic(
    query: String,
    limit: usize,
    format: OutputFormat,
) -> Result<()> {
    let client = create_http_client()?;

    let request = json!({
        "query": query,
        "limit": limit,
    });

    let response = client
        .post(&format!("{}/api/v1/query/probabilistic", base_url()))
        .json(&request)
        .send()
        .await?;

    let result: ProbabilisticQueryResponse = response.json().await?;

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        OutputFormat::Table => {
            print_probabilistic_table(&result);
        }
        OutputFormat::Compact => {
            print_probabilistic_compact(&result);
        }
    }

    Ok(())
}

fn print_probabilistic_table(result: &ProbabilisticQueryResponse) {
    println!("\n📊 Probabilistic Query Results\n");
    println!("Confidence: {:.2}% [{:.2}% - {:.2}%]",
        result.confidence_interval.point * 100.0,
        result.confidence_interval.lower * 100.0,
        result.confidence_interval.upper * 100.0
    );
    println!("Status: {}\n", if result.is_successful { "✅ Successful" } else { "⚠️  Low Confidence" });

    // Print episodes table
    for (i, ep_conf) in result.episodes.iter().enumerate() {
        println!("{:2}. [{}] {}",
            i + 1,
            format_confidence(ep_conf.confidence),
            ep_conf.episode.what
        );
    }

    // Print evidence chain
    if !result.evidence_chain.is_empty() {
        println!("\n🔗 Evidence Chain:");
        for evidence in &result.evidence_chain {
            println!("  - {}: {:.2}%", evidence.source, evidence.confidence * 100.0);
        }
    }

    // Print uncertainty sources
    if !result.uncertainty_sources.is_empty() {
        println!("\n⚠️  Uncertainty Sources:");
        for source in &result.uncertainty_sources {
            println!("  - {}", source.description);
        }
    }
}
```

### 4. Integration Tests

Create `engram-cli/tests/probabilistic_api_tests.rs`:

```rust
#[tokio::test]
async fn test_probabilistic_query_endpoint() {
    let (server, client) = setup_test_server().await;

    // Store test episodes
    store_test_episodes(&client, 10).await;

    // Execute probabilistic query
    let response = client
        .post(&format!("{}/api/v1/query/probabilistic", server.url()))
        .json(&json!({
            "query": "test content",
            "limit": 5
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let result: ProbabilisticQueryResponse = response.json().await.unwrap();

    // Validate response
    assert!(!result.episodes.is_empty());
    assert!(result.confidence_interval.point > 0.0);
    assert!(result.confidence_interval.lower <= result.confidence_interval.upper);
    assert!(!result.evidence_chain.is_empty());
}

#[tokio::test]
async fn test_cli_query_probabilistic() {
    let server = start_test_server().await;

    // Execute CLI command
    let output = Command::new("engram")
        .args(&["query-probabilistic", "-q", "test query", "-l", "5"])
        .output()
        .await
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Probabilistic Query Results"));
    assert!(stdout.contains("Confidence:"));
    assert!(stdout.contains("Evidence Chain"));
}
```

## Acceptance Criteria

- [ ] `MemoryStore::recall_probabilistic()` implemented and tested
- [ ] HTTP endpoint `/api/v1/query/probabilistic` returns correct responses
- [ ] CLI command `engram query-probabilistic` works correctly
- [ ] Response includes episodes, confidence intervals, evidence chain, uncertainty sources
- [ ] Integration tests validate end-to-end flow with live HTTP server
- [ ] OpenAPI documentation updated with new endpoint
- [ ] CLI help text documents new command
- [ ] Error handling for invalid requests (malformed cue, etc.)
- [ ] Performance validated: <10ms P95 latency for API endpoint
- [ ] Zero clippy warnings

## Files

- **Modify**: `engram-core/src/store.rs` (add `recall_probabilistic` method)
- **Modify**: `engram-cli/src/api.rs` (add HTTP endpoint)
- **Modify**: `engram-cli/src/main.rs` (add CLI command)
- **Create**: `engram-cli/tests/probabilistic_api_tests.rs` (integration tests)
- **Modify**: `engram-cli/src/openapi.rs` (update OpenAPI spec)

## Testing Approach

1. **Unit Tests**: Test `recall_probabilistic()` method in isolation
2. **Integration Tests**: Test HTTP endpoint with live server
3. **CLI Tests**: Test command-line interface
4. **E2E Tests**: Full flow from CLI → HTTP → MemoryStore → Query Executor

## Technical Notes

- Reuses existing `ProbabilisticQueryExecutor` (no changes needed)
- Uncertainty sources gathered from real system metrics
- Evidence chain includes spreading activation paths when available
- Backward compatible - doesn't change existing `/api/v1/recall` endpoint

## Future Enhancements

- Add query expansion integration (semantic expansion before recall)
- Support for query combinators in API (`AND`, `OR`, `NOT`)
- Streaming results for large queries
- Batch query API for multiple queries at once
- GraphQL endpoint for more flexible querying
