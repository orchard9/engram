//! Deterministic workload replay from traces

use crate::metrics_collector::MetricsCollector;
use crate::workload_generator::Operation;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub operations: Vec<TimedOperation>,
    pub metadata: TraceMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedOperation {
    pub timestamp_ms: u64,
    pub operation: Operation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub start_time: String,
    pub duration_seconds: u64,
    pub total_operations: u64,
}

pub fn load_trace(path: &Path) -> Result<Trace> {
    let content = std::fs::read_to_string(path)?;
    let trace: Trace = serde_json::from_str(&content)?;
    Ok(trace)
}

pub async fn replay_trace(
    trace: Trace,
    rate_multiplier: f64,
    endpoint: &str,
) -> Result<MetricsCollector> {
    use std::time::{Duration, Instant};

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    let mut metrics = MetricsCollector::new();

    let replay_start = Instant::now();
    let mut last_timestamp_ms = 0u64;

    for timed_op in trace.operations {
        // Calculate delay based on rate multiplier
        let original_delay_ms = timed_op.timestamp_ms.saturating_sub(last_timestamp_ms);
        let _adjusted_delay_ms = (original_delay_ms as f64 / rate_multiplier) as u64;

        // Wait for the appropriate time
        let target_time = replay_start + Duration::from_millis(timed_op.timestamp_ms);
        let now = Instant::now();
        if target_time > now {
            tokio::time::sleep(target_time - now).await;
        }

        // Execute operation
        let op_start = Instant::now();
        let result = execute_operation(&client, endpoint, &timed_op.operation).await;
        let op_elapsed = op_start.elapsed();

        // Record metrics
        metrics.record_operation(timed_op.operation.op_type(), op_elapsed, result.is_ok());

        if let Err(e) = result {
            tracing::debug!("Operation failed: {}", e);
        }

        last_timestamp_ms = timed_op.timestamp_ms;
    }

    Ok(metrics)
}

async fn execute_operation(
    client: &reqwest::Client,
    endpoint: &str,
    operation: &Operation,
) -> Result<()> {
    use crate::workload_generator::Operation;

    match operation {
        Operation::Store { memory } => {
            let url = format!("{}/api/v1/memories", endpoint);
            client
                .post(&url)
                .json(memory)
                .send()
                .await?
                .error_for_status()?;
        }
        Operation::Recall { cue } => {
            let url = format!("{}/api/v1/recall", endpoint);
            client
                .post(&url)
                .json(cue)
                .send()
                .await?
                .error_for_status()?;
        }
        Operation::EmbeddingSearch { query, k } => {
            let url = format!("{}/api/v1/search", endpoint);
            let body = serde_json::json!({
                "query": query,
                "k": k,
            });
            client
                .post(&url)
                .json(&body)
                .send()
                .await?
                .error_for_status()?;
        }
        Operation::PatternCompletion { partial } => {
            let url = format!("{}/api/v1/complete", endpoint);
            client
                .post(&url)
                .json(partial)
                .send()
                .await?
                .error_for_status()?;
        }
    }

    Ok(())
}
