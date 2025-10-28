//! Enhanced operational benchmark commands

use crate::output::{OperationProgress, spinner};
use anyhow::{Context, Result};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Run latency benchmark with percentile calculations
pub async fn run_latency_benchmark(
    operation: &str,
    iterations: usize,
    warmup: usize,
) -> Result<()> {
    let (port, _) = crate::cli::server::get_server_connection().await?;

    println!("Benchmarking {} latency", operation);
    println!("Warmup: {} iterations", warmup);
    println!("Measurement: {} iterations\n", iterations);

    let spinner_obj = spinner("Running warmup phase");

    // Warmup phase
    for _ in 0..warmup {
        execute_operation(operation, port).await?;
    }

    spinner_obj.finish_with_message("Warmup complete");

    // Measurement phase
    let progress = OperationProgress::new("Benchmark", iterations as u64);
    let mut latencies = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = Instant::now();
        execute_operation(operation, port).await?;
        let elapsed = start.elapsed();

        latencies.push(elapsed);
        progress.inc(1);

        if i % 100 == 0 {
            progress.set_message(&format!("Completed {}/{}", i, iterations));
        }
    }

    progress.finish("Benchmark complete");

    // Calculate percentiles
    latencies.sort();
    let p50 = latencies[iterations / 2];
    let p95 = latencies[(iterations * 95) / 100];
    let p99 = latencies[(iterations * 99) / 100];
    let min = latencies[0];
    let max = latencies[iterations - 1];
    let mean = latencies.iter().sum::<Duration>().as_secs_f64() / iterations as f64;

    println!("\nLatency Results:");
    println!("  Min:    {:>10.3} ms", min.as_secs_f64() * 1000.0);
    println!("  Mean:   {:>10.3} ms", mean * 1000.0);
    println!("  P50:    {:>10.3} ms", p50.as_secs_f64() * 1000.0);
    println!("  P95:    {:>10.3} ms", p95.as_secs_f64() * 1000.0);
    println!("  P99:    {:>10.3} ms", p99.as_secs_f64() * 1000.0);
    println!("  Max:    {:>10.3} ms", max.as_secs_f64() * 1000.0);

    Ok(())
}

/// Run throughput benchmark with concurrent clients
pub async fn run_throughput_benchmark(duration_secs: u64, clients: usize) -> Result<()> {
    let (port, _) = crate::cli::server::get_server_connection().await?;

    println!("Benchmarking throughput");
    println!("Duration: {} seconds", duration_secs);
    println!("Concurrent clients: {}\n", clients);

    let start = Instant::now();
    let target_duration = Duration::from_secs(duration_secs);

    let (tx, mut rx) = mpsc::channel(1000);

    // Spawn client tasks
    let mut handles = vec![];
    for client_id in 0..clients {
        let tx = tx.clone();
        let handle = tokio::spawn(async move {
            let mut ops = 0u64;
            let client_start = Instant::now();
            while client_start.elapsed() < target_duration {
                // Execute operation
                if execute_operation("create", port).await.is_ok() {
                    ops += 1;
                }
            }
            let _ = tx.send((client_id, ops)).await;
        });
        handles.push(handle);
    }

    drop(tx);

    // Collect results
    let mut total_ops = 0u64;
    let mut client_ops = vec![0u64; clients];
    while let Some((client_id, ops)) = rx.recv().await {
        total_ops += ops;
        client_ops[client_id] = ops;
    }

    // Wait for all tasks
    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let throughput = total_ops as f64 / elapsed;

    println!("\nThroughput Results:");
    println!("  Total operations: {}", total_ops);
    println!("  Duration: {:.2} seconds", elapsed);
    println!("  Throughput: {:.2} ops/sec", throughput);
    println!("  Per-client: {:.2} ops/sec", throughput / clients as f64);

    // Show per-client breakdown
    println!("\nPer-client operations:");
    for (i, ops) in client_ops.iter().enumerate() {
        println!(
            "  Client {}: {} ops ({:.2} ops/sec)",
            i,
            ops,
            *ops as f64 / elapsed
        );
    }

    Ok(())
}

/// Benchmark spreading activation performance
pub async fn run_spreading_benchmark(nodes: usize, depth: usize) -> Result<()> {
    let (_port, _) = crate::cli::server::get_server_connection().await?;

    println!("Benchmarking spreading activation");
    println!("  Nodes: {}", nodes);
    println!("  Depth: {}\n", depth);

    // TODO: Implement spreading activation benchmark
    // This requires API support for spreading activation queries
    println!("Note: Spreading activation benchmark requires API implementation");
    println!(
        "Placeholder: Would activate {} nodes to depth {}",
        nodes, depth
    );

    Ok(())
}

/// Benchmark memory consolidation
pub async fn run_consolidation_benchmark(load_test: bool) -> Result<()> {
    let (_port, _) = crate::cli::server::get_server_connection().await?;

    println!("Benchmarking memory consolidation");
    if load_test {
        println!("Load test mode enabled\n");
    }

    // TODO: Implement consolidation benchmark
    // This requires API support for triggering consolidation
    println!("Note: Consolidation benchmark requires API implementation");
    println!("Placeholder: Would benchmark consolidation cycles");

    Ok(())
}

async fn execute_operation(operation: &str, port: u16) -> Result<()> {
    let client = reqwest::Client::new();

    match operation {
        "create" => {
            let _response = client
                .post(format!("http://127.0.0.1:{}/api/v1/memories", port))
                .json(&serde_json::json!({
                    "what": "benchmark test memory",
                    "confidence": 0.9
                }))
                .send()
                .await
                .context("Failed to create memory")?;
        }
        "get" => {
            // Get status instead of specific memory (which may not exist)
            let _response = client
                .get(format!("http://127.0.0.1:{}/api/v1/status", port))
                .send()
                .await
                .context("Failed to get status")?;
        }
        "search" => {
            let _response = client
                .get(format!("http://127.0.0.1:{}/api/v1/memories/search", port))
                .query(&[("query", "test"), ("limit", "10")])
                .send()
                .await
                .context("Failed to search memories")?;
        }
        _ => anyhow::bail!("Unknown operation: {}", operation),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_latency_calculation() {
        let latencies = [
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];

        let mean = latencies.iter().sum::<Duration>().as_secs_f64() / latencies.len() as f64;
        assert!((mean - 0.030).abs() < 0.001); // 30ms mean
    }
}
