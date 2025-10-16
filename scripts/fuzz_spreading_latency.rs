use clap::Parser;
use engram_core::activation::{
    create_activation_graph, BreakerSettings, ParallelSpreadingConfig, ParallelSpreadingEngine,
    SpreadingAutoTuner,
};
use engram_core::activation::circuit_breaker::SpreadingCircuitBreaker;
use engram_core::activation::storage_aware::StorageTier;
use engram_core::metrics;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

#[derive(Parser, Debug)]
#[command(about = "Chaos harness for spreading activation latency/failure injection", author, version)]
struct ChaosArgs {
    /// Number of simulated seconds to run.
    #[arg(long = "duration", default_value_t = 60)]
    duration_seconds: u64,

    /// Additional latency (milliseconds) injected during the spike window.
    #[arg(long = "latency-spike", default_value_t = 15.0)]
    latency_spike_ms: f64,

    /// Failure rate applied during the spike window (0.0 - 1.0).
    #[arg(long = "failure-rate", default_value_t = 0.12)]
    failure_rate: f64,

    /// Second at which the spike window begins.
    #[arg(long = "spike-start", default_value_t = 10)]
    spike_start: u64,

    /// Duration of the spike window in seconds.
    #[arg(long = "spike-duration", default_value_t = 20)]
    spike_duration: u64,

    /// Samples per simulated second.
    #[arg(long = "samples", default_value_t = 200)]
    samples_per_second: u64,

    /// RNG seed for reproducibility.
    #[arg(long = "seed", default_value_t = 42)]
    seed: u64,
}

fn main() {
    let args = ChaosArgs::parse();
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    if !(0.0..=1.0).contains(&args.failure_rate) {
        eprintln!("failure-rate must be between 0.0 and 1.0");
        std::process::exit(2);
    }

    let metrics = metrics::init();
    let graph = Arc::new(create_activation_graph());
    let engine = Arc::new(
        ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph)
            .expect("failed to create spreading engine"),
    );

    let metrics_handle = engine.metrics_handle();
    let breaker_settings = BreakerSettings::default();
    let breaker = SpreadingCircuitBreaker::new(metrics_handle, &breaker_settings);
    let auto_tuner = SpreadingAutoTuner::new(0.10, 32);

    let mut rng = StdRng::seed_from_u64(args.seed);

    let budget = Duration::from_millis(10);
    let base_hot = Duration::from_micros(80);
    let base_warm = Duration::from_micros(800);
    let base_cold = Duration::from_millis(8);
    let spike_delta = Duration::from_micros((args.latency_spike_ms * 1_000.0) as u64);

    let spike_end = args.spike_start.saturating_add(args.spike_duration);
    let denominator = if args.failure_rate <= f64::EPSILON {
        u64::MAX
    } else {
        (1.0 / args.failure_rate).round().max(1.0) as u64
    };

    info!(
        duration = args.duration_seconds,
        latency_spike_ms = args.latency_spike_ms,
        failure_rate = args.failure_rate,
        samples = args.samples_per_second,
        "starting chaos run"
    );

    let spreading_metrics = engine.get_metrics();
    for second in 0..args.duration_seconds {
        let in_spike = second >= args.spike_start && second < spike_end;
        for offset in 0..args.samples_per_second {
            let event_index = second * args.samples_per_second + offset;

            let jitter = Duration::from_micros(rng.gen_range(0..30));
            let hot_latency = if in_spike {
                base_hot + spike_delta + jitter
            } else {
                base_hot + jitter
            };
            let warm_latency = if in_spike {
                base_warm + spike_delta + jitter
            } else {
                base_warm + jitter
            };
            let cold_latency = if in_spike {
                base_cold + spike_delta + jitter
            } else {
                base_cold + jitter
            };

            spreading_metrics.record_activation_latency(StorageTier::Hot, hot_latency);
            spreading_metrics.record_activation_latency(StorageTier::Warm, warm_latency);
            spreading_metrics.record_activation_latency(StorageTier::Cold, cold_latency);

            let should_attempt = breaker.should_attempt();
            let should_fail = in_spike && event_index % denominator == 0;

            if !should_attempt {
                spreading_metrics.record_fallback();
                continue;
            }

            if should_fail {
                spreading_metrics.record_spread_failure();
                spreading_metrics.record_latency_budget_violation();
                breaker.on_result(false, hot_latency, budget);
            } else {
                breaker.on_result(true, hot_latency, budget);
            }
        }
    }

    // Drain metrics into a snapshot and run auto-tune evaluation for reporting.
    let snapshot = metrics.streaming_snapshot();
    let applied_change = snapshot.spreading.map_or(None, |summary| auto_tuner.evaluate(&summary, &engine));

    let final_snapshot = metrics.streaming_snapshot();
    if let Some(summary) = final_snapshot.spreading.as_ref() {
        if let Some(hot) = summary.per_tier.get("hot") {
            println!(
                "Hot tier latency p95: {:.4}s (samples: {})",
                hot.p95_seconds, hot.samples
            );
        }
        if let Some(open_state) = summary.breaker_state {
            println!("Breaker state code: {open_state}");
        }
    }

    let breaker_transitions = spreading_metrics.breaker_transitions();

    println!("Breaker transitions recorded: {breaker_transitions}");
    if let Some(change) = applied_change {
        println!(
            "Auto-tune applied change on tier {}: batch {}→{}, depth {}→{}, timeout {:.4}s→{:.4}s",
            change.tier,
            change.batch_size_before,
            change.batch_size_after,
            change.max_depth_before,
            change.max_depth_after,
            change.timeout_before_seconds,
            change.timeout_after_seconds
        );
    } else {
        println!("Auto-tune did not observe a triggering condition during the run");
    }

    if breaker_transitions == 0 {
        eprintln!("breaker never opened during chaos run");
        std::process::exit(3);
    }
}
