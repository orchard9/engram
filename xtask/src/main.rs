use serde::Deserialize;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_RESULTS_DIR: &str = "docs/assets/benchmarks/spreading";
const DEFAULT_BASELINE_PATH: &str = "docs/assets/benchmarks/spreading/baseline.json";
const DEFAULT_TOLERANCE: f64 = 0.10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("update-spreading-snapshots") => update_spreading_snapshots()?,
        Some("check-spreading-benchmarks") => {
            let mut baseline_path: Option<PathBuf> = None;
            let mut results_dir: Option<PathBuf> = None;

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--baseline" => {
                        let value = args
                            .next()
                            .ok_or_else(|| "--baseline expects a path argument".to_string())?;
                        baseline_path = Some(PathBuf::from(value));
                    }
                    "--results" => {
                        let value = args
                            .next()
                            .ok_or_else(|| "--results expects a path argument".to_string())?;
                        results_dir = Some(PathBuf::from(value));
                    }
                    "--help" | "-h" => {
                        print_check_help();
                        return Ok(());
                    }
                    other => {
                        exit_with_err(format!(
                            "unknown flag for check-spreading-benchmarks: {other}"
                        ));
                    }
                }
            }

            check_spreading_benchmarks(baseline_path, results_dir)?;
        }
        Some(cmd) => return Err(format!("unknown xtask command: {cmd}").into()),
        None => {
            eprintln!(
                "available xtask commands:\n  update-spreading-snapshots        Refresh insta YAML baselines for spreading validation\n  check-spreading-benchmarks       Compare current Criterion results against baseline (median & P95 drift)\n"
            );
        }
    }

    Ok(())
}

fn exit_with_err(message: impl AsRef<str>) -> ! {
    eprintln!("error: {}", message.as_ref());
    std::process::exit(1);
}

fn print_check_help() {
    println!(
        "Usage: cargo xtask check-spreading-benchmarks [--baseline <path>] [--results <dir>]\n\n\
         Compares the most recent Criterion benchmark run against the stored baseline.\n\n\
         Options:\n\
           --baseline <path>   Path to baseline JSON (default: {DEFAULT_BASELINE_PATH})\n\
           --results <dir>     Directory containing Criterion outputs (default: {DEFAULT_RESULTS_DIR})\n"
    );
}

fn update_spreading_snapshots() -> Result<(), String> {
    let mut command = Command::new("cargo");
    command
        .env("INSTA_FORCE_UPDATE", "1")
        .env("CARGO_NET_OFFLINE", "true")
        .args([
            "--offline",
            "test",
            "-p",
            "engram-core",
            "--test",
            "spreading_validation",
            "canonical_spreading_snapshots_are_stable",
            "--",
            "--nocapture",
        ]);

    let status = command
        .status()
        .map_err(|err| format!("failed to execute cargo test for snapshots: {err}"))?;
    if !status.success() {
        return Err("snapshot regeneration failed".into());
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct BaselineEntry {
    #[serde(default)]
    median_ns: Option<f64>,
    #[serde(default)]
    p95_ns: Option<f64>,
    #[serde(default = "default_tolerance")] // allow override per scenario
    tolerance: f64,
}

#[derive(Debug, Clone, Copy)]
struct ScenarioMetrics {
    median_ns: f64,
    p95_ns: f64,
}

fn default_tolerance() -> f64 {
    DEFAULT_TOLERANCE
}

fn check_spreading_benchmarks(
    baseline_path: Option<PathBuf>,
    results_dir: Option<PathBuf>,
) -> Result<(), String> {
    let baseline_path = baseline_path.unwrap_or_else(|| PathBuf::from(DEFAULT_BASELINE_PATH));
    let results_dir = results_dir.unwrap_or_else(|| PathBuf::from(DEFAULT_RESULTS_DIR));
    let baseline_path_display = baseline_path.display().to_string();

    let baseline_data = fs::read_to_string(&baseline_path)
        .map_err(|err| format!("failed to read baseline {}: {err}", baseline_path_display))?;
    let baseline: HashMap<String, BaselineEntry> = serde_json::from_str(&baseline_data)
        .map_err(|err| format!("failed to parse baseline {}: {err}", baseline_path_display))?;

    if baseline.is_empty() {
        return Err(format!(
            "baseline {} does not contain any scenarios",
            baseline_path_display
        ));
    }

    let actuals = collect_benchmark_results(&results_dir)?;

    let mut regressions = Vec::new();

    for (scenario, entry) in baseline.iter() {
        let Some(actual) = actuals.get(scenario) else {
            regressions.push(format!(
                "baseline scenario '{scenario}' missing from benchmark outputs in {}",
                results_dir.display()
            ));
            continue;
        };

        let mut missing_metrics = false;

        match entry.median_ns {
            Some(baseline_median) if baseline_median > 0.0 => {
                let allowed = baseline_median * (1.0 + entry.tolerance);
                if actual.median_ns > allowed {
                    let drift = ((actual.median_ns / baseline_median) - 1.0) * 100.0;
                    regressions.push(format!(
                        "{scenario}: median regression {drift:.2}% (baseline {:.2} ns vs current {:.2} ns)",
                        baseline_median,
                        actual.median_ns
                    ));
                }
            }
            Some(_) => {}
            None => {
                missing_metrics = true;
            }
        }

        match entry.p95_ns {
            Some(baseline_p95) if baseline_p95 > 0.0 => {
                let allowed = baseline_p95 * (1.0 + entry.tolerance);
                if actual.p95_ns > allowed {
                    let drift = ((actual.p95_ns / baseline_p95) - 1.0) * 100.0;
                    regressions.push(format!(
                        "{scenario}: p95 regression {drift:.2}% (baseline {:.2} ns vs current {:.2} ns)",
                        baseline_p95,
                        actual.p95_ns
                    ));
                }
            }
            Some(_) => {}
            None => {
                missing_metrics = true;
            }
        }

        if missing_metrics {
            regressions.push(format!(
                "baseline for '{scenario}' lacks median/p95 values – update {}",
                baseline_path_display
            ));
        }
    }

    for scenario in actuals.keys() {
        if !baseline.contains_key(scenario) {
            regressions.push(format!(
                "new benchmark scenario '{scenario}' detected – add to baseline {}",
                baseline_path_display
            ));
        }
    }

    if regressions.is_empty() {
        println!(
            "Benchmark drift check passed for {} scenarios",
            baseline.len()
        );
        Ok(())
    } else {
        Err(regressions.join("\n"))
    }
}

fn collect_benchmark_results(
    results_dir: &Path,
) -> Result<HashMap<String, ScenarioMetrics>, String> {
    let mut stack = vec![results_dir.to_path_buf()];
    let mut metrics = HashMap::new();

    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|err| format!("failed to read directory {}: {err}", dir.display()))?;
        for entry in entries {
            let entry =
                entry.map_err(|err| format!("failed to read entry in {}: {err}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                if path.file_name() == Some(OsStr::new("new")) {
                    if let Some(result) = parse_benchmark_dir(&path)? {
                        let key = normalize_scenario_name(&result.0);
                        metrics.insert(key, result.1);
                    }
                } else {
                    stack.push(path);
                }
            }
        }
    }

    Ok(metrics)
}

fn parse_benchmark_dir(path: &Path) -> Result<Option<(String, ScenarioMetrics)>, String> {
    let Some(scenario_dir) = path.parent().and_then(|p| p.file_name()) else {
        return Ok(None);
    };
    let scenario_dir = scenario_dir.to_string_lossy().to_string();

    let estimates_path = path.join("estimates.json");
    let sample_path = path.join("sample.json");

    if !estimates_path.exists() {
        return Ok(None);
    }

    let estimates_str = fs::read_to_string(&estimates_path).map_err(|err| {
        format!(
            "failed to read estimates.json for {}: {err}",
            estimates_path.display()
        )
    })?;
    let estimates: Estimates = serde_json::from_str(&estimates_str).map_err(|err| {
        format!(
            "failed to parse estimates.json for {}: {err}",
            estimates_path.display()
        )
    })?;

    let median_ns = estimates.median.point_estimate * 1_000_000_000.0;

    let p95_ns = if sample_path.exists() {
        let sample_str = fs::read_to_string(&sample_path).map_err(|err| {
            format!(
                "failed to read sample.json for {}: {err}",
                sample_path.display()
            )
        })?;
        let mut samples: Vec<f64> = serde_json::from_str(&sample_str).map_err(|err| {
            format!(
                "failed to parse sample.json for {}: {err}",
                sample_path.display()
            )
        })?;
        if samples.is_empty() {
            return Err(format!(
                "benchmark sample for {} is empty",
                sample_path.display()
            ));
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
        let idx = percentile_index(samples.len(), 0.95);
        samples[idx] * 1_000_000_000.0
    } else {
        return Err(format!(
            "sample.json missing for {} (required to compute P95)",
            path.display()
        ));
    };

    Ok(Some((scenario_dir, ScenarioMetrics { median_ns, p95_ns })))
}

#[derive(Debug, Deserialize)]
struct Estimate {
    point_estimate: f64,
}

#[derive(Debug, Deserialize)]
struct Estimates {
    median: Estimate,
}

fn normalize_scenario_name(raw: &str) -> String {
    raw.trim_start_matches("scenario_").to_string()
}

fn percentile_index(len: usize, quantile: f64) -> usize {
    if len == 0 {
        return 0;
    }
    let position = (len as f64 - 1.0) * quantile;
    position.round().clamp(0.0, (len - 1) as f64) as usize
}
