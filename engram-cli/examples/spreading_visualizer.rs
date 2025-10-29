//! Generate a GraphViz DOT diagram from deterministic spreading activation traces.
//!
//! Usage examples:
//!
//! ```bash
//! # Generate DOT from a built-in seeded recall
//! cargo run -p engram-cli --example spreading_visualizer --features hnsw_index
//!
//! # Convert an existing trace JSON file into DOT and render PNG
//! cargo run -p engram-cli --example spreading_visualizer \
//!     --features hnsw_index \
//!     -- --input-trace docs/assets/spreading/trace_samples/seed_42_trace.json \
//!        --output target/spread.dot \
//!        --render-png target/spread.png
//! ```

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value;

#[cfg(feature = "hnsw_index")]
use engram_core::activation::{SpreadingResults, TraceEntry};

#[cfg(not(feature = "hnsw_index"))]
fn main() {
    eprintln!("This example requires the 'hnsw_index' feature to be enabled.");
    eprintln!(
        "Run with: cargo run -p engram-cli --example spreading_visualizer --features hnsw_index"
    );
}

#[cfg(feature = "hnsw_index")]
fn main() -> Result<()> {
    let args = CliArgs::parse()?;

    let results = if let Some(trace_path) = &args.input_trace {
        load_trace(trace_path)?
    } else {
        run_seeded_recall()?
    };

    let dot = engram_core::activation::visualization::build_dot(
        &results,
        None,
        engram_core::activation::visualization::DotOptions::default(),
    );

    fs::write(&args.output, dot.as_bytes())
        .with_context(|| format!("failed to write DOT output to {}", args.output.display()))?;

    println!("DOT written to {}", args.output.display());

    if let Some(png_path) = &args.render_png {
        render_png(&args.output, png_path)?;
    }

    if results.deterministic_trace.is_empty() {
        println!("(trace empty – enable trace_activation_flow to populate it)");
    } else {
        println!(
            "trace entries exported: {}",
            results.deterministic_trace.len()
        );
    }

    Ok(())
}

#[cfg(feature = "hnsw_index")]
struct CliArgs {
    output: PathBuf,
    input_trace: Option<PathBuf>,
    render_png: Option<PathBuf>,
}

#[cfg(feature = "hnsw_index")]
impl CliArgs {
    fn parse() -> Result<Self> {
        let mut output = PathBuf::from("spreading.dot");
        let mut input_trace = None;
        let mut render_png = None;

        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--output" => {
                    let path = iter
                        .next()
                        .ok_or_else(|| anyhow!("--output requires a path"))?;
                    output = PathBuf::from(path);
                }
                "--input-trace" => {
                    let path = iter
                        .next()
                        .ok_or_else(|| anyhow!("--input-trace requires a path"))?;
                    input_trace = Some(PathBuf::from(path));
                }
                "--render-png" => {
                    let path = iter
                        .next()
                        .ok_or_else(|| anyhow!("--render-png requires a path"))?;
                    render_png = Some(PathBuf::from(path));
                }
                other => return Err(anyhow!("unknown argument: {other}")),
            }
        }

        Ok(Self {
            output,
            input_trace,
            render_png,
        })
    }
}

#[cfg(feature = "hnsw_index")]
fn run_seeded_recall() -> Result<SpreadingResults> {
    use chrono::Utc;
    use engram_core::activation::{
        ActivationGraphExt, EdgeType, ParallelSpreadingConfig, create_activation_graph,
        parallel::ParallelSpreadingEngine, seeding::VectorActivationSeeder,
        similarity_config::SimilarityConfig,
    };
    use engram_core::{Confidence, Cue, EpisodeBuilder, MemoryStore};
    use std::sync::Arc;

    let store = MemoryStore::new(128).with_hnsw_index();

    for (id, content, magnitude, confidence) in [
        (
            "doctor_harmon",
            "Dr. Harmon triages patient Lucy",
            0.90,
            0.88,
        ),
        (
            "nurse_lucy",
            "Nurse Lucy schedules the cardiology consult",
            0.88,
            0.86,
        ),
        (
            "cardiology_consult",
            "Cardiology reviews heart rate telemetry",
            0.65,
            0.74,
        ),
        (
            "follow_up_call",
            "Lucy confirms follow-up appointment over the phone",
            0.76,
            0.80,
        ),
    ] {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(content.to_string())
            .embedding(uniform_embedding(magnitude))
            .confidence(Confidence::from_raw(confidence))
            .build();
        store.store(episode);
    }

    let index = store
        .hnsw_index()
        .expect("HNSW index should be available with the feature enabled");

    let graph = Arc::new(create_activation_graph());

    for (source, target, weight) in [
        ("doctor_harmon", "nurse_lucy", 0.92),
        ("doctor_harmon", "cardiology_consult", 0.78),
        ("nurse_lucy", "follow_up_call", 0.86),
        ("cardiology_consult", "follow_up_call", 0.74),
    ] {
        ActivationGraphExt::add_edge(
            &*graph,
            source.to_string(),
            target.to_string(),
            weight,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            target.to_string(),
            source.to_string(),
            weight * 0.9,
            EdgeType::Excitatory,
        );
    }

    for (node, magnitude) in [
        ("doctor_harmon", 0.90),
        ("nurse_lucy", 0.88),
        ("cardiology_consult", 0.65),
        ("follow_up_call", 0.76),
    ] {
        ActivationGraphExt::set_embedding(
            &*graph,
            &node.to_string(),
            &uniform_embedding(magnitude),
        );
    }

    let seeder = VectorActivationSeeder::with_default_resolver(index, SimilarityConfig::default());

    let spreading_config = ParallelSpreadingConfig {
        deterministic: true,
        seed: Some(42),
        trace_activation_flow: true,
        max_depth: 3,
        threshold: 0.08,
        ..ParallelSpreadingConfig::default()
    };

    let spreading_engine = ParallelSpreadingEngine::new(spreading_config, graph)
        .expect("failed to initialise spreading engine");

    let cue = Cue::embedding(
        "doctor_cue".to_string(),
        uniform_embedding(0.9),
        Confidence::from_raw(0.92),
    );

    let seeding_outcome = seeder
        .seed_from_cue(&cue)
        .context("failed to seed activation from cue")?;

    let seeds: Vec<(String, f32)> = seeding_outcome
        .seeds
        .iter()
        .map(|seed| (seed.memory_id.clone(), seed.activation))
        .collect();

    if seeds.is_empty() {
        return Err(anyhow!("cue did not generate activation seeds"));
    }

    spreading_engine
        .spread_activation(&seeds)
        .context("spreading activation failed")
}

#[cfg(feature = "hnsw_index")]
fn load_trace(path: &Path) -> Result<SpreadingResults> {
    let payload = fs::read_to_string(path)
        .with_context(|| format!("failed to read trace file {}", path.display()))?;

    let value: Value = serde_json::from_str(&payload)
        .with_context(|| format!("invalid JSON in {}", path.display()))?;

    let entries = if let Some(trace) = value.get("trace") {
        parse_trace_entries(trace)?
    } else {
        parse_trace_entries(&value)?
    };

    Ok(SpreadingResults {
        activations: Vec::new(),
        tier_summaries: HashMap::new(),
        cycle_paths: Vec::new(),
        deterministic_trace: entries,
    })
}

#[cfg(feature = "hnsw_index")]
fn parse_trace_entries(value: &Value) -> Result<Vec<TraceEntry>> {
    if value.is_array() {
        let entries: Vec<TraceEntryDto> = serde_json::from_value(value.clone())?;
        return Ok(entries.into_iter().map(TraceEntry::from).collect());
    }

    Err(anyhow!("expected an array of trace entries"))
}

#[cfg(feature = "hnsw_index")]
fn render_png(dot_path: &Path, png_path: &Path) -> Result<()> {
    let status = Command::new("dot")
        .arg("-Tpng")
        .arg(dot_path)
        .arg("-o")
        .arg(png_path)
        .status();

    match status {
        Ok(exit) if exit.success() => {
            println!("PNG rendered to {}", png_path.display());
            Ok(())
        }
        Ok(exit) => Err(anyhow!(
            "GraphViz 'dot' exited with status {}",
            exit.code().unwrap_or(-1)
        )),
        Err(err) => Err(anyhow!("failed to invoke GraphViz 'dot': {err}")),
    }
}

#[cfg(feature = "hnsw_index")]
fn uniform_embedding(value: f32) -> [f32; 768] {
    let mut embedding = [value; 768];
    embedding[0] = value;
    embedding[1] = value * 0.95;
    embedding[2] = value * 0.9;
    embedding
}

#[cfg(feature = "hnsw_index")]
#[derive(Debug, Deserialize)]
struct TraceEntryDto {
    depth: u16,
    #[serde(alias = "source", alias = "source_node")]
    source_node: Option<String>,
    #[serde(alias = "target", alias = "target_node")]
    target_node: String,
    activation: f32,
    confidence: f32,
}

#[cfg(feature = "hnsw_index")]
impl From<TraceEntryDto> for TraceEntry {
    fn from(dto: TraceEntryDto) -> Self {
        Self {
            depth: dto.depth,
            target_node: dto.target_node,
            activation: dto.activation,
            confidence: dto.confidence,
            source_node: dto.source_node,
        }
    }
}
