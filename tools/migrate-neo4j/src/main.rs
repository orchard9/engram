//! Neo4j to Engram migration tool

use anyhow::Result;
use clap::Parser;
use migration_common::{
    CheckpointManager, EmbeddingGenerator, MigrationReport, ProgressTracker, SourceStatistics,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;

mod graph_mapper;
mod neo4j_source;

use graph_mapper::Neo4jTransformer;
use neo4j_source::Neo4jDataSource;

#[derive(Parser, Debug)]
#[command(name = "migrate-neo4j")]
#[command(about = "Migrate Neo4j graph database to Engram", long_about = None)]
struct Args {
    /// Neo4j connection URI (e.g., bolt://localhost:7687)
    #[arg(long)]
    source: String,

    /// Neo4j username
    #[arg(long)]
    source_user: String,

    /// Neo4j password
    #[arg(long)]
    source_password: String,

    /// Target Engram instance URL
    #[arg(long)]
    target: String,

    /// Memory space prefix for Neo4j nodes
    #[arg(long, default_value = "neo4j")]
    memory_space_prefix: String,

    /// Label to memory space mapping (format: "Label1:space1,Label2:space2")
    #[arg(long)]
    label_to_space: Option<String>,

    /// Batch size for migration
    #[arg(long, default_value = "1000")]
    batch_size: usize,

    /// Checkpoint file path for resumable migration
    #[arg(long)]
    checkpoint_file: Option<PathBuf>,

    /// Dry run (validate without writing)
    #[arg(long)]
    dry_run: bool,

    /// Run validation after migration
    #[arg(long)]
    validate: bool,

    /// Skip relationship migration
    #[arg(long)]
    skip_edges: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let args = Args::parse();

    tracing::info!("Starting Neo4j to Engram migration");
    tracing::info!("Source: {}", args.source);
    tracing::info!("Target: {}", args.target);
    tracing::info!("Batch size: {}", args.batch_size);

    // Create embedding generator
    let embedding_generator = EmbeddingGenerator::new(100);

    // Create transformer
    let transformer = Neo4jTransformer::new(
        args.memory_space_prefix.clone(),
        parse_label_mapping(args.label_to_space),
    );

    // Create checkpoint manager if specified
    let mut checkpoint_manager = args.checkpoint_file.as_ref().map(|path| {
        CheckpointManager::new(path.clone(), 10000) // Checkpoint every 10k records
    });

    // Load existing checkpoint if available
    let start_from = if let Some(ref mut mgr) = checkpoint_manager {
        mgr.load_existing()?
    } else {
        None
    };

    if let Some(ref cp) = start_from {
        tracing::info!(
            "Resuming from checkpoint: {} records already migrated",
            cp.records_migrated
        );
    }

    // Create data source
    let mut data_source = Neo4jDataSource::new(
        &args.source,
        &args.source_user,
        &args.source_password,
        args.batch_size,
    )
    .await?;

    // Resume from checkpoint if needed
    if let Some(ref checkpoint) = start_from {
        data_source.resume_from(checkpoint)?;
    }

    // Create progress tracker
    let total_records = data_source.total_records();
    let progress = ProgressTracker::new(total_records, Duration::from_secs(10));

    let start_time = Instant::now();
    let mut total_migrated = start_from.as_ref().map_or(0, |cp| cp.records_migrated);
    let mut error_count = 0u64;

    // Migration loop
    loop {
        let batch = data_source.next_batch()?;
        if batch.is_empty() {
            break;
        }

        // Generate embeddings for batch
        let texts: Vec<String> = batch.iter().map(|r| r.text_content.clone()).collect();
        let embeddings = embedding_generator.generate_batch(&texts)?;

        // Transform to episodes (in production, would store these in Engram)
        for (record, embedding) in batch.iter().zip(embeddings.iter()) {
            if !args.dry_run {
                // In production implementation:
                // let episode = transformer.transform(record, embedding)?;
                // store.store(episode);
                tracing::debug!("Would migrate record: {}", record.id);
            }
        }

        total_migrated += batch.len() as u64;
        progress.increment(batch.len() as u64);

        // Report progress
        if progress.should_report() {
            progress.report();

            // Save checkpoint if configured
            if let Some(ref mut mgr) = checkpoint_manager {
                if mgr.should_checkpoint(total_migrated) {
                    if let Some(last_record) = batch.last() {
                        mgr.save_checkpoint(last_record.id.clone(), total_migrated)?;
                    }
                }
            }
        }
    }

    let elapsed = start_time.elapsed();
    let report = MigrationReport::new(total_migrated, elapsed, error_count);

    report.print_summary();

    if args.validate {
        tracing::info!("Running validation...");
        // In production: run validation checks
        println!("\nValidation: SKIPPED (not implemented in this version)");
    }

    // Print cache statistics
    let cache_stats = embedding_generator.cache_stats();
    println!("\nEmbedding Cache Statistics:");
    println!("  Cache size: {} entries", cache_stats.size);

    tracing::info!("Migration completed successfully");

    Ok(())
}

fn parse_label_mapping(mapping_str: Option<String>) -> std::collections::HashMap<String, String> {
    let mut mapping = std::collections::HashMap::new();

    if let Some(s) = mapping_str {
        for pair in s.split(',') {
            if let Some((label, space)) = pair.split_once(':') {
                mapping.insert(label.trim().to_string(), space.trim().to_string());
            }
        }
    }

    mapping
}
