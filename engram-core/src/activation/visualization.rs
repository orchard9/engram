//! GraphViz rendering helpers for deterministic spreading traces.

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::sync::atomic::Ordering;

use super::{SpreadingResults, TraceEntry, storage_aware::StorageTier};

/// Configuration for DOT generation.
#[derive(Debug, Clone, Copy)]
pub struct DotOptions {
    /// Include a legend comment referencing documented assets.
    pub include_legend_note: bool,
}

impl Default for DotOptions {
    fn default() -> Self {
        Self {
            include_legend_note: true,
        }
    }
}

/// Render the provided spreading results into a GraphViz DOT string.
///
/// The renderer consumes activation summaries and deterministic traces where
/// available. Nodes are coloured by activation magnitude and edges encode the
/// confidence score.
#[must_use]
pub fn build_dot(
    results: &SpreadingResults,
    trace_override: Option<&[TraceEntry]>,
    options: DotOptions,
) -> String {
    let mut buffer = String::with_capacity(8 * 1024);

    writeln!(&mut buffer, "digraph SpreadingActivation {{").ok();
    writeln!(
        &mut buffer,
        "  graph [rankdir=LR, bgcolor=\"#ffffff\", fontname=\"Helvetica\"];"
    )
    .ok();
    writeln!(
        &mut buffer,
        "  node [shape=ellipse, style=filled, fontname=\"Helvetica\", color=\"#1f2933\"];"
    )
    .ok();
    writeln!(
        &mut buffer,
        "  edge [color=\"#111827\", fontname=\"Helvetica\", arrowsize=0.7];"
    )
    .ok();

    let mut node_map: HashMap<String, NodeAppearance> = HashMap::new();
    for activation in &results.activations {
        let node_id = activation.memory_id.clone();
        let activation_level = activation.activation_level.load(Ordering::Relaxed);
        let confidence = activation.confidence.load(Ordering::Relaxed);
        let tier = activation.storage_tier;
        node_map.insert(
            node_id,
            NodeAppearance {
                activation: activation_level,
                confidence,
                tier,
            },
        );
    }

    let trace = trace_override.unwrap_or(&results.deterministic_trace);

    // Merge deterministic trace data to ensure we cover every visited node.
    for entry in trace {
        let node_id = entry.target_node.clone();
        node_map
            .entry(node_id)
            .and_modify(|node| {
                // Retain maximum activation and confidence across sources.
                node.activation = node.activation.max(entry.activation);
                node.confidence = node.confidence.max(entry.confidence);
            })
            .or_insert_with(|| NodeAppearance {
                activation: entry.activation,
                confidence: entry.confidence,
                tier: StorageTier::from_depth(entry.depth),
            });
    }

    // Deterministic order for reproducible DOT snapshots.
    let mut node_ids: BTreeSet<String> = node_map.keys().cloned().collect();
    for entry in trace {
        if let Some(source) = &entry.source_node {
            node_ids.insert(source.clone());
        }
    }

    for node_id in node_ids {
        let appearance = node_map.entry(node_id.clone()).or_insert(NodeAppearance {
            activation: 0.0,
            confidence: 0.0,
            tier: StorageTier::from_depth(0),
        });

        let fill = activation_colour(appearance.activation);
        let label = format!(
            "{}\nactivation: {:.2}\nconfidence: {:.2}\ntier: {}",
            node_id,
            appearance.activation,
            appearance.confidence,
            tier_label(appearance.tier),
        );
        writeln!(
            &mut buffer,
            "  \"{id}\" [label=\"{label}\", fillcolor=\"{fill}\"];",
            id = escape_id(&node_id),
            label = escape_label(&label),
            fill = fill
        )
        .ok();
    }

    for entry in trace {
        if let Some(source) = &entry.source_node {
            let style = if entry.confidence < 0.8 {
                "dashed"
            } else {
                "solid"
            };
            let penwidth = 1.2_f32 + (entry.confidence * 2.0);
            let colour = if entry.confidence >= 0.8 {
                "#1f2933"
            } else {
                "#6b7280"
            };
            writeln!(
                &mut buffer,
                "  \"{src}\" -> \"{dst}\" [label=\"{conf:.2}\", color=\"{colour}\", penwidth={penwidth:.2}, style={style}];",
                src = escape_id(source),
                dst = escape_id(&entry.target_node),
                conf = entry.confidence,
                colour = colour,
                penwidth = penwidth,
                style = style,
            )
            .ok();
        }
    }

    if options.include_legend_note {
        writeln!(
            &mut buffer,
            "  // Legend reference: docs/assets/spreading_legend.svg"
        )
        .ok();
    }

    writeln!(&mut buffer, "}}").ok();
    buffer
}

#[derive(Debug, Clone, Copy)]
struct NodeAppearance {
    activation: f32,
    confidence: f32,
    tier: StorageTier,
}

fn activation_colour(value: f32) -> &'static str {
    if value >= 0.75 {
        "#1338bf"
    } else if value >= 0.35 {
        "#6ba4ff"
    } else {
        "#dbeafe"
    }
}

const fn tier_label(tier: StorageTier) -> &'static str {
    match tier {
        StorageTier::Hot => "hot",
        StorageTier::Warm => "warm",
        StorageTier::Cold => "cold",
    }
}

fn escape_id(text: &str) -> String {
    text.replace('"', "\\\"")
}

fn escape_label(text: &str) -> String {
    text.replace('"', "\\\"")
}

/// Convenience helper that uses the deterministic trace embedded in the
/// `SpreadingResults`.
#[must_use]
pub fn build_dot_from_results(results: &SpreadingResults) -> String {
    build_dot(results, None, DotOptions::default())
}

/// Build a DOT graph using a caller-supplied trace rather than the one stored
/// on the results structure.
#[must_use]
pub fn build_dot_with_trace(results: &SpreadingResults, trace: &[TraceEntry]) -> String {
    build_dot(results, Some(trace), DotOptions::default())
}
