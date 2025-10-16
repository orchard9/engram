#![allow(missing_docs)]

use engram_core::activation::{
    SpreadingResults, TraceEntry,
    visualization::{DotOptions, build_dot},
};

#[test]
fn generates_deterministic_dot_snapshot() {
    let results = SpreadingResults {
        deterministic_trace: vec![
            TraceEntry {
                depth: 0,
                target_node: "doctor".to_string(),
                activation: 1.0,
                confidence: 0.9,
                source_node: None,
            },
            TraceEntry {
                depth: 1,
                target_node: "nurse".to_string(),
                activation: 0.7,
                confidence: 0.8,
                source_node: Some("doctor".to_string()),
            },
        ],
        ..Default::default()
    };

    let dot = build_dot(&results, None, DotOptions::default());
    insta::assert_snapshot!(
        dot,
        @r###"digraph SpreadingActivation {
  graph [rankdir=LR, bgcolor="#ffffff", fontname="Helvetica"];
  node [shape=ellipse, style=filled, fontname="Helvetica", color="#1f2933"];
  edge [color="#111827", fontname="Helvetica", arrowsize=0.7];
  "doctor" [label="doctor
activation: 1.00
confidence: 0.90
tier: hot", fillcolor="#1338bf"];
  "nurse" [label="nurse
activation: 0.70
confidence: 0.80
tier: warm", fillcolor="#6ba4ff"];
  "doctor" -> "nurse" [label="0.80", color="#1f2933", penwidth=2.80, style=solid];
  // Legend reference: docs/assets/spreading_legend.svg
}
"###
    );
}
