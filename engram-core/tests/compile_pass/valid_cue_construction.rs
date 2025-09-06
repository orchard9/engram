//! Test valid Cue construction patterns that should compile successfully
//!
//! These tests demonstrate correct usage of the Cue builder typestate pattern
//! for creating memory retrieval queries with different search strategies.

use engram_core::memory::{CueBuilder, TemporalPattern};
use engram_core::Confidence;
use chrono::{Utc, Duration};

fn main() {
    let now = Utc::now();
    let past = now - Duration::hours(24);

    // Test 1: Basic embedding-based cue
    let _embedding_cue = CueBuilder::new()
        .id("embedding_search".to_string())
        .embedding_search([0.1f32; 768], Confidence::MEDIUM)
        .build();

    // Test 2: Context-based cue with time range
    let _context_cue = CueBuilder::new()
        .id("context_search".to_string())
        .context_search(
            Some((past, now)),
            Some("Conference Room".to_string()),
            Confidence::HIGH
        )
        .build();

    // Test 3: Semantic content search
    let _semantic_cue = CueBuilder::new()
        .id("semantic_search".to_string())
        .semantic_search("project planning meeting".to_string(), Confidence::LOW)
        .build();

    // Test 4: Temporal pattern search - recent events
    let _temporal_recent = CueBuilder::new()
        .id("recent_events".to_string())
        .temporal_search(
            TemporalPattern::Recent(Duration::hours(12)),
            Confidence::MEDIUM
        )
        .build();

    // Test 5: Temporal pattern search - events before a time
    let _temporal_before = CueBuilder::new()
        .id("events_before".to_string())
        .temporal_search(
            TemporalPattern::Before(now),
            Confidence::HIGH
        )
        .build();

    // Test 6: Temporal pattern search - events after a time
    let _temporal_after = CueBuilder::new()
        .id("events_after".to_string())
        .temporal_search(
            TemporalPattern::After(past),
            Confidence::MEDIUM
        )
        .build();

    // Test 7: Temporal pattern search - events in range
    let _temporal_between = CueBuilder::new()
        .id("events_between".to_string())
        .temporal_search(
            TemporalPattern::Between(past, now),
            Confidence::HIGH
        )
        .build();

    // Test 8: Cue with customized parameters
    let _custom_cue = CueBuilder::new()
        .id("custom_parameters".to_string())
        .embedding_search([0.5f32; 768], Confidence::HIGH)
        .cue_confidence(Confidence::MEDIUM)
        .result_threshold(Confidence::LOW)
        .max_results(50)
        .build();

    // Test 9: Context search without location
    let _time_only_context = CueBuilder::new()
        .id("time_only".to_string())
        .context_search(Some((past, now)), None, Confidence::MEDIUM)
        .build();

    // Test 10: Context search without time range
    let _location_only_context = CueBuilder::new()
        .id("location_only".to_string())
        .context_search(None, Some("Library".to_string()), Confidence::HIGH)
        .build();

    // Test 11: Multiple cues for different search strategies
    let search_strategies = vec![
        ("strategy_1", "find recent meetings"),
        ("strategy_2", "locate project documents"),
        ("strategy_3", "search presentation notes"),
    ];

    for (id, query) in search_strategies {
        let _cue = CueBuilder::new()
            .id(id.to_string())
            .semantic_search(query.to_string(), Confidence::MEDIUM)
            .max_results(25)
            .build();
    }

    // All cue constructions demonstrate different valid search patterns
    // that the typestate pattern supports while maintaining type safety
}