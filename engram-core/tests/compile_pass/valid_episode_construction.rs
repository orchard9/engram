//! Test valid Episode construction patterns that should compile successfully
//!
//! These tests demonstrate correct usage of the Episode builder typestate pattern
//! for creating episodic memories with temporal and contextual information.

use engram_core::memory::EpisodeBuilder;
use engram_core::Confidence;
use chrono::{Utc, Duration};

fn main() {
    let now = Utc::now();
    let past = now - Duration::hours(2);

    // Test 1: Basic valid Episode construction
    let _basic_episode = EpisodeBuilder::new()
        .id("basic_episode".to_string())
        .when(now)
        .what("Something interesting happened".to_string())
        .embedding([0.1f32; 768])
        .confidence(Confidence::HIGH)
        .build();

    // Test 2: Episode with location information
    let _located_episode = EpisodeBuilder::new()
        .id("located_episode".to_string())
        .when(past)
        .what("A meeting took place".to_string())
        .embedding([0.2f32; 768])
        .confidence(Confidence::MEDIUM)
        .where_location("Conference Room A".to_string())
        .build();

    // Test 3: Episode with participants
    let _social_episode = EpisodeBuilder::new()
        .id("social_episode".to_string())
        .when(now)
        .what("Team discussion about project".to_string())
        .embedding([0.3f32; 768])
        .confidence(Confidence::HIGH)
        .who(vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()])
        .build();

    // Test 4: Complete episode with all optional fields
    let _complete_episode = EpisodeBuilder::new()
        .id("complete_episode".to_string())
        .when(past)
        .what("Comprehensive project review meeting".to_string())
        .embedding([0.4f32; 768])
        .confidence(Confidence::exact(0.85))
        .where_location("Main Conference Room".to_string())
        .who(vec![
            "Project Manager".to_string(),
            "Lead Developer".to_string(),
            "Designer".to_string(),
        ])
        .decay_rate(0.03)
        .build();

    // Test 5: Multiple episodes with different timestamps
    for i in 0..3 {
        let timestamp = now - Duration::hours(i as i64);
        let _episode = EpisodeBuilder::new()
            .id(format!("episode_{}", i))
            .when(timestamp)
            .what(format!("Event {} occurred", i + 1))
            .embedding([i as f32 * 0.1; 768])
            .confidence(Confidence::MEDIUM)
            .build();
    }

    // Test 6: Episode with different confidence construction methods
    let _percent_episode = EpisodeBuilder::new()
        .id("percent_episode".to_string())
        .when(now)
        .what("Event with percentage confidence".to_string())
        .embedding([0.5f32; 768])
        .confidence(Confidence::from_percent(75))
        .build();

    let _frequency_episode = EpisodeBuilder::new()
        .id("frequency_episode".to_string())
        .when(now)
        .what("Event with frequency-based confidence".to_string())
        .embedding([0.6f32; 768])
        .confidence(Confidence::from_successes(4, 5))
        .build();

    // All episode constructions demonstrate the episodic memory
    // builder pattern working correctly with temporal information
}