//! Test complex but valid typestate patterns that should compile successfully
//!
//! These tests demonstrate advanced usage patterns that combine multiple
//! builders and show realistic usage scenarios.

use engram_core::memory::{MemoryBuilder, EpisodeBuilder, CueBuilder, TemporalPattern};
use engram_core::Confidence;
use chrono::{Utc, Duration};

fn main() {
    let now = Utc::now();
    
    // Test 1: Create memory, episode, and cue as a related set
    let memory_id = "knowledge_base_entry_1".to_string();
    let episode_id = "learning_session_1".to_string();
    let cue_id = "search_for_entry_1".to_string();
    
    let _memory = MemoryBuilder::new()
        .id(memory_id.clone())
        .embedding([0.1f32; 768])
        .confidence(Confidence::HIGH)
        .content("Rust typestate patterns for memory safety".to_string())
        .decay_rate(0.01) // Long-term knowledge
        .build();
        
    let _episode = EpisodeBuilder::new()
        .id(episode_id.clone())
        .when(now)
        .what("Learned about typestate patterns in Rust".to_string())
        .embedding([0.1f32; 768]) // Related embedding
        .confidence(Confidence::HIGH)
        .where_location("Home Office".to_string())
        .decay_rate(0.05) // Episodic memories decay faster
        .build();
        
    let _cue = CueBuilder::new()
        .id(cue_id)
        .semantic_search("typestate patterns Rust".to_string(), Confidence::MEDIUM)
        .result_threshold(Confidence::LOW) // Cast wide net
        .max_results(10)
        .build();

    // Test 2: Batch processing pattern with error handling
    let topics = vec![
        ("machine_learning", "ML fundamentals"),
        ("data_structures", "Advanced trees and graphs"),
        ("algorithms", "Dynamic programming techniques"),
    ];
    
    for (topic, description) in topics {
        let memory_id = format!("topic_{}", topic);
        let episode_id = format!("study_{}", topic);
        
        // Create knowledge memory
        let _knowledge = MemoryBuilder::new()
            .id(memory_id)
            .embedding([0.2f32; 768])
            .confidence(Confidence::MEDIUM)
            .content(description.to_string())
            .build();
            
        // Create learning episode
        let _learning = EpisodeBuilder::new()
            .id(episode_id)
            .when(now - Duration::hours(2))
            .what(format!("Studied {}", description))
            .embedding([0.2f32; 768])
            .confidence(Confidence::MEDIUM)
            .build();
    }

    // Test 3: Different temporal search patterns
    let temporal_patterns = vec![
        ("recent_24h", TemporalPattern::Recent(Duration::hours(24))),
        ("before_yesterday", TemporalPattern::Before(now - Duration::days(1))),
        ("after_last_week", TemporalPattern::After(now - Duration::days(7))),
        ("this_week", TemporalPattern::Between(
            now - Duration::days(7), 
            now
        )),
    ];
    
    for (pattern_name, pattern) in temporal_patterns {
        let _temporal_cue = CueBuilder::new()
            .id(format!("temporal_{}", pattern_name))
            .temporal_search(pattern, Confidence::MEDIUM)
            .build();
    }

    // Test 4: Confidence calibration patterns
    let confidence_levels = vec![
        ("high_precision", Confidence::exact(0.9), Confidence::HIGH),
        ("medium_precision", Confidence::exact(0.7), Confidence::MEDIUM),
        ("low_precision", Confidence::exact(0.3), Confidence::LOW),
    ];
    
    for (precision_type, memory_conf, cue_threshold) in confidence_levels {
        let _calibrated_memory = MemoryBuilder::new()
            .id(format!("calibrated_{}", precision_type))
            .embedding([0.3f32; 768])
            .confidence(memory_conf)
            .build();
            
        let _precision_cue = CueBuilder::new()
            .id(format!("precision_{}", precision_type))
            .embedding_search([0.3f32; 768], cue_threshold)
            .cue_confidence(memory_conf)
            .result_threshold(cue_threshold)
            .build();
    }

    // Test 5: Nested function pattern (builders as parameters)
    fn create_memory_episode_pair(
        base_id: &str,
        content: &str,
        confidence: Confidence
    ) -> (engram_core::memory::Memory, engram_core::memory::Episode) {
        let memory = MemoryBuilder::new()
            .id(format!("mem_{}", base_id))
            .embedding([0.4f32; 768])
            .confidence(confidence)
            .content(content.to_string())
            .build();
            
        let episode = EpisodeBuilder::new()
            .id(format!("epi_{}", base_id))
            .when(now)
            .what(format!("Encoded: {}", content))
            .embedding([0.4f32; 768])
            .confidence(confidence)
            .build();
            
        (memory, episode)
    }
    
    let (_mem1, _epi1) = create_memory_episode_pair(
        "pair1", 
        "First memory-episode pair", 
        Confidence::HIGH
    );
    let (_mem2, _epi2) = create_memory_episode_pair(
        "pair2", 
        "Second memory-episode pair", 
        Confidence::MEDIUM
    );

    // All complex patterns demonstrate that the typestate system
    // scales to realistic usage scenarios while maintaining safety
}