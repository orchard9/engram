//! Test that direct construction without builders is prevented
//! 
//! This test validates that the typestate pattern encourages proper
//! construction through builders rather than direct struct instantiation.

use engram_core::memory::{Memory, Episode, Cue};
use engram_core::Confidence;

fn main() {
    // Test 1: Cannot directly construct Memory - fields are private
    let _memory = Memory {
        id: "test".to_string(),
        embedding: [0.1f32; 768],
        confidence: Confidence::HIGH,
        last_access: chrono::Utc::now(),
        created_at: chrono::Utc::now(),
        decay_rate: 0.1,
        content: None,
        activation: atomic_float::AtomicF32::new(0.0), //~ ERROR: field `activation` of struct `Memory` is private
        //~| NOTE: Memory construction requires builder pattern
        //~| HELP: Use MemoryBuilder::new().id(...).embedding(...).confidence(...).build()
        //~| HELP: Builder pattern ensures all required fields are set correctly
    };
    
    // Test 2: Cannot directly construct Episode - encourages builder usage  
    let _episode = Episode {
        id: "episode".to_string(),
        when: chrono::Utc::now(),
        where_location: None,
        who: None,
        what: "Event".to_string(),
        embedding: [0.2f32; 768],
        encoding_confidence: Confidence::HIGH,
        vividness_confidence: Confidence::HIGH,
        reliability_confidence: Confidence::HIGH,
        last_recall: chrono::Utc::now(),
        recall_count: 0, //~ ERROR: field `recall_count` of struct `Episode` is private
        //~| NOTE: Episode construction requires builder pattern
        //~| HELP: Use EpisodeBuilder::new().id(...).when(...).what(...).embedding(...).confidence(...).build()
        //~| HELP: Builder ensures proper episodic memory construction
        decay_rate: 0.05,
    };
    
    // Test 3: Cannot directly construct Cue - private fields prevent it
    let _cue = Cue {
        id: "cue".to_string(),
        cue_type: engram_core::memory::CueType::Semantic {
            content: "search".to_string(),
            fuzzy_threshold: Confidence::MEDIUM,
        },
        cue_confidence: Confidence::HIGH,
        result_threshold: Confidence::LOW, //~ ERROR: field `result_threshold` of struct `Cue` is private
        //~| NOTE: Cue construction requires builder pattern
        //~| HELP: Use CueBuilder::new().id(...).semantic_search(...).build()
        //~| HELP: Builder pattern guides proper cue configuration
        max_results: 10,
    };
}