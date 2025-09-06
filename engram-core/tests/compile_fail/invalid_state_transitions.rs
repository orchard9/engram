//! Test that invalid state transitions are prevented at compile time
//! 
//! This test validates that the typestate pattern prevents impossible
//! state transitions and operations that would violate memory system invariants.

use engram_core::memory::{MemoryBuilder, EpisodeBuilder, CueBuilder};
use engram_core::Confidence;

fn main() {
    // Test 1: Cannot reuse builder after build() - move semantics prevent this
    let memory_builder = MemoryBuilder::new()
        .id("test".to_string())
        .embedding([0.1f32; 768])
        .confidence(Confidence::HIGH);
        
    let _memory1 = memory_builder.build();
    // This should fail - builder has been moved
    let _memory2 = memory_builder.build(); //~ ERROR: borrow of moved value: `memory_builder`
    //~| NOTE: Builder consumed by first build() call
    //~| HELP: Create separate builders for multiple memories
    //~| HELP: Move semantics prevent accidental reuse
    
    // Test 2: Cannot call methods on moved builders
    let episode_builder = EpisodeBuilder::new()
        .id("episode".to_string())
        .when(chrono::Utc::now())
        .what("Event".to_string())
        .embedding([0.2f32; 768])
        .confidence(Confidence::MEDIUM);
        
    let _episode = episode_builder.build();
    // This should fail - cannot use builder after move
    let _invalid = episode_builder.content("extra".to_string()); //~ ERROR: borrow of moved value: `episode_builder`
    //~| NOTE: Builder state transitions are one-way and consume the builder
    //~| HELP: Set all desired options before calling build()
    
    // Test 3: Cannot mix builder types inappropriately 
    let cue_builder = CueBuilder::new()
        .id("cue".to_string());
    
    // This should fail - wrong builder method on wrong type
    let _wrong = cue_builder.embedding([0.3f32; 768]); //~ ERROR: method `embedding` not found
    //~| NOTE: CueBuilder uses different methods than MemoryBuilder  
    //~| HELP: Use .embedding_search() for cues, .embedding() for memories
    //~| HELP: Each builder type has specific method signatures
}