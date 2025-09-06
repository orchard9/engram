//! Test that Memory builder provides educational error messages
//! 
//! This test validates that compilation failures provide clear guidance
//! about correct usage patterns and the reasoning behind typestate constraints.

use engram_core::memory::MemoryBuilder;

fn main() {
    // Test 1: Attempting to build immediately fails with guidance
    let _memory = MemoryBuilder::new().build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Memory requires sequential construction: ID → embedding → confidence → build
    //~| HELP: Start with: MemoryBuilder::new().id("unique_id".to_string())
    //~| HELP: Memory construction follows cognitive-friendly builder pattern
    
    // Test 2: Trying to skip required steps
    let builder = MemoryBuilder::new();
    let _fail = builder.confidence(engram_core::Confidence::HIGH); //~ ERROR: method `confidence` not found for this value
    //~| NOTE: Builder state transitions prevent skipping required fields
    //~| HELP: ID and embedding must be set before confidence
    //~| HELP: Typestate pattern ensures no incomplete memories are created
    
    // Test 3: Attempting invalid operations on incomplete builders
    let partial = MemoryBuilder::new()
        .id("test".to_string());
    let _invalid = partial.content("content".to_string()); //~ ERROR: method `content` not found for this value  
    //~| NOTE: Content is optional and only available after required fields
    //~| HELP: Complete required sequence first: .embedding(...).confidence(...)
    //~| HELP: Optional methods available only in Ready state
}