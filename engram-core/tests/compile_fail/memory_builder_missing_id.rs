//! Test that Memory cannot be built without an ID
//! 
//! This test validates that the typestate pattern prevents construction
//! of Memory without the required ID field, teaching developers the
//! correct construction sequence.

use engram_core::memory::MemoryBuilder;
use engram_core::Confidence;

fn main() {
    // This should fail: cannot build Memory without setting ID first
    let _memory = MemoryBuilder::new()
        .embedding([0.1f32; 768])  
        .confidence(Confidence::HIGH)
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: ID is required before building Memory
    //~| HELP: Start with: MemoryBuilder::new().id("your_id".to_string())
    //~| HELP: The ID serves as unique identifier for memory retrieval
}