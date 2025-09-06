//! Test that Memory cannot be built without confidence
//! 
//! This test validates that the typestate pattern prevents construction
//! of Memory without the required confidence value, ensuring cognitive
//! confidence is always a first-class property.

use engram_core::memory::MemoryBuilder;

fn main() {
    // This should fail: cannot build Memory without setting confidence
    let _memory = MemoryBuilder::new()
        .id("test_memory".to_string())
        .embedding([0.1f32; 768])
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Confidence is required for all memory operations
    //~| HELP: Add confidence: .confidence(Confidence::HIGH)
    //~| HELP: Confidence enables probabilistic memory operations and decay modeling
}