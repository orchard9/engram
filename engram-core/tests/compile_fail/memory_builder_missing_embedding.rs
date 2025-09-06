//! Test that Memory cannot be built without an embedding
//! 
//! This test validates that the typestate pattern prevents construction
//! of Memory without the required 768-dimensional embedding vector.

use engram_core::memory::MemoryBuilder;
use engram_core::Confidence;

fn main() {
    // This should fail: cannot build Memory without setting embedding
    let _memory = MemoryBuilder::new()
        .id("test_memory".to_string())
        .confidence(Confidence::HIGH)
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Embedding vector is required for Memory construction
    //~| HELP: Add embedding: .embedding([0.1f32; 768])
    //~| HELP: Embeddings represent the semantic content for similarity search
}