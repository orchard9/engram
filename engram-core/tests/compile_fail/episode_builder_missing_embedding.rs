//! Test that Episode cannot be built without an embedding vector
//! 
//! This test validates that the typestate pattern prevents construction
//! of Episode without the required 768-dimensional embedding.

use engram_core::memory::EpisodeBuilder;
use engram_core::Confidence;
use chrono::Utc;

fn main() {
    // This should fail: cannot build Episode without setting embedding
    let _episode = EpisodeBuilder::new()
        .id("test_episode".to_string())
        .when(Utc::now())
        .what("Something happened".to_string())
        .confidence(Confidence::HIGH)
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Embedding vector is required for Episode similarity search
    //~| HELP: Add embedding: .embedding([0.1f32; 768])
    //~| HELP: Embeddings enable semantic retrieval of similar episodes
}