//! Test that Episode cannot be built without an ID
//! 
//! This test validates that the typestate pattern prevents construction
//! of Episode without the required ID field.

use engram_core::memory::EpisodeBuilder;
use engram_core::Confidence;
use chrono::Utc;

fn main() {
    // This should fail: cannot build Episode without setting ID first
    let _episode = EpisodeBuilder::new()
        .when(Utc::now())
        .what("Something happened".to_string())
        .embedding([0.1f32; 768])
        .confidence(Confidence::HIGH)
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: ID is required before building Episode
    //~| HELP: Start with: EpisodeBuilder::new().id("unique_id".to_string())
    //~| HELP: Episode ID enables temporal and contextual retrieval
}