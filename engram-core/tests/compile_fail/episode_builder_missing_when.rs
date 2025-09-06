//! Test that Episode cannot be built without a timestamp
//! 
//! This test validates that the typestate pattern prevents construction
//! of Episode without the required temporal information.

use engram_core::memory::EpisodeBuilder;
use engram_core::Confidence;

fn main() {
    // This should fail: cannot build Episode without setting timestamp
    let _episode = EpisodeBuilder::new()
        .id("test_episode".to_string())
        .what("Something happened".to_string())
        .embedding([0.1f32; 768])
        .confidence(Confidence::HIGH)
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Timestamp (when) is required for Episode construction
    //~| HELP: Add timestamp: .when(chrono::Utc::now())
    //~| HELP: Episodes are inherently temporal - when did this happen?
}