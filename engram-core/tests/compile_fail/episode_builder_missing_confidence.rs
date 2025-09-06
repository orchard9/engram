//! Test that Episode cannot be built without encoding confidence
//! 
//! This test validates that the typestate pattern prevents construction
//! of Episode without the required confidence in encoding quality.

use engram_core::memory::EpisodeBuilder;
use chrono::Utc;

fn main() {
    // This should fail: cannot build Episode without setting encoding confidence
    let _episode = EpisodeBuilder::new()
        .id("test_episode".to_string())
        .when(Utc::now())
        .what("Something happened".to_string())
        .embedding([0.1f32; 768])
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Encoding confidence is required for Episode construction
    //~| HELP: Add confidence: .confidence(Confidence::HIGH)
    //~| HELP: Confidence tracks encoding quality and memory reliability
}