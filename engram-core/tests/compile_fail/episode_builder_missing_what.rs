//! Test that Episode cannot be built without content description
//! 
//! This test validates that the typestate pattern prevents construction
//! of Episode without the required "what happened" description.

use engram_core::memory::EpisodeBuilder;
use engram_core::Confidence;
use chrono::Utc;

fn main() {
    // This should fail: cannot build Episode without describing what happened
    let _episode = EpisodeBuilder::new()
        .id("test_episode".to_string())
        .when(Utc::now())
        .embedding([0.1f32; 768])
        .confidence(Confidence::HIGH)
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Content description (what) is required for Episode
    //~| HELP: Add description: .what("what happened".to_string())
    //~| HELP: Episodes need semantic content describing the event
}