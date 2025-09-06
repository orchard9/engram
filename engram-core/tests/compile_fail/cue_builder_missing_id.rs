//! Test that Cue cannot be built without an ID
//! 
//! This test validates that the typestate pattern prevents construction
//! of Cue without the required ID field.

use engram_core::memory::CueBuilder;
use engram_core::Confidence;

fn main() {
    // This should fail: cannot build Cue without setting ID first
    let _cue = CueBuilder::new()
        .embedding_search([0.1f32; 768], Confidence::MEDIUM)
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: ID is required before building Cue
    //~| HELP: Start with: CueBuilder::new().id("unique_cue_id".to_string())
    //~| HELP: Cue ID enables tracking and caching of query results
}