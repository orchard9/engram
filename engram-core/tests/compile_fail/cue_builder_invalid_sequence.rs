//! Test that Cue builder enforces correct method call sequence
//! 
//! This test validates that the typestate pattern prevents calling
//! methods in invalid order for cue construction.

use engram_core::memory::CueBuilder;
use engram_core::Confidence;

fn main() {
    // This should fail: cannot set search type before ID
    let builder = CueBuilder::new();
    
    // Try to set embedding search without ID first - this should be impossible
    let _builder = builder.embedding_search([0.1f32; 768], Confidence::MEDIUM); //~ ERROR: method `embedding_search` not found for this value
    //~| NOTE: ID must be set before cue type in builder pattern  
    //~| HELP: Use correct sequence: .id(...).embedding_search(...)
    //~| HELP: ID establishes cue identity before search parameters
    
    // Also test that build() is not available without cue type
    let partial_builder = CueBuilder::new()
        .id("test".to_string());
        
    // Cannot build with just ID - need cue type
    let _cue = partial_builder.build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Cue construction requires both ID and search type
    //~| HELP: Add search method: .embedding_search(...) or other cue type
}