//! Test that Cue builder provides educational error messages
//! 
//! This test validates that compilation failures provide clear guidance
//! about proper cue construction for memory retrieval.

use engram_core::memory::CueBuilder;

fn main() {
    // Test 1: Attempting to build immediately fails with guidance
    let _cue = CueBuilder::new().build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Cue requires ID and search type specification
    //~| HELP: Sequence: CueBuilder::new().id(...).embedding_search(...) or other search type
    //~| HELP: Cues define how to search memory - semantic, temporal, or contextual
    
    // Test 2: Trying to use advanced options before basics
    let builder = CueBuilder::new();
    let _fail = builder.max_results(10); //~ ERROR: method `max_results` not found for this value
    //~| NOTE: Basic cue parameters must be set before optional configurations
    //~| HELP: Set ID and cue type first: .id(...).embedding_search(...)
    //~| HELP: Optional parameters available only in Ready state
    
    // Test 3: Attempting invalid cue construction patterns
    let partial = CueBuilder::new()
        .id("test".to_string());
    let _invalid = partial.cue_confidence(engram_core::Confidence::HIGH); //~ ERROR: method `cue_confidence` not found for this value
    //~| NOTE: Cue confidence tuning available only after search type is set
    //~| HELP: Complete required sequence: .embedding_search(...) or other cue type
    //~| HELP: Confidence tuning helps optimize retrieval precision vs recall
}