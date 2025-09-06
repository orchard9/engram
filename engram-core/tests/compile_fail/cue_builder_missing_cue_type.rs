//! Test that Cue cannot be built without a cue type
//! 
//! This test validates that the typestate pattern prevents construction
//! of Cue without specifying how the search should be performed.

use engram_core::memory::CueBuilder;

fn main() {
    // This should fail: cannot build Cue without setting search type
    let _cue = CueBuilder::new()
        .id("test_cue".to_string())
        .build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Cue type is required - how should memory search be performed?
    //~| HELP: Choose search method: .embedding_search(...) or .context_search(...) or .semantic_search(...)
    //~| HELP: Different cue types enable different retrieval patterns
}