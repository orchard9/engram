//! Test that Episode builder provides educational error messages
//! 
//! This test validates that compilation failures provide clear guidance
//! about the required episodic memory construction sequence.

use engram_core::memory::EpisodeBuilder;

fn main() {
    // Test 1: Attempting to build immediately fails with educational guidance
    let _episode = EpisodeBuilder::new().build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Episode requires sequential construction following cognitive memory patterns
    //~| HELP: Episode construction sequence: ID → when → what → embedding → confidence → build
    //~| HELP: This matches how humans naturally encode episodic memories
    
    // Test 2: Trying to skip to later steps without completing earlier ones  
    let builder = EpisodeBuilder::new();
    let _fail = builder.embedding([0.1f32; 768]); //~ ERROR: method `embedding` not found for this value
    //~| NOTE: Episode builder enforces natural construction order
    //~| HELP: ID, timestamp, and content must be set before embedding
    //~| HELP: Typestate prevents incomplete episodic memories
    
    // Test 3: Attempting to set optional fields before required ones
    let partial = EpisodeBuilder::new()
        .id("test".to_string());
    let _invalid = partial.where_location("somewhere".to_string()); //~ ERROR: method `where_location` not found for this value
    //~| NOTE: Optional episodic details only available after required fields
    //~| HELP: Complete core sequence: .when(...).what(...).embedding(...).confidence(...)
    //~| HELP: Location, participants, and other details are optional enhancements
}