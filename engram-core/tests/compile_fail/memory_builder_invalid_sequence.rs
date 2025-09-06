//! Test that Memory builder enforces correct method call sequence
//! 
//! This test validates that the typestate pattern prevents calling
//! methods in invalid order, teaching the natural construction sequence.

use engram_core::memory::MemoryBuilder;
use engram_core::Confidence;

fn main() {
    // This should fail: cannot set embedding before ID  
    let builder = MemoryBuilder::new();
    
    // Skip ID and try to set embedding directly - this should be impossible
    let _builder = builder.embedding([0.1f32; 768]); //~ ERROR: method `embedding` not found for this value
    //~| NOTE: ID must be set before embedding in builder pattern
    //~| HELP: Use the correct sequence: .id(...).embedding(...).confidence(...)
    //~| HELP: This sequence matches natural memory creation flow
    
    // Also test that build() is not available in intermediate states
    let partial_builder = MemoryBuilder::new()
        .id("test".to_string());
        
    // Cannot build with just ID - need embedding and confidence
    let _memory = partial_builder.build(); //~ ERROR: method `build` not found for this value
    //~| NOTE: Memory construction requires ID, embedding, and confidence
    //~| HELP: Complete the sequence: .embedding([...]).confidence(...)
}