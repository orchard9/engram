//! Test valid Memory construction patterns that should compile successfully
//!
//! These tests demonstrate correct usage of the Memory builder typestate pattern
//! and serve as examples for developers learning the API.

use engram_core::memory::MemoryBuilder;
use engram_core::Confidence;

fn main() {
    // Test 1: Basic valid Memory construction
    let _basic_memory = MemoryBuilder::new()
        .id("basic_memory".to_string())
        .embedding([0.1f32; 768])
        .confidence(Confidence::HIGH)
        .build();

    // Test 2: Memory with optional content
    let _content_memory = MemoryBuilder::new()
        .id("content_memory".to_string())
        .embedding([0.2f32; 768])
        .confidence(Confidence::MEDIUM)
        .content("This memory has descriptive content".to_string())
        .build();

    // Test 3: Memory with custom decay rate
    let _custom_decay = MemoryBuilder::new()
        .id("custom_decay".to_string())
        .embedding([0.3f32; 768])
        .confidence(Confidence::LOW)
        .decay_rate(0.05)
        .build();

    // Test 4: Memory with all optional parameters
    let _full_memory = MemoryBuilder::new()
        .id("full_memory".to_string())
        .embedding([0.4f32; 768])
        .confidence(Confidence::exact(0.75))
        .content("Complete memory with all options".to_string())
        .decay_rate(0.02)
        .build();

    // Test 5: Multiple memories from same pattern
    for i in 0..3 {
        let _memory = MemoryBuilder::new()
            .id(format!("memory_{}", i))
            .embedding([i as f32 * 0.1; 768])
            .confidence(Confidence::MEDIUM)
            .build();
    }

    // Test 6: Memory using different confidence construction methods
    let _percent_memory = MemoryBuilder::new()
        .id("percent_confidence".to_string())
        .embedding([0.5f32; 768])
        .confidence(Confidence::from_percent(85))
        .build();

    let _frequency_memory = MemoryBuilder::new()
        .id("frequency_confidence".to_string())
        .embedding([0.6f32; 768])
        .confidence(Confidence::from_successes(7, 10))
        .build();

    // All constructions above should compile successfully,
    // demonstrating the flexibility of the typestate pattern
    // while maintaining compile-time safety.
}