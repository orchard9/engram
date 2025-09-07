//! Test integration scenarios that should compile successfully
//!
//! These tests demonstrate realistic integration patterns where typestate
//! builders are used within larger application contexts and workflows.

use engram_core::memory::{MemoryBuilder, EpisodeBuilder, CueBuilder, TemporalPattern};
use engram_core::Confidence;
use chrono::{Utc, Duration};

fn main() {
    let now = Utc::now();
    
    // Test 1: Knowledge Management System Integration
    struct KnowledgeManager;
    
    impl KnowledgeManager {
        fn store_learning_session(
            &self,
            topic: &str,
            content: &str,
            learning_quality: f32
        ) -> (engram_core::memory::Memory, engram_core::memory::Episode) {
            let confidence = Confidence::exact(learning_quality);
            let topic_embedding = self.generate_topic_embedding(topic);
            
            let memory = MemoryBuilder::new()
                .id(format!("knowledge_{}", topic))
                .embedding(topic_embedding)
                .confidence(confidence)
                .content(content.to_string())
                .decay_rate(0.02) // Knowledge decays slowly
                .build();
                
            let episode = EpisodeBuilder::new()
                .id(format!("learning_{}", topic))
                .when(Utc::now())
                .what(format!("Learned about {}", topic))
                .embedding(topic_embedding)
                .confidence(confidence)
                .where_location("Study Space".to_string())
                .build();
                
            (memory, episode)
        }
        
        fn create_search_cue(&self, query: &str) -> engram_core::memory::Cue {
            CueBuilder::new()
                .id(format!("search_{}", query.replace(' ', "_")))
                .semantic_search(query.to_string(), Confidence::MEDIUM)
                .result_threshold(Confidence::LOW)
                .max_results(20)
                .build()
        }
        
        fn generate_topic_embedding(&self, _topic: &str) -> [f32; 768] {
            // Simplified embedding generation
            [0.1f32; 768]
        }
    }
    
    let km = KnowledgeManager;
    let (_rust_memory, _rust_episode) = km.store_learning_session(
        "rust_ownership",
        "Ownership prevents memory safety issues",
        0.85
    );
    
    let _search_cue = km.create_search_cue("rust memory safety");

    // Test 2: Memory Consolidation Workflow
    fn consolidation_workflow() -> Vec<engram_core::memory::Memory> {
        let base_memories = vec![
            ("concept_a", "Basic concept understanding", 0.6),
            ("concept_b", "Intermediate application", 0.7),
            ("concept_c", "Advanced synthesis", 0.8),
        ];
        
        base_memories
            .into_iter()
            .map(|(id, content, confidence)| {
                MemoryBuilder::new()
                    .id(format!("consolidated_{}", id))
                    .embedding([confidence; 768])
                    .confidence(Confidence::exact(confidence))
                    .content(content.to_string())
                    .decay_rate(0.01) // Consolidated memories are stable
                    .build()
            })
            .collect()
    }
    
    let _consolidated_memories = consolidation_workflow();

    // Test 3: Temporal Query Builder
    struct TemporalQueryBuilder {
        base_id: String,
    }
    
    impl TemporalQueryBuilder {
        fn new(base_id: String) -> Self {
            Self { base_id }
        }
        
        fn recent_memories(&self, hours: i64) -> engram_core::memory::Cue {
            CueBuilder::new()
                .id(format!("{}_recent_{}h", self.base_id, hours))
                .temporal_search(
                    TemporalPattern::Recent(Duration::hours(hours)),
                    Confidence::MEDIUM
                )
                .max_results(100)
                .build()
        }
        
        fn memories_before(&self, cutoff: chrono::DateTime<Utc>) -> engram_core::memory::Cue {
            CueBuilder::new()
                .id(format!("{}_before", self.base_id))
                .temporal_search(
                    TemporalPattern::Before(cutoff),
                    Confidence::HIGH
                )
                .build()
        }
        
        fn memories_in_range(
            &self, 
            start: chrono::DateTime<Utc>, 
            end: chrono::DateTime<Utc>
        ) -> engram_core::memory::Cue {
            CueBuilder::new()
                .id(format!("{}_range", self.base_id))
                .temporal_search(
                    TemporalPattern::Between(start, end),
                    Confidence::MEDIUM
                )
                .result_threshold(Confidence::LOW)
                .build()
        }
    }
    
    let temporal_builder = TemporalQueryBuilder::new("session_1".to_string());
    let _recent = temporal_builder.recent_memories(24);
    let _before = temporal_builder.memories_before(now - Duration::days(7));
    let _range = temporal_builder.memories_in_range(
        now - Duration::days(30), 
        now
    );

    // Test 4: Batch Memory Processing
    fn process_memory_batch(entries: Vec<(String, String, f32)>) -> Vec<engram_core::memory::Memory> {
        entries
            .into_iter()
            .enumerate()
            .map(|(idx, (id, content, conf))| {
                MemoryBuilder::new()
                    .id(format!("batch_{}_{}", idx, id))
                    .embedding([conf; 768])
                    .confidence(Confidence::exact(conf.clamp(0.0, 1.0)))
                    .content(content)
                    .decay_rate(0.1)
                    .build()
            })
            .collect()
    }
    
    let batch_data = vec![
        ("item1".to_string(), "First batch item".to_string(), 0.7),
        ("item2".to_string(), "Second batch item".to_string(), 0.8),
        ("item3".to_string(), "Third batch item".to_string(), 0.6),
    ];
    
    let _batch_memories = process_memory_batch(batch_data);

    // Test 5: Error Recovery Pattern
    fn safe_memory_creation(
        id: Option<String>,
        embedding: Option<[f32; 768]>,
        confidence: Option<Confidence>
    ) -> Option<engram_core::memory::Memory> {
        let id = id?;
        let embedding = embedding?;  
        let confidence = confidence?;
        
        Some(
            MemoryBuilder::new()
                .id(id)
                .embedding(embedding)
                .confidence(confidence)
                .build()
        )
    }
    
    // Valid creation
    let _valid = safe_memory_creation(
        Some("safe_memory".to_string()),
        Some([0.5f32; 768]),
        Some(Confidence::MEDIUM)
    );
    
    // Creation with missing data returns None (safe failure)
    let _invalid = safe_memory_creation(
        None, // Missing ID
        Some([0.5f32; 768]),
        Some(Confidence::MEDIUM)
    );

    // Test 6: Builder Chain Composition
    fn compose_memory_with_episode(
        base_id: &str,
        content: &str,
        when: chrono::DateTime<Utc>
    ) -> (engram_core::memory::Memory, engram_core::memory::Episode) {
        let shared_embedding = [0.7f32; 768];
        let shared_confidence = Confidence::HIGH;
        
        let memory = MemoryBuilder::new()
            .id(format!("mem_{}", base_id))
            .embedding(shared_embedding)
            .confidence(shared_confidence)
            .content(content.to_string())
            .build();
            
        let episode = EpisodeBuilder::new()
            .id(format!("epi_{}", base_id))
            .when(when)
            .what(format!("Created memory: {}", content))
            .embedding(shared_embedding)
            .confidence(shared_confidence)
            .build();
            
        (memory, episode)
    }
    
    let (_composed_mem, _composed_epi) = compose_memory_with_episode(
        "composition_test",
        "This tests builder composition",
        now
    );

    // All integration scenarios demonstrate that the typestate pattern
    // integrates well with realistic application architectures and workflows
}