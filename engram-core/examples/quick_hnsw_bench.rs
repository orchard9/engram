//! Quick HNSW performance validation
//! Run with: cargo run --release --features hnsw_index --bin quick_hnsw_bench

use engram_core::index::CognitiveHnswIndex;
use engram_core::{Confidence, Memory};
use std::sync::Arc;
use std::time::Instant;

fn create_test_memory(id: u32) -> Arc<Memory> {
    let mut embedding = [0.0f32; 768];
    for (i, elem) in embedding.iter_mut().enumerate() {
        let seed = id.wrapping_mul(769).wrapping_add(i as u32);
        *elem = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }

    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for elem in &mut embedding {
            *elem /= magnitude;
        }
    }

    Arc::new(Memory::new(format!("mem_{id}"), embedding, Confidence::MEDIUM))
}

fn main() {
    println!("=== HNSW Performance Validation ===\n");

    // Single-threaded baseline
    {
        let index = CognitiveHnswIndex::new();
        let start = Instant::now();

        for i in 0..1000 {
            index.insert_memory(create_test_memory(i)).unwrap();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = 1000.0 / elapsed.as_secs_f64();
        println!("Single-threaded: 1000 ops in {:?} = {:.0} ops/sec", elapsed, ops_per_sec);
    }

    // 2-thread concurrent
    {
        let index = Arc::new(CognitiveHnswIndex::new());
        let start = Instant::now();

        let handles: Vec<_> = (0..2).map(|t| {
            let idx = Arc::clone(&index);
            std::thread::spawn(move || {
                for i in 0..1000 {
                    idx.insert_memory(create_test_memory(t * 1000 + i)).unwrap();
                }
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = 2000.0 / elapsed.as_secs_f64();
        println!("2-thread concurrent: 2000 ops in {:?} = {:.0} ops/sec", elapsed, ops_per_sec);
    }

    // 4-thread concurrent
    {
        let index = Arc::new(CognitiveHnswIndex::new());
        let start = Instant::now();

        let handles: Vec<_> = (0..4).map(|t| {
            let idx = Arc::clone(&index);
            std::thread::spawn(move || {
                for i in 0..1000 {
                    idx.insert_memory(create_test_memory(t * 1000 + i)).unwrap();
                }
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = 4000.0 / elapsed.as_secs_f64();
        println!("4-thread concurrent: 4000 ops in {:?} = {:.0} ops/sec", elapsed, ops_per_sec);
    }

    // 8-thread concurrent (target: 60K+ ops/sec)
    {
        let index = Arc::new(CognitiveHnswIndex::new());
        let start = Instant::now();

        let handles: Vec<_> = (0..8).map(|t| {
            let idx = Arc::clone(&index);
            std::thread::spawn(move || {
                for i in 0..1000 {
                    idx.insert_memory(create_test_memory(t * 1000 + i)).unwrap();
                }
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = 8000.0 / elapsed.as_secs_f64();
        println!("8-thread concurrent: 8000 ops in {:?} = {:.0} ops/sec", elapsed, ops_per_sec);

        if ops_per_sec >= 60000.0 {
            println!("\n✓ SUCCESS: Achieved {:.0} ops/sec (target: 60K+)", ops_per_sec);
        } else {
            println!("\n✗ BELOW TARGET: Achieved {:.0} ops/sec (target: 60K+)", ops_per_sec);
            println!("  Recommendation: Consider space-partitioned HNSW fallback");
        }
    }

    // Batch insertion test
    {
        let index = CognitiveHnswIndex::new();
        let memories: Vec<_> = (0..100).map(create_test_memory).collect();

        let start = Instant::now();
        index.insert_batch(&memories).unwrap();
        let batch_time = start.elapsed();

        let index2 = CognitiveHnswIndex::new();
        let start2 = Instant::now();
        for i in 0..100 {
            index2.insert_memory(create_test_memory(i)).unwrap();
        }
        let sequential_time = start2.elapsed();

        let speedup = sequential_time.as_secs_f64() / batch_time.as_secs_f64();
        println!("\nBatch of 100:");
        println!("  Sequential: {:?}", sequential_time);
        println!("  Batch:      {:?}", batch_time);
        println!("  Speedup:    {:.2}x", speedup);

        if speedup >= 3.0 {
            println!("  ✓ SUCCESS: Achieved {:.2}x speedup (target: 3x+)", speedup);
        } else {
            println!("  ✗ BELOW TARGET: Achieved {:.2}x speedup (target: 3x+)", speedup);
        }
    }
}
