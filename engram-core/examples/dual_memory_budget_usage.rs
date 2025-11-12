//! Example: Using DualMemoryBudget for episode-concept storage
//!
//! Demonstrates lock-free memory budget coordination for dual-tier storage.

use engram_core::storage::DualMemoryBudget;
use std::sync::Arc;
use std::thread;

const NODE_SIZE: usize = 3328; // Approximate DualMemoryNode size

fn main() {
    println!("DualMemoryBudget Example\n");
    println!("========================\n");

    // Create budget: 512MB for episodes (high churn), 1GB for concepts (stable)
    let budget = Arc::new(DualMemoryBudget::new(512, 1024));

    println!("Initial state:");
    println!("  Episode capacity: {} nodes", budget.episode_capacity());
    println!("  Concept capacity: {} nodes", budget.concept_capacity());
    println!(
        "  Episode utilization: {:.2}%",
        budget.episode_utilization()
    );
    println!(
        "  Concept utilization: {:.2}%\n",
        budget.concept_utilization()
    );

    // Simulate concurrent episode allocation (8 threads)
    println!("Simulating concurrent episode allocations (8 threads x 1000 ops)...");
    let mut handles = vec![];
    for _ in 0..8 {
        let budget = Arc::clone(&budget);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                if budget.can_allocate_episode() {
                    budget.record_episode_allocation(NODE_SIZE);
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("After episode allocations:");
    println!(
        "  Episode utilization: {:.2}%",
        budget.episode_utilization()
    );
    println!("  Allocated bytes: {}", budget.episode_allocated_bytes());
    println!(
        "  Concept utilization: {:.2}% (independent)\n",
        budget.concept_utilization()
    );

    // Simulate concept formation (slower, less frequent)
    println!("Simulating concept formation (4 threads x 500 ops)...");
    let mut handles = vec![];
    for _ in 0..4 {
        let budget = Arc::clone(&budget);
        let handle = thread::spawn(move || {
            for _ in 0..500 {
                if budget.can_allocate_concept() {
                    budget.record_concept_allocation(NODE_SIZE);
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("After concept allocations:");
    println!(
        "  Episode utilization: {:.2}%",
        budget.episode_utilization()
    );
    println!(
        "  Concept utilization: {:.2}%\n",
        budget.concept_utilization()
    );

    // Simulate eviction (deallocating old episodes)
    println!("Simulating episode eviction (50% of episodes)...");
    let to_evict = budget.episode_allocated_bytes() / 2;
    budget.record_episode_deallocation(to_evict);

    println!("After eviction:");
    println!(
        "  Episode utilization: {:.2}%",
        budget.episode_utilization()
    );
    println!(
        "  Concept utilization: {:.2}% (unaffected)\n",
        budget.concept_utilization()
    );

    println!("Summary:");
    println!(
        "  Total episode nodes: ~{}",
        budget.episode_allocated_bytes() / NODE_SIZE
    );
    println!(
        "  Total concept nodes: ~{}",
        budget.concept_allocated_bytes() / NODE_SIZE
    );
    println!(
        "  Episode capacity used: {:.1}%",
        budget.episode_utilization()
    );
    println!(
        "  Concept capacity used: {:.1}%",
        budget.concept_utilization()
    );
    println!("\nDemonstrated:");
    println!("  ✓ Lock-free concurrent allocation");
    println!("  ✓ Independent episode/concept budgets");
    println!("  ✓ Atomic utilization tracking");
    println!("  ✓ Saturating overflow/underflow protection");
}
