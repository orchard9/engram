# Usage

## Core API

```rust
use engram::{Memory, Cue, Episode, Config};

// Initialize
let mut memory = Memory::new(Config::default());

// Store episodic memory
let episode = Episode::now()
    .at_location([lat, lon])
    .with_entities(["alice", "bob"])
    .with_emotion(0.7)  // valence
    .embed(text);  // -> vector embedding

memory.store(episode);

// Recall with uncertainty
let results = memory.recall(Cue::from_embedding(vector))
    .with_confidence(0.6)  // minimum confidence
    .limit(10)
    .await;

for (episode, confidence) in results {
    println!("{}: {:.2}", episode.when(), confidence);
}

// Consolidate memories into semantic knowledge
memory.consolidate_async(Duration::hours(8));  // "sleep"

// Query semantic abstractions
let pattern = memory.generalize("restaurant visits with alice")
    .over_period(Duration::days(30));
```

## Memory Operations

### Storage
```rust
// Episodes decay automatically
memory.store(episode);  // Returns activation level

// Strengthen specific memory
memory.reinforce(episode_id, strength: 2.0);

// Explicit forgetting
memory.decay(episode_id, rate: 0.1);
```

### Retrieval
```rust
// Content-based (vector similarity)
memory.recall(Cue::from_embedding(vec));

// Context-based (spreading activation)
memory.recall(Cue::from_context()
    .location([lat, lon])
    .time_range(start..end)
    .mood(0.3));

// Pattern completion
memory.complete(partial_episode);  // Fills gaps
```

### Consolidation
```rust
// Extract semantic patterns
let schema = memory.extract_schema(
    episodes.filter(|e| e.has_entity("restaurant"))
);

// Dream consolidation (offline replay)
memory.dream(cycles: 100);  // Strengthens important, prunes weak

// Compression
memory.compress_temporal(before: Timestamp);  // Merges similar
```

### Query Language
```rust
// DSL for complex queries
let memories = memory.query(r#"
    RECALL EPISODES
    WHERE confidence > 0.7
    NEAR embedding($vector)
    DURING past_week()
    SPREADING FROM "work"
"#);

// Probabilistic operations
let prediction = memory.query(r#"
    PREDICT NEXT
    GIVEN SEQUENCE [episode_1, episode_2]
    WITH uncertainty
"#);

// Counterfactuals
let alternative = memory.query(r#"
    IMAGINE EPISODE 
    WHERE entity = "alice"
    BUT location = "paris"
    RECONSTRUCTING context
"#);
```

## Configuration

```rust
let config = Config::builder()
    .embedding_dim(768)
    .decay_function(DecayFunction::Exponential(0.05))
    .activation_threshold(0.3)
    .consolidation_period(Duration::hours(24))
    .gpu_enabled(true)
    .build();

let memory = Memory::new(config);
```

## Streaming Interface

```rust
// Real-time memory formation
let mut stream = memory.stream();

while let Some(observation) = observations.next().await {
    stream.observe(observation);  // Continuous encoding
    
    // Query while streaming
    if let Some(recall) = stream.recall_immediate(cue).await {
        process(recall);
    }
}
```

## Distributed Usage

```rust
// Partition across nodes
let memory = Memory::distributed()
    .nodes(["node1:8080", "node2:8080"])
    .partition_by(PartitionStrategy::Entity)
    .connect()
    .await?;

// Operations work identically
memory.store(episode);  // Routes to appropriate shard
memory.recall(cue);      // Gathers from all shards
```

## Cognitive Patterns

```rust
// Priming
memory.prime(concepts: ["food", "restaurant"]);
let biased_recall = memory.recall(neutral_cue);  // Biased toward primed

// Interference
let competing = memory.find_interference(episode);  // Similar memories that compete

// Reconsolidation
let mut episode = memory.recall_mut(cue);  // Makes temporarily plastic
episode.update_context(new_emotion: 0.2);
memory.reconsolidate(episode);  // Re-stores with updates
```

## Performance Control

```rust
// Control activation spreading
memory.recall(cue)
    .max_hops(3)  // Limit spreading distance
    .decay_rate(0.5)  // How fast activation decays
    .parallel(true);  // Use GPU if available

// Batch operations
memory.batch_store(episodes);  // Optimized bulk insert
memory.batch_recall(cues);      // Parallel activation

// Memory pressure
memory.forget_below_threshold(0.1);  // Prune weak memories
memory.compact();  // Defragment storage
```

## Error Handling

```rust
use engram::Error;

match memory.recall(cue).await {
    Ok(results) => process(results),
    Err(Error::LowConfidence) => {},  // No memories above threshold
    Err(Error::StorageFailure(_)) => {},  // Disk/network error
    Err(Error::Timeout) => {},  // Activation spreading timeout
}
```

## Observability

```rust
// Metrics
let metrics = memory.metrics();
println!("Active memories: {}", metrics.active_count());
println!("Avg activation: {:.3}", metrics.mean_activation());
println!("Consolidation rate: {:.1}/hour", metrics.consolidation_rate());

// Tracing activation
let trace = memory.trace_activation(cue);
for (node, activation) in trace.path() {
    println!("{}: {:.3}", node.id(), activation);
}
```

## Type Safety

```rust
// Strongly typed entities
#[derive(Entity)]
struct Person { name: String }

#[derive(Entity)]  
struct Location { lat: f64, lon: f64 }

let episode = Episode::now()
    .with_entity(Person { name: "alice".into() })
    .with_entity(Location { lat: 37.7, lon: -122.4 });

// Compile-time query checking
let people: Vec<(Person, f32)> = memory
    .recall_type::<Person>(cue)
    .await?;
```

## Minimal Example

```rust
use engram::Memory;

fn main() -> engram::Result<()> {
    let mut memory = Memory::new_default();
    
    // Store
    let id = memory.store_text("Met Alice at conference")?;
    
    // Recall
    let similar = memory.recall_text("Alice meeting")?;
    
    // Consolidate
    memory.sleep()?;
    
    Ok(())
}
```
