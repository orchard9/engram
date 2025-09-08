# Task 006: Storage Tier Migration Policies

## Status: Pending
## Priority: P1 - System Critical
## Estimated Effort: 3 days
## Dependencies: Task 001 (three-tier storage)

## Objective
Implement intelligent migration policies between storage tiers based on activation frequency, access patterns, and memory pressure with probabilistic confidence tracking.

## Current State Analysis
- **Existing**: Basic activation tracking from milestone-1
- **Existing**: Decay functions for temporal patterns
- **Missing**: Migration decision engine
- **Missing**: Batch migration coordination
- **Missing**: Memory pressure response

## Technical Specification

### 1. Migration Decision Engine

```rust
// engram-core/src/storage/migration_engine.rs

use std::collections::BinaryHeap;

pub struct MigrationEngine {
    /// Policy configuration
    policy: MigrationPolicy,
    
    /// Access pattern tracker
    access_tracker: AccessPatternTracker,
    
    /// Memory pressure monitor
    memory_monitor: MemoryPressureMonitor,
    
    /// Migration queue
    migration_queue: BinaryHeap<MigrationTask>,
    
    /// Migration executor
    executor: MigrationExecutor,
}

#[derive(Debug, Clone)]
pub struct MigrationPolicy {
    /// Thresholds for tier transitions
    hot_threshold: AccessThreshold,
    warm_threshold: AccessThreshold,
    cold_threshold: AccessThreshold,
    
    /// Memory limits per tier
    hot_memory_limit: usize,
    warm_memory_limit: usize,
    
    /// Batch settings
    batch_size: usize,
    max_concurrent_migrations: usize,
    
    /// Confidence requirements
    min_confidence_for_promotion: Confidence,
    decay_factor_for_demotion: f32,
}

#[derive(Debug, Clone)]
struct AccessThreshold {
    /// Minimum activation level
    activation_threshold: f32,
    
    /// Access frequency (accesses per hour)
    frequency_threshold: f32,
    
    /// Time since last access
    idle_duration: Duration,
    
    /// Pattern consistency requirement
    pattern_stability: f32,
}

impl MigrationEngine {
    /// Evaluate vectors for migration
    pub fn evaluate_migrations(&mut self) -> Vec<MigrationTask> {
        let mut tasks = Vec::new();
        
        // Check hot tier for demotion
        let hot_candidates = self.evaluate_hot_tier();
        tasks.extend(hot_candidates);
        
        // Check warm tier for promotion/demotion
        let warm_candidates = self.evaluate_warm_tier();
        tasks.extend(warm_candidates);
        
        // Check cold tier for promotion
        let cold_candidates = self.evaluate_cold_tier();
        tasks.extend(cold_candidates);
        
        // Handle memory pressure
        if self.memory_monitor.is_under_pressure() {
            let pressure_migrations = self.handle_memory_pressure();
            tasks.extend(pressure_migrations);
        }
        
        // Sort by priority and confidence
        tasks.sort_by(|a, b| {
            b.priority.partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        tasks.truncate(self.policy.batch_size);
        tasks
    }
    
    fn evaluate_hot_tier(&self) -> Vec<MigrationTask> {
        let mut tasks = Vec::new();
        
        for (id, stats) in self.access_tracker.hot_tier_stats() {
            let decision = self.decide_hot_migration(&stats);
            
            if let MigrationDecision::Demote(confidence) = decision {
                tasks.push(MigrationTask {
                    vector_id: id.clone(),
                    source_tier: StorageTier::Hot,
                    target_tier: StorageTier::Warm,
                    priority: self.calculate_priority(&stats, confidence),
                    confidence,
                    reason: MigrationReason::LowActivation,
                });
            }
        }
        
        tasks
    }
    
    fn decide_hot_migration(&self, stats: &AccessStats) -> MigrationDecision {
        let threshold = &self.policy.hot_threshold;
        
        // Check multiple criteria
        let activation_score = stats.current_activation / threshold.activation_threshold;
        let frequency_score = stats.access_frequency / threshold.frequency_threshold;
        let idle_score = threshold.idle_duration.as_secs() as f32 / 
                        stats.idle_time.as_secs() as f32;
        
        // Weighted decision with confidence
        let combined_score = activation_score * 0.4 + 
                           frequency_score * 0.4 + 
                           idle_score * 0.2;
        
        if combined_score < 0.5 {
            let confidence = Confidence::from_score(1.0 - combined_score);
            MigrationDecision::Demote(confidence)
        } else {
            MigrationDecision::Stay
        }
    }
}
```

### 2. Access Pattern Tracking

```rust
// engram-core/src/storage/access_tracking.rs

use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};

pub struct AccessPatternTracker {
    /// Per-vector access statistics
    vector_stats: DashMap<String, AccessStats>,
    
    /// Sliding window for pattern detection
    access_history: CircularBuffer<AccessEvent>,
    
    /// Pattern classifier
    pattern_classifier: PatternClassifier,
}

#[derive(Debug, Clone)]
pub struct AccessStats {
    /// Current activation level
    pub current_activation: f32,
    
    /// Exponentially weighted access frequency
    pub access_frequency: f32,
    
    /// Time since last access
    pub idle_time: Duration,
    
    /// Total access count
    pub total_accesses: u64,
    
    /// Access pattern type
    pub pattern: AccessPattern,
    
    /// Confidence in pattern classification
    pub pattern_confidence: Confidence,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    /// Frequently accessed, high activation
    Hot,
    /// Periodic access with predictable pattern
    Periodic { period: Duration },
    /// Burst access followed by idle
    Bursty { burst_size: usize },
    /// Declining access over time
    Decaying { half_life: Duration },
    /// Rarely accessed
    Cold,
}

impl AccessPatternTracker {
    /// Record vector access
    pub fn record_access(&mut self, vector_id: &str, activation: f32) {
        let now = SystemTime::now();
        
        let event = AccessEvent {
            vector_id: vector_id.to_string(),
            timestamp: now,
            activation,
        };
        
        // Update statistics
        self.vector_stats
            .entry(vector_id.to_string())
            .and_modify(|stats| {
                stats.update_access(now, activation);
            })
            .or_insert_with(|| AccessStats::new(activation));
        
        // Add to history
        self.access_history.push(event);
        
        // Reclassify pattern if enough data
        if self.should_reclassify(vector_id) {
            self.reclassify_pattern(vector_id);
        }
    }
    
    fn reclassify_pattern(&mut self, vector_id: &str) {
        if let Some(mut stats) = self.vector_stats.get_mut(vector_id) {
            let history = self.get_vector_history(vector_id);
            
            let (pattern, confidence) = self.pattern_classifier.classify(&history);
            
            stats.pattern = pattern;
            stats.pattern_confidence = confidence;
        }
    }
}

pub struct PatternClassifier {
    /// Periodicity detector using FFT
    periodicity_detector: PeriodicityDetector,
    
    /// Burst detector using statistical methods
    burst_detector: BurstDetector,
    
    /// Decay curve fitter
    decay_fitter: DecayFitter,
}

impl PatternClassifier {
    pub fn classify(&self, history: &[AccessEvent]) -> (AccessPattern, Confidence) {
        if history.len() < 10 {
            return (AccessPattern::Cold, Confidence::LOW);
        }
        
        // Check for periodicity
        if let Some((period, confidence)) = self.periodicity_detector.detect(history) {
            if confidence > Confidence::MEDIUM {
                return (AccessPattern::Periodic { period }, confidence);
            }
        }
        
        // Check for bursts
        if let Some((burst_size, confidence)) = self.burst_detector.detect(history) {
            if confidence > Confidence::MEDIUM {
                return (AccessPattern::Bursty { burst_size }, confidence);
            }
        }
        
        // Check for decay pattern
        if let Some((half_life, confidence)) = self.decay_fitter.fit(history) {
            if confidence > Confidence::MEDIUM {
                return (AccessPattern::Decaying { half_life }, confidence);
            }
        }
        
        // Classify by frequency
        let frequency = history.len() as f32 / 
                       history.last().unwrap().timestamp
                           .duration_since(history.first().unwrap().timestamp)
                           .unwrap()
                           .as_secs() as f32;
        
        if frequency > 1.0 {
            (AccessPattern::Hot, Confidence::HIGH)
        } else {
            (AccessPattern::Cold, Confidence::HIGH)
        }
    }
}
```

### 3. Memory Pressure Response

```rust
// engram-core/src/storage/memory_pressure.rs

use sysinfo::{System, SystemExt};

pub struct MemoryPressureMonitor {
    /// System information
    system: System,
    
    /// Pressure thresholds
    warning_threshold: f32,  // e.g., 80% memory usage
    critical_threshold: f32, // e.g., 95% memory usage
    
    /// Current pressure level
    pressure_level: AtomicU8,
    
    /// Response strategy
    response_strategy: PressureResponseStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum PressureLevel {
    Normal = 0,
    Warning = 1,
    Critical = 2,
    Emergency = 3,
}

pub enum PressureResponseStrategy {
    /// Aggressive demotion to lower tiers
    AggressiveDemotion,
    
    /// Selective eviction of low-value items
    SelectiveEviction,
    
    /// Compression of warm/cold tiers
    CompressionFirst,
    
    /// Hybrid approach
    Adaptive,
}

impl MemoryPressureMonitor {
    pub fn is_under_pressure(&self) -> bool {
        self.pressure_level.load(Ordering::Relaxed) > PressureLevel::Normal as u8
    }
    
    pub fn get_pressure_level(&mut self) -> PressureLevel {
        self.system.refresh_memory();
        
        let used_memory = self.system.used_memory();
        let total_memory = self.system.total_memory();
        let usage_ratio = used_memory as f32 / total_memory as f32;
        
        let level = if usage_ratio > self.critical_threshold {
            PressureLevel::Critical
        } else if usage_ratio > self.warning_threshold {
            PressureLevel::Warning
        } else {
            PressureLevel::Normal
        };
        
        self.pressure_level.store(level as u8, Ordering::Relaxed);
        level
    }
    
    pub fn recommend_migrations(&mut self) -> Vec<MigrationTask> {
        let pressure_level = self.get_pressure_level();
        
        match pressure_level {
            PressureLevel::Normal => Vec::new(),
            
            PressureLevel::Warning => {
                // Moderate demotion of least active hot tier items
                self.demote_least_active(100)
            }
            
            PressureLevel::Critical => {
                // Aggressive demotion and compression
                let mut tasks = self.demote_least_active(500);
                tasks.extend(self.compress_warm_tier());
                tasks
            }
            
            PressureLevel::Emergency => {
                // Emergency eviction of cold items
                let mut tasks = self.demote_least_active(1000);
                tasks.extend(self.evict_cold_items(500));
                tasks
            }
        }
    }
}
```

### 4. Batch Migration Coordinator

```rust
// engram-core/src/storage/migration_coordinator.rs

use tokio::sync::Semaphore;

pub struct MigrationCoordinator {
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
    
    /// Active migrations
    active_migrations: DashMap<String, MigrationStatus>,
    
    /// Migration metrics
    metrics: MigrationMetrics,
}

#[derive(Debug, Clone)]
pub struct MigrationTask {
    pub vector_id: String,
    pub source_tier: StorageTier,
    pub target_tier: StorageTier,
    pub priority: f32,
    pub confidence: Confidence,
    pub reason: MigrationReason,
}

impl MigrationCoordinator {
    /// Execute migration batch
    pub async fn execute_batch(
        &self,
        tasks: Vec<MigrationTask>,
        storage: &TieredStorage,
    ) -> MigrationResults {
        let mut handles = Vec::new();
        
        for task in tasks {
            let permit = self.semaphore.clone().acquire_owned().await.unwrap();
            let storage = storage.clone();
            let coordinator = self.clone();
            
            let handle = tokio::spawn(async move {
                let result = coordinator.execute_single(task, &storage).await;
                drop(permit);
                result
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut results = MigrationResults::default();
        for handle in handles {
            match handle.await {
                Ok(Ok(())) => results.successful += 1,
                Ok(Err(e)) => {
                    results.failed += 1;
                    results.errors.push(e);
                }
                Err(e) => {
                    results.failed += 1;
                    results.errors.push(e.into());
                }
            }
        }
        
        results
    }
    
    async fn execute_single(
        &self,
        task: MigrationTask,
        storage: &TieredStorage,
    ) -> Result<()> {
        // Update status
        self.active_migrations.insert(
            task.vector_id.clone(),
            MigrationStatus::InProgress,
        );
        
        // Perform migration
        let start = Instant::now();
        
        let data = storage.retrieve_from_tier(&task.vector_id, task.source_tier).await?;
        storage.store_to_tier(&task.vector_id, data, task.target_tier).await?;
        storage.remove_from_tier(&task.vector_id, task.source_tier).await?;
        
        let duration = start.elapsed();
        
        // Update metrics
        self.metrics.record_migration(task.source_tier, task.target_tier, duration);
        
        // Update status
        self.active_migrations.insert(
            task.vector_id,
            MigrationStatus::Completed,
        );
        
        Ok(())
    }
}
```

## Integration Points

### Modify TieredStorage (from Task 001)
```rust
// Add around line 100:
impl TieredStorage {
    pub fn set_migration_policy(&mut self, policy: MigrationPolicy) {
        self.migration_policy = policy;
        self.migration_engine = MigrationEngine::new(policy);
    }
    
    pub async fn run_migration_cycle(&mut self) -> MigrationResults {
        let tasks = self.migration_engine.evaluate_migrations();
        
        if tasks.is_empty() {
            return MigrationResults::default();
        }
        
        self.migration_coordinator.execute_batch(tasks, self).await
    }
}
```

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_migration_decision() {
    let engine = MigrationEngine::new(MigrationPolicy::default());
    
    let stats = AccessStats {
        current_activation: 0.1,
        access_frequency: 0.5,
        idle_time: Duration::from_secs(3600),
        total_accesses: 10,
        pattern: AccessPattern::Decaying { 
            half_life: Duration::from_secs(1800) 
        },
        pattern_confidence: Confidence::HIGH,
    };
    
    let decision = engine.decide_hot_migration(&stats);
    assert!(matches!(decision, MigrationDecision::Demote(_)));
}

#[test]
fn test_pattern_classification() {
    let classifier = PatternClassifier::new();
    
    // Create periodic access pattern
    let mut history = Vec::new();
    for i in 0..20 {
        history.push(AccessEvent {
            vector_id: "test".to_string(),
            timestamp: SystemTime::now() + Duration::from_secs(i * 3600),
            activation: 0.8,
        });
    }
    
    let (pattern, confidence) = classifier.classify(&history);
    assert!(matches!(pattern, AccessPattern::Periodic { .. }));
    assert!(confidence > Confidence::MEDIUM);
}
```

## Acceptance Criteria
- [ ] Migration decisions based on multiple criteria
- [ ] Pattern detection accuracy >85%
- [ ] Memory pressure response within 100ms
- [ ] Zero data loss during migrations
- [ ] Batch migrations complete within 5s for 1000 vectors
- [ ] Confidence tracking for all migration decisions

## Performance Targets
- Migration decision: <1ms per vector
- Pattern classification: <10ms per vector
- Batch migration: <5ms per vector
- Memory pressure check: <10ms
- Maximum concurrent migrations: 100

## Risk Mitigation
- Rollback mechanism for failed migrations
- Duplicate detection before migration
- Throttling under high load
- Metrics and alerting for migration failures