//! Workload generator with deterministic seeding

use crate::distribution::{ArrivalPattern, EmbeddingDistribution};
use anyhow::Result;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    pub name: String,
    pub description: String,

    #[serde(default)]
    pub duration: DurationConfig,

    pub arrival: ArrivalPattern,

    pub operations: OperationWeights,

    pub data: DataConfig,

    #[serde(default)]
    pub validation: ValidationCriteria,

    #[serde(default)]
    pub chaos: ChaosConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationConfig {
    pub total_seconds: u64,
}

impl Default for DurationConfig {
    fn default() -> Self {
        Self { total_seconds: 300 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationWeights {
    pub store_weight: f64,
    pub recall_weight: f64,
    pub embedding_search_weight: f64,
    #[serde(default)]
    pub pattern_completion_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub num_nodes: usize,
    pub embedding_dim: usize,
    pub memory_spaces: usize,
    #[serde(default = "default_embedding_distribution")]
    pub embedding_distribution: EmbeddingDistribution,
}

fn default_embedding_distribution() -> EmbeddingDistribution {
    EmbeddingDistribution::Clustered {
        num_clusters: 10,
        std_dev: 0.1,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationCriteria {
    pub expected_p99_latency_ms: Option<f64>,
    pub expected_throughput_ops_sec: Option<f64>,
    pub max_error_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChaosConfig {
    pub enabled: bool,
}

impl WorkloadConfig {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn target_rate(&self) -> u64 {
        self.arrival.mean_rate() as u64
    }

    pub fn set_target_rate(&mut self, rate: u64) {
        // Update arrival pattern to match new rate
        self.arrival = ArrivalPattern::Constant { rate: rate as f64 };
    }

    pub fn duration(&self) -> u64 {
        self.duration.total_seconds
    }

    pub fn set_duration(&mut self, seconds: u64) {
        self.duration.total_seconds = seconds;
    }

    pub fn from_trace_metadata() -> Result<Self> {
        // Placeholder for trace-based config
        Ok(Self {
            name: "Trace Replay".to_string(),
            description: "Replayed from trace file".to_string(),
            duration: DurationConfig::default(),
            arrival: ArrivalPattern::Constant { rate: 1000.0 },
            operations: OperationWeights {
                store_weight: 0.5,
                recall_weight: 0.5,
                embedding_search_weight: 0.0,
                pattern_completion_weight: 0.0,
            },
            data: DataConfig {
                num_nodes: 100_000,
                embedding_dim: 768,
                memory_spaces: 1,
                embedding_distribution: default_embedding_distribution(),
            },
            validation: ValidationCriteria::default(),
            chaos: ChaosConfig::default(),
        })
    }
}

pub struct WorkloadGenerator {
    rng: StdRng,
    config: WorkloadConfig,
    total_weight: f64,
}

impl WorkloadGenerator {
    pub fn new(seed: u64, config: WorkloadConfig) -> Result<Self> {
        let total_weight = config.operations.store_weight
            + config.operations.recall_weight
            + config.operations.embedding_search_weight
            + config.operations.pattern_completion_weight;

        Ok(Self {
            rng: StdRng::seed_from_u64(seed),
            config,
            total_weight,
        })
    }

    pub fn next_operation(&mut self) -> Operation {
        let op_choice = self.rng.gen_range(0.0..self.total_weight);

        let mut cumulative = 0.0;

        cumulative += self.config.operations.store_weight;
        if op_choice < cumulative {
            return self.generate_store();
        }

        cumulative += self.config.operations.recall_weight;
        if op_choice < cumulative {
            return self.generate_recall();
        }

        cumulative += self.config.operations.embedding_search_weight;
        if op_choice < cumulative {
            return self.generate_search();
        }

        self.generate_pattern_completion()
    }

    fn generate_store(&mut self) -> Operation {
        let embedding = self
            .config
            .data
            .embedding_distribution
            .generate(&mut self.rng, self.config.data.embedding_dim);

        let memory_space = if self.config.data.memory_spaces > 1 {
            format!(
                "space_{}",
                self.rng.gen_range(0..self.config.data.memory_spaces)
            )
        } else {
            "default".to_string()
        };

        Operation::Store {
            memory: MemoryData {
                embedding,
                memory_space,
                confidence: self.rng.gen_range(0.7..1.0),
            },
        }
    }

    fn generate_recall(&mut self) -> Operation {
        let cue_embedding = self
            .config
            .data
            .embedding_distribution
            .generate(&mut self.rng, self.config.data.embedding_dim);

        let memory_space = if self.config.data.memory_spaces > 1 {
            format!(
                "space_{}",
                self.rng.gen_range(0..self.config.data.memory_spaces)
            )
        } else {
            "default".to_string()
        };

        Operation::Recall {
            cue: CueData {
                embedding: cue_embedding,
                memory_space,
                threshold: self.rng.gen_range(0.5..0.9),
                max_depth: self.rng.gen_range(1..=5),
            },
        }
    }

    fn generate_search(&mut self) -> Operation {
        let query = self
            .config
            .data
            .embedding_distribution
            .generate(&mut self.rng, self.config.data.embedding_dim);

        let k = self.rng.gen_range(5..=50);

        Operation::EmbeddingSearch { query, k }
    }

    fn generate_pattern_completion(&mut self) -> Operation {
        let partial_embedding = self
            .config
            .data
            .embedding_distribution
            .generate(&mut self.rng, self.config.data.embedding_dim / 2);

        Operation::PatternCompletion {
            partial: partial_embedding,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Operation {
    Store { memory: MemoryData },
    Recall { cue: CueData },
    EmbeddingSearch { query: Vec<f32>, k: usize },
    PatternCompletion { partial: Vec<f32> },
}

impl Operation {
    pub fn op_type(&self) -> OperationType {
        match self {
            Self::Store { .. } => OperationType::Store,
            Self::Recall { .. } => OperationType::Recall,
            Self::EmbeddingSearch { .. } => OperationType::Search,
            Self::PatternCompletion { .. } => OperationType::PatternCompletion,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OperationType {
    Store,
    Recall,
    Search,
    PatternCompletion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryData {
    pub embedding: Vec<f32>,
    pub memory_space: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CueData {
    pub embedding: Vec<f32>,
    pub memory_space: String,
    pub threshold: f32,
    pub max_depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> WorkloadConfig {
        WorkloadConfig {
            name: "Test".to_string(),
            description: "Test config".to_string(),
            duration: DurationConfig { total_seconds: 60 },
            arrival: ArrivalPattern::Constant { rate: 1000.0 },
            operations: OperationWeights {
                store_weight: 0.5,
                recall_weight: 0.5,
                embedding_search_weight: 0.0,
                pattern_completion_weight: 0.0,
            },
            data: DataConfig {
                num_nodes: 1000,
                embedding_dim: 768,
                memory_spaces: 1,
                embedding_distribution: EmbeddingDistribution::Uniform,
            },
            validation: ValidationCriteria::default(),
            chaos: ChaosConfig::default(),
        }
    }

    #[test]
    fn test_deterministic_generation() {
        let config = test_config();
        let mut gen1 = WorkloadGenerator::new(42, config.clone()).unwrap();
        let mut gen2 = WorkloadGenerator::new(42, config).unwrap();

        for _ in 0..100 {
            let op1 = gen1.next_operation();
            let op2 = gen2.next_operation();

            assert_eq!(op1.op_type(), op2.op_type());
        }
    }

    #[test]
    fn test_operation_weights() {
        let config = test_config();
        let mut generator = WorkloadGenerator::new(42, config).unwrap();

        let mut store_count = 0;
        let mut recall_count = 0;

        for _ in 0..1000 {
            match generator.next_operation().op_type() {
                OperationType::Store => store_count += 1,
                OperationType::Recall => recall_count += 1,
                _ => {}
            }
        }

        // Should be roughly 50/50
        let store_ratio = store_count as f64 / 1000.0;
        let recall_ratio = recall_count as f64 / 1000.0;
        assert!((store_ratio - 0.5).abs() < 0.1);
        assert!((recall_ratio - 0.5).abs() < 0.1);
    }
}
