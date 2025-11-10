//! Statistical distributions for realistic traffic generation

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, Normal, Poisson};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "pattern", rename_all = "snake_case")]
pub enum ArrivalPattern {
    /// Poisson arrival (random, memoryless)
    Poisson { lambda: f64 },
    /// Constant rate (stress test)
    Constant { rate: f64 },
    /// Periodic bursts
    PeriodicBurst {
        base_rate: f64,
        burst_rate: f64,
        burst_duration_sec: u64,
        burst_period_sec: u64,
    },
    /// Exponential inter-arrival times
    Exponential { lambda: f64 },
}

impl ArrivalPattern {
    /// Calculate inter-arrival time in seconds
    #[allow(dead_code)] // Used in replay mode
    pub fn next_interval<R: Rng>(&self, rng: &mut R, elapsed_secs: u64) -> f64 {
        match self {
            Self::Poisson { lambda } => {
                let poisson = Poisson::new(*lambda).expect("Invalid lambda");
                let count: f64 = poisson.sample(rng);
                // Convert count per unit time to interval
                if count > 0.0 {
                    1.0 / count
                } else {
                    1.0 / lambda // Fallback to mean rate
                }
            }
            Self::Constant { rate } => 1.0 / rate,
            Self::PeriodicBurst {
                base_rate,
                burst_rate,
                burst_duration_sec,
                burst_period_sec,
            } => {
                // Determine if we're in a burst period
                let position_in_cycle = elapsed_secs % burst_period_sec;
                let in_burst = position_in_cycle < *burst_duration_sec;

                let current_rate = if in_burst { *burst_rate } else { *base_rate };
                1.0 / current_rate
            }
            Self::Exponential { lambda } => {
                let exp = Exp::new(*lambda).expect("Invalid lambda");
                exp.sample(rng)
            }
        }
    }

    /// Get mean operations per second
    pub fn mean_rate(&self) -> f64 {
        match self {
            Self::Poisson { lambda } | Self::Exponential { lambda } => *lambda,
            Self::Constant { rate } => *rate,
            Self::PeriodicBurst {
                base_rate,
                burst_rate,
                burst_duration_sec,
                burst_period_sec,
            } => {
                // Weighted average
                let burst_fraction = *burst_duration_sec as f64 / *burst_period_sec as f64;
                burst_rate * burst_fraction + base_rate * (1.0 - burst_fraction)
            }
        }
    }
}

/// Embedding distribution patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingDistribution {
    /// Clustered embeddings (realistic)
    Clustered { num_clusters: usize, std_dev: f32 },
    /// Uniform random (pathological)
    Uniform,
    /// Single hot-spot (stress test)
    HotSpot { center: Vec<f32> },
}

impl EmbeddingDistribution {
    /// Generate an embedding based on the distribution
    pub fn generate<R: Rng>(&self, rng: &mut R, dim: usize) -> Vec<f32> {
        match self {
            Self::Clustered {
                num_clusters,
                std_dev,
            } => {
                // Select a random cluster
                let cluster_id = rng.gen_range(0..*num_clusters);

                // Generate cluster center deterministically from cluster ID
                let mut center = vec![0.0f32; dim];
                let cluster_rng_seed = cluster_id as u64;
                let mut cluster_rng = rand::rngs::StdRng::seed_from_u64(cluster_rng_seed);

                for val in &mut center {
                    *val = cluster_rng.gen_range(-1.0..1.0);
                }

                // Add noise around cluster center
                let normal = Normal::new(0.0, *std_dev as f64).expect("Invalid std_dev");
                let mut embedding = vec![0.0f32; dim];
                for i in 0..dim {
                    let noise = normal.sample(rng) as f32;
                    embedding[i] = (center[i] + noise).clamp(-1.0, 1.0);
                }

                // Normalize
                normalize(&mut embedding);
                embedding
            }
            Self::Uniform => {
                let mut embedding = vec![0.0f32; dim];
                for val in &mut embedding {
                    *val = rng.gen_range(-1.0..1.0);
                }
                normalize(&mut embedding);
                embedding
            }
            Self::HotSpot { center } => {
                let mut embedding = center.clone();
                embedding.resize(dim, 0.0);

                // Add small noise
                let normal = Normal::new(0.0, 0.01).expect("Invalid std_dev");
                for val in &mut embedding {
                    let noise = normal.sample(rng) as f32;
                    *val = (*val + noise).clamp(-1.0, 1.0);
                }

                normalize(&mut embedding);
                embedding
            }
        }
    }
}

/// Normalize a vector to unit length
fn normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in vec {
            *val /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_poisson_arrival_deterministic() {
        let pattern = ArrivalPattern::Poisson { lambda: 100.0 };
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let interval1 = pattern.next_interval(&mut rng1, 0);
            let interval2 = pattern.next_interval(&mut rng2, 0);
            assert!((interval1 - interval2).abs() < 1e-9);
        }
    }

    #[test]
    fn test_constant_arrival() {
        let pattern = ArrivalPattern::Constant { rate: 1000.0 };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let interval = pattern.next_interval(&mut rng, 0);
            assert!((interval - 0.001).abs() < 1e-9);
        }
    }

    #[test]
    fn test_periodic_burst() {
        let pattern = ArrivalPattern::PeriodicBurst {
            base_rate: 1000.0,
            burst_rate: 5000.0,
            burst_duration_sec: 10,
            burst_period_sec: 60,
        };

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // During burst (0-10 seconds)
        let burst_interval = pattern.next_interval(&mut rng, 5);
        assert!((burst_interval - 1.0 / 5000.0).abs() < 1e-9);

        // Outside burst (10-60 seconds)
        let base_interval = pattern.next_interval(&mut rng, 30);
        assert!((base_interval - 1.0 / 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_clustered_embeddings_deterministic() {
        let dist = EmbeddingDistribution::Clustered {
            num_clusters: 5,
            std_dev: 0.1,
        };

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);

        let emb1 = dist.generate(&mut rng1, 768);
        let emb2 = dist.generate(&mut rng2, 768);

        assert_eq!(emb1.len(), 768);
        assert_eq!(emb2.len(), 768);

        for (a, b) in emb1.iter().zip(emb2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_embedding_normalization() {
        let dist = EmbeddingDistribution::Uniform;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let embedding = dist.generate(&mut rng, 768);

        // Check unit norm
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
