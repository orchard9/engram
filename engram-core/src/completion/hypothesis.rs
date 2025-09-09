//! System 2 hypothesis generation for deliberative reasoning.

use super::{CompletionConfig, PartialEpisode};
use crate::{Confidence, Episode};
use chrono::Utc;
use std::collections::VecDeque;

/// Represents a hypothesis about a completed episode
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// The hypothesized episode
    pub episode: Episode,

    /// Evidence supporting this hypothesis
    pub evidence: Vec<Evidence>,

    /// Confidence in this hypothesis
    pub confidence: Confidence,

    /// Reasoning steps taken
    pub reasoning_chain: Vec<ReasoningStep>,
}

/// Evidence supporting a hypothesis
#[derive(Debug, Clone)]
pub enum Evidence {
    /// Direct memory match
    DirectRecall(String),

    /// Semantic similarity
    SemanticSimilarity(String, f32),

    /// Temporal proximity
    TemporalProximity(String, f64),

    /// Logical inference
    LogicalInference(String),

    /// Pattern-based prediction
    PatternPrediction(String),
}

/// A step in the reasoning process
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Description of the step
    pub description: String,

    /// Confidence after this step
    pub confidence: Confidence,

    /// Working memory state
    pub working_memory: Vec<String>,
}

/// System 2 reasoning engine for deliberative hypothesis generation
pub struct System2Reasoner {
    /// Configuration
    config: CompletionConfig,

    /// Working memory buffer
    working_memory: VecDeque<WorkingMemoryItem>,

    /// Rule base for logical inference
    inference_rules: Vec<InferenceRule>,

    /// Pattern templates for prediction
    pattern_templates: Vec<PatternTemplate>,
}

/// Item in working memory
#[derive(Debug, Clone)]
struct WorkingMemoryItem {
    content: String,
    activation: f32,
    source: String,
}

/// Logical inference rule
#[derive(Debug, Clone)]
struct InferenceRule {
    antecedents: Vec<String>,
    consequent: String,
    confidence: f32,
}

/// Pattern template for prediction
#[derive(Debug, Clone)]
struct PatternTemplate {
    pattern: Vec<String>,
    prediction: String,
    confidence: f32,
}

impl System2Reasoner {
    /// Create a new System 2 reasoner
    #[must_use]
    pub fn new(config: CompletionConfig) -> Self {
        Self {
            config,
            working_memory: VecDeque::with_capacity(7), // Miller's magic number
            inference_rules: Self::default_inference_rules(),
            pattern_templates: Self::default_pattern_templates(),
        }
    }

    /// Generate multiple hypotheses for a partial episode
    pub fn generate_hypotheses(
        &mut self,
        partial: &PartialEpisode,
        context_episodes: &[Episode],
    ) -> Vec<Hypothesis> {
        let mut hypotheses = Vec::new();

        // Clear working memory
        self.working_memory.clear();

        // Load partial information into working memory
        self.load_partial_to_working_memory(partial);

        // Generate hypotheses using different strategies

        // Strategy 1: Pattern completion
        if let Some(hyp) = self.pattern_based_hypothesis(partial, context_episodes) {
            hypotheses.push(hyp);
        }

        // Strategy 2: Logical inference
        if let Some(hyp) = self.inference_based_hypothesis(partial, context_episodes) {
            hypotheses.push(hyp);
        }

        // Strategy 3: Analogical reasoning
        if let Some(hyp) = self.analogy_based_hypothesis(partial, context_episodes) {
            hypotheses.push(hyp);
        }

        // Sort by confidence
        hypotheses.sort_by(|a, b| b.confidence.raw().partial_cmp(&a.confidence.raw()).unwrap());

        // Take top-k hypotheses
        hypotheses.truncate(self.config.num_hypotheses);

        hypotheses
    }

    /// Load partial episode information into working memory
    fn load_partial_to_working_memory(&mut self, partial: &PartialEpisode) {
        for (field, value) in &partial.known_fields {
            self.add_to_working_memory(WorkingMemoryItem {
                content: format!("{field}: {value}"),
                activation: 1.0,
                source: "input".to_string(),
            });
        }

        for context in &partial.temporal_context {
            self.add_to_working_memory(WorkingMemoryItem {
                content: format!("context: {context}"),
                activation: 0.8,
                source: "temporal".to_string(),
            });
        }
    }

    /// Add item to working memory with capacity constraint
    fn add_to_working_memory(&mut self, item: WorkingMemoryItem) {
        if self.working_memory.len() >= self.config.working_memory_capacity {
            self.working_memory.pop_front(); // Remove oldest/least active
        }
        self.working_memory.push_back(item);
    }

    /// Generate hypothesis using pattern matching
    fn pattern_based_hypothesis(
        &self,
        partial: &PartialEpisode,
        _context_episodes: &[Episode],
    ) -> Option<Hypothesis> {
        // Find matching pattern templates
        for template in &self.pattern_templates {
            if self.matches_pattern(partial, &template.pattern) {
                // Create hypothesis based on pattern prediction
                let mut embedding = [0.0f32; 768];

                // Use partial embedding as base
                for (i, val) in partial.partial_embedding.iter().enumerate() {
                    if i >= 768 {
                        break;
                    }
                    if let Some(v) = val {
                        embedding[i] = *v;
                    }
                }

                let episode = Episode::new(
                    format!("pattern_hypothesis_{}", Utc::now().timestamp()),
                    Utc::now(),
                    template.prediction.clone(),
                    embedding,
                    Confidence::exact(template.confidence),
                );

                let evidence = vec![Evidence::PatternPrediction(template.prediction.clone())];

                let reasoning_chain = vec![ReasoningStep {
                    description: format!("Matched pattern: {:?}", template.pattern),
                    confidence: Confidence::exact(template.confidence),
                    working_memory: self
                        .working_memory
                        .iter()
                        .map(|item| item.content.clone())
                        .collect(),
                }];

                return Some(Hypothesis {
                    episode,
                    evidence,
                    confidence: Confidence::exact(template.confidence * partial.cue_strength.raw()),
                    reasoning_chain,
                });
            }
        }

        None
    }

    /// Generate hypothesis using logical inference
    fn inference_based_hypothesis(
        &self,
        partial: &PartialEpisode,
        _context_episodes: &[Episode],
    ) -> Option<Hypothesis> {
        // Apply inference rules
        for rule in &self.inference_rules {
            if self.satisfies_antecedents(partial, &rule.antecedents) {
                let mut embedding = [0.0f32; 768];

                for (i, val) in partial.partial_embedding.iter().enumerate() {
                    if i >= 768 {
                        break;
                    }
                    if let Some(v) = val {
                        embedding[i] = *v;
                    }
                }

                let episode = Episode::new(
                    format!("inference_hypothesis_{}", Utc::now().timestamp()),
                    Utc::now(),
                    rule.consequent.clone(),
                    embedding,
                    Confidence::exact(rule.confidence),
                );

                let evidence = vec![Evidence::LogicalInference(rule.consequent.clone())];

                let reasoning_chain = vec![ReasoningStep {
                    description: format!(
                        "Applied rule: {:?} → {}",
                        rule.antecedents, rule.consequent
                    ),
                    confidence: Confidence::exact(rule.confidence),
                    working_memory: self
                        .working_memory
                        .iter()
                        .map(|item| item.content.clone())
                        .collect(),
                }];

                return Some(Hypothesis {
                    episode,
                    evidence,
                    confidence: Confidence::exact(rule.confidence * partial.cue_strength.raw()),
                    reasoning_chain,
                });
            }
        }

        None
    }

    /// Generate hypothesis using analogical reasoning
    fn analogy_based_hypothesis(
        &self,
        partial: &PartialEpisode,
        context_episodes: &[Episode],
    ) -> Option<Hypothesis> {
        // Find most similar episode in context
        let mut best_match = None;
        let mut best_similarity = 0.0;

        for episode in context_episodes {
            let similarity = self.calculate_similarity(partial, episode);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some(episode);
            }
        }

        if let Some(analog) = best_match {
            if best_similarity > 0.5 {
                // Create hypothesis based on analogy
                let mut new_episode = analog.clone();
                new_episode.id = format!("analogy_hypothesis_{}", Utc::now().timestamp());

                // Update with known fields
                if let Some(what) = partial.known_fields.get("what") {
                    new_episode.what = what.clone();
                }

                let evidence = vec![Evidence::SemanticSimilarity(
                    analog.id.clone(),
                    best_similarity as f32,
                )];

                let reasoning_chain = vec![ReasoningStep {
                    description: format!("Analogical reasoning from episode: {}", analog.id),
                    confidence: Confidence::exact(best_similarity as f32),
                    working_memory: self
                        .working_memory
                        .iter()
                        .map(|item| item.content.clone())
                        .collect(),
                }];

                return Some(Hypothesis {
                    episode: new_episode,
                    evidence,
                    confidence: Confidence::exact(
                        best_similarity as f32 * partial.cue_strength.raw(),
                    ),
                    reasoning_chain,
                });
            }
        }

        None
    }

    /// Check if partial matches a pattern (relaxed matching - ANY keyword can match)
    fn matches_pattern(&self, partial: &PartialEpisode, pattern: &[String]) -> bool {
        // Return true if ANY requirement is found (instead of ALL)
        for requirement in pattern {
            let found = partial
                .known_fields
                .values()
                .any(|v| v.to_lowercase().contains(&requirement.to_lowercase()));
            if found {
                return true;
            }
            // Also check temporal context for matches
            if partial.temporal_context.iter().any(|ctx| ctx.to_lowercase().contains(&requirement.to_lowercase())) {
                return true;
            }
        }
        false
    }

    /// Check if partial satisfies rule antecedents (relaxed matching)
    fn satisfies_antecedents(&self, partial: &PartialEpisode, antecedents: &[String]) -> bool {
        // Need at least half the antecedents to match (instead of all)
        let mut matches = 0;
        for antecedent in antecedents {
            let found = partial
                .known_fields
                .values()
                .any(|v| v.to_lowercase().contains(&antecedent.to_lowercase()))
                || partial.temporal_context.iter()
                    .any(|ctx| ctx.to_lowercase().contains(&antecedent.to_lowercase()));
            if found {
                matches += 1;
            }
        }
        // At least half the antecedents must match, or at least one if list is small
        matches >= (antecedents.len() + 1) / 2 || matches > 0 && antecedents.len() <= 2
    }

    /// Calculate similarity between partial and episode
    fn calculate_similarity(&self, partial: &PartialEpisode, episode: &Episode) -> f64 {
        let mut similarity = 0.0;
        let mut count = 0;

        if let Some(what) = partial.known_fields.get("what") {
            if episode.what.contains(what) {
                similarity += 1.0;
            }
            count += 1;
        }

        if let Some(where_loc) = partial.known_fields.get("where") {
            if let Some(ref ep_where) = episode.where_location {
                if ep_where.contains(where_loc) {
                    similarity += 1.0;
                }
            }
            count += 1;
        }

        if count > 0 {
            similarity / f64::from(count)
        } else {
            0.0
        }
    }

    /// Default inference rules
    fn default_inference_rules() -> Vec<InferenceRule> {
        vec![
            InferenceRule {
                antecedents: vec!["morning".to_string(), "coffee".to_string()],
                consequent: "breakfast routine".to_string(),
                confidence: 0.8,
            },
            InferenceRule {
                antecedents: vec!["meeting".to_string(), "calendar".to_string()],
                consequent: "work event".to_string(),
                confidence: 0.9,
            },
            InferenceRule {
                antecedents: vec!["meeting".to_string(), "office".to_string()],
                consequent: "business meeting".to_string(),
                confidence: 0.85,
            },
            InferenceRule {
                antecedents: vec!["meeting".to_string()],
                consequent: "collaborative work session".to_string(),
                confidence: 0.7,
            },
            InferenceRule {
                antecedents: vec!["office".to_string()],
                consequent: "workplace activity".to_string(),
                confidence: 0.6,
            },
            InferenceRule {
                antecedents: vec!["calendar".to_string()],
                consequent: "scheduled event".to_string(),
                confidence: 0.65,
            },
            InferenceRule {
                antecedents: vec!["evening".to_string(), "home".to_string()],
                consequent: "relaxation time".to_string(),
                confidence: 0.7,
            },
            InferenceRule {
                antecedents: vec!["breakfast".to_string()],
                consequent: "morning meal".to_string(),
                confidence: 0.9,
            },
        ]
    }

    /// Default pattern templates
    fn default_pattern_templates() -> Vec<PatternTemplate> {
        vec![
            PatternTemplate {
                pattern: vec!["wake".to_string(), "alarm".to_string()],
                prediction: "morning routine beginning".to_string(),
                confidence: 0.85,
            },
            PatternTemplate {
                pattern: vec!["meeting".to_string()],
                prediction: "collaborative work session".to_string(),
                confidence: 0.8,
            },
            PatternTemplate {
                pattern: vec!["office".to_string()],
                prediction: "workplace activity".to_string(),
                confidence: 0.75,
            },
            PatternTemplate {
                pattern: vec!["calendar".to_string()],
                prediction: "scheduled event".to_string(),
                confidence: 0.7,
            },
            PatternTemplate {
                pattern: vec!["breakfast".to_string()],
                prediction: "morning meal preparation".to_string(),
                confidence: 0.85,
            },
            PatternTemplate {
                pattern: vec!["drive".to_string(), "traffic".to_string()],
                prediction: "commute to destination".to_string(),
                confidence: 0.75,
            },
            PatternTemplate {
                pattern: vec!["tired".to_string(), "bed".to_string()],
                prediction: "preparing to sleep".to_string(),
                confidence: 0.9,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system2_reasoner_creation() {
        let config = CompletionConfig::default();
        let reasoner = System2Reasoner::new(config);
        assert_eq!(reasoner.working_memory.capacity(), 7);
        assert!(!reasoner.inference_rules.is_empty());
        assert!(!reasoner.pattern_templates.is_empty());
    }

    #[test]
    fn test_working_memory_capacity() {
        let config = CompletionConfig::default();
        let mut reasoner = System2Reasoner::new(config);

        // Add more items than capacity
        for i in 0..10 {
            reasoner.add_to_working_memory(WorkingMemoryItem {
                content: format!("item_{}", i),
                activation: 1.0,
                source: "test".to_string(),
            });
        }

        // Should maintain capacity limit
        assert_eq!(reasoner.working_memory.len(), 7);
    }
}
