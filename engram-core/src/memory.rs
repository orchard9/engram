//! Core memory types with cognitive confidence integration.
//!
//! This module implements Memory, Episode, and Cue types that use cognitive
//! confidence as first-class properties for probabilistic memory operations.
//! All types support confidence propagation and follow forgetting curve research.

use crate::Confidence;
use crate::numeric::saturating_f32_from_f64;
use atomic_float::AtomicF32;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;

fn small_usize_to_f32(value: usize) -> f32 {
    let value_u16 = u16::try_from(value).unwrap_or(u16::MAX);
    f32::from(value_u16)
}

/// Core Memory type with 768-dimensional embeddings and cognitive confidence.
///
/// Represents a memory in the probabilistic graph with activation, confidence,
/// and decay properties that follow psychological research on memory systems.
#[derive(Debug, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier for this memory
    pub id: String,

    /// Dense 768-dimensional embedding vector for semantic content
    #[serde(with = "embedding_serde")]
    pub embedding: [f32; 768],

    /// Current activation level (atomic for concurrent updates)
    #[serde(skip)]
    activation: AtomicF32,

    /// Activation value for serialization
    #[serde(rename = "activation")]
    pub activation_value: f32,

    /// Cognitive confidence in memory accuracy and reliability
    pub confidence: Confidence,

    /// When this memory was last accessed
    pub last_access: DateTime<Utc>,

    /// When this memory was created/encoded
    pub created_at: DateTime<Utc>,

    /// Node-specific decay rate for forgetting curve
    pub decay_rate: f32,

    /// Optional content for human-readable debugging
    pub content: Option<String>,
}

// Custom serialization for large arrays
mod embedding_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(embedding: &[f32; 768], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        embedding.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f32; 768], D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<f32>::deserialize(deserializer)?;
        if vec.len() != 768 {
            return Err(serde::de::Error::custom(format!(
                "Expected array of length 768, got {}",
                vec.len()
            )));
        }

        let mut array = [0.0f32; 768];
        array.copy_from_slice(&vec);
        Ok(array)
    }
}

impl Memory {
    /// Creates a new memory with given parameters
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)] // The embedding becomes owned by the memory; taking by value avoids an extra copy.
    pub fn new(id: String, embedding: [f32; 768], confidence: Confidence) -> Self {
        let now = Utc::now();
        Self {
            id,
            embedding,
            activation: AtomicF32::new(0.0),
            activation_value: 0.0,
            confidence,
            last_access: now,
            created_at: now,
            decay_rate: 0.1, // Default decay rate
            content: None,
        }
    }

    /// Create a memory from an episode with initial activation
    #[must_use]
    pub fn from_episode(episode: Episode, activation: f32) -> Self {
        let mut memory = Self::new(episode.id, episode.embedding, episode.encoding_confidence);
        memory.set_activation(activation);
        memory.content = Some(episode.what);
        memory.created_at = episode.when;
        memory.last_access = episode.when;
        memory
    }

    /// Gets current activation level (thread-safe)
    pub fn activation(&self) -> f32 {
        self.activation.load(Ordering::Relaxed)
    }

    /// Sets activation level (thread-safe)
    pub fn set_activation(&self, value: f32) {
        let clamped = value.clamp(0.0, 1.0);
        self.activation.store(clamped, Ordering::Relaxed);
    }

    /// Adds to current activation (thread-safe)
    pub fn add_activation(&self, delta: f32) {
        let current = self.activation();
        let new_value = (current + delta).clamp(0.0, 1.0);
        self.set_activation(new_value);
    }

    /// Cognitive confidence methods using System 1 thinking patterns
    /// Natural confidence check: "Does this memory seem reliable?"
    pub const fn seems_reliable(&self) -> bool {
        self.confidence.seems_legitimate()
    }

    /// Natural confidence check: "Is this memory accurate?"
    pub const fn is_accurate(&self) -> bool {
        self.confidence.is_high()
    }

    /// Natural confidence check: "Should I trust this memory?"
    pub fn is_trustworthy(&self) -> bool {
        self.confidence.raw() >= 0.8
    }

    /// Apply forgetting curve decay to confidence based on time elapsed
    pub fn apply_forgetting_decay(&mut self, time_elapsed_hours: f64) {
        // Ebbinghaus forgetting curve: R = e^(-t/S)
        // where R is retention, t is time, S is memory strength
        let strength = 1.0 / f64::from(self.decay_rate);
        let retention_factor = (-time_elapsed_hours / strength).exp();

        let current_confidence = self.confidence.raw();
        let decayed_confidence = current_confidence * saturating_f32_from_f64(retention_factor);

        self.confidence = Confidence::exact(decayed_confidence);
    }

    /// Update confidence using base rate information (prevents base rate neglect)
    pub const fn update_confidence_with_base_rate(&mut self, base_rate: Confidence) {
        self.confidence = self.confidence.update_with_base_rate(base_rate);
    }

    /// Apply overconfidence calibration
    pub const fn calibrate_confidence(&mut self) {
        self.confidence = self.confidence.calibrate_overconfidence();
    }
}

impl Clone for Memory {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            embedding: self.embedding,
            activation: AtomicF32::new(self.activation()),
            activation_value: self.activation_value,
            confidence: self.confidence,
            last_access: self.last_access,
            created_at: self.created_at,
            decay_rate: self.decay_rate,
            content: self.content.clone(),
        }
    }
}

/// Episode type capturing temporal, spatial, and semantic information.
///
/// Episodes represent specific memory events with rich contextual information
/// and confidence measures for encoding quality and vividness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier
    pub id: String,

    /// When the episode occurred
    pub when: DateTime<Utc>,

    /// Where the episode occurred (optional location)
    pub where_location: Option<String>,

    /// Who was involved (optional participants)
    pub who: Option<Vec<String>>,

    /// What happened (semantic content)
    pub what: String,

    /// Embedding representation of the episode
    #[serde(with = "embedding_serde")]
    pub embedding: [f32; 768],

    /// Confidence in episode encoding quality
    pub encoding_confidence: Confidence,

    /// Confidence in episode vividness/detail richness
    pub vividness_confidence: Confidence,

    /// Overall episode reliability confidence
    pub reliability_confidence: Confidence,

    /// When this episode was last recalled
    pub last_recall: DateTime<Utc>,

    /// Number of times this episode has been recalled
    pub recall_count: u32,

    /// Decay rate for this specific episode
    pub decay_rate: f32,
}

impl Episode {
    /// Creates a new episode
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)] // Owning the embedding is intentional to keep episodes self-contained.
    pub fn new(
        id: String,
        when: DateTime<Utc>,
        what: String,
        embedding: [f32; 768],
        encoding_confidence: Confidence,
    ) -> Self {
        Self {
            id,
            when,
            where_location: None,
            who: None,
            what,
            embedding,
            encoding_confidence,
            vividness_confidence: encoding_confidence, // Start with same confidence
            reliability_confidence: encoding_confidence,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.05, // Episodes decay slower than individual memories
        }
    }

    /// Cognitive confidence methods for episodes
    /// "Is this episode vivid and detailed?"
    #[must_use]
    pub const fn is_vivid(&self) -> bool {
        self.vividness_confidence.is_high()
    }

    /// "Can I trust the details of this episode?"
    #[must_use]
    pub const fn is_detailed(&self) -> bool {
        self.encoding_confidence.seems_legitimate()
    }

    /// "Does this episode feel real and accurate?"
    #[must_use]
    pub const fn feels_authentic(&self) -> bool {
        self.reliability_confidence.is_high() && self.encoding_confidence.seems_legitimate()
    }

    /// Record a recall event (affects confidence over time)
    pub fn record_recall(&mut self) {
        self.last_recall = Utc::now();
        self.recall_count += 1;

        // Testing effect: repeated recall can strengthen confidence
        if self.recall_count % 3 == 0 {
            let boost = Confidence::exact(0.05); // Small confidence boost
            self.reliability_confidence = self
                .reliability_confidence
                .combine_weighted(boost, 0.95, 0.05);
        }
    }

    /// Apply forgetting to all episode confidence measures
    pub fn apply_episode_forgetting(&mut self, time_elapsed_hours: f64) {
        let strength = 1.0 / f64::from(self.decay_rate);
        let retention_factor = (-time_elapsed_hours / strength).exp();
        let retention_factor = saturating_f32_from_f64(retention_factor);

        // Different confidence types decay at different rates
        let encoding_decay = Confidence::exact(self.encoding_confidence.raw() * retention_factor);
        let vividness_decay = Confidence::exact(
            self.vividness_confidence.raw() * retention_factor * 0.8, // Vividness decays faster
        );
        let reliability_decay =
            Confidence::exact(self.reliability_confidence.raw() * retention_factor * 0.9);

        self.encoding_confidence = encoding_decay;
        self.vividness_confidence = vividness_decay;
        self.reliability_confidence = reliability_decay;
    }

    /// Create a partial episode from available fields for pattern completion
    #[cfg(feature = "pattern_completion")]
    #[must_use]
    pub fn to_partial(&self, mask_percentage: f32) -> crate::completion::PartialEpisode {
        use std::collections::HashMap;

        let mut known_fields = HashMap::new();

        // Randomly mask some fields (simplified for now)
        if mask_percentage < 0.25 {
            known_fields.insert("what".to_string(), self.what.clone());
        }
        if mask_percentage < 0.5 {
            if let Some(ref loc) = self.where_location {
                known_fields.insert("where".to_string(), loc.clone());
            }
        }
        if mask_percentage < 0.75 {
            if let Some(ref who) = self.who {
                known_fields.insert("who".to_string(), who.join(", "));
            }
        }

        // Mask embedding dimensions
        let mut partial_embedding = Vec::with_capacity(768);
        for i in 0..768 {
            let index_ratio = small_usize_to_f32(i) / 768.0;
            if index_ratio > mask_percentage {
                partial_embedding.push(Some(self.embedding[i]));
            } else {
                partial_embedding.push(None);
            }
        }

        crate::completion::PartialEpisode {
            known_fields,
            partial_embedding,
            cue_strength: self.encoding_confidence,
            temporal_context: Vec::new(),
        }
    }

    /// Check if this episode is complete (has all major fields)
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        !self.what.is_empty() && self.where_location.is_some() && self.who.is_some()
    }

    /// Calculate completeness percentage
    #[must_use]
    pub fn completeness(&self) -> f32 {
        let mut score = 0.0;
        let mut total = 0.0;

        // Check what field
        if !self.what.is_empty() {
            score += 1.0;
        }
        total += 1.0;

        // Check where field
        if self.where_location.is_some() {
            score += 1.0;
        }
        total += 1.0;

        // Check who field
        if self.who.is_some() {
            score += 1.0;
        }
        total += 1.0;

        // Check embedding quality (non-zero values)
        let non_zero_count = self.embedding.iter().filter(|&&x| x != 0.0).count();
        score += small_usize_to_f32(non_zero_count) / 768.0;
        total += 1.0;

        score / total
    }
}

/// Different types of cues for memory queries
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)] // Embedding cues need to embed the vector inline to avoid extra allocations downstream.
pub enum CueType {
    /// Embedding-based similarity search
    Embedding {
        #[serde(with = "embedding_serde")]
        /// Embedding vector for similarity search
        vector: [f32; 768],
        /// Similarity threshold for matches
        threshold: Confidence,
    },
    /// Context-based search (temporal, spatial)
    Context {
        /// Optional time range for temporal filtering
        time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
        /// Optional location for spatial filtering
        location: Option<String>,
        /// Confidence threshold for context matches
        confidence_threshold: Confidence,
    },
    /// Temporal pattern search
    Temporal {
        /// Temporal pattern to match
        pattern: TemporalPattern,
        /// Confidence threshold for pattern matches
        confidence_threshold: Confidence,
    },
    /// Semantic content search
    Semantic {
        /// Semantic content to search for
        content: String,
        /// Fuzzy matching threshold
        fuzzy_threshold: Confidence,
    },
}

/// Temporal patterns for memory queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPattern {
    /// Events before a specific time
    Before(DateTime<Utc>),
    /// Events after a specific time
    After(DateTime<Utc>),
    /// Events within a time range
    Between(DateTime<Utc>, DateTime<Utc>),
    /// Recent events (within duration)
    Recent(chrono::Duration),
}

/// Cue for memory retrieval with confidence thresholds
#[allow(clippy::struct_field_names)] // Maintain stable serialized field names that align with the public API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cue {
    /// Unique identifier for this cue
    pub id: String,

    /// Type of cue and its parameters
    pub cue_type: CueType,

    /// Confidence in the cue itself
    pub cue_confidence: Confidence,

    /// Minimum confidence threshold for results
    pub result_threshold: Confidence,

    /// Maximum number of results to return
    pub max_results: usize,
}

impl Cue {
    /// Creates a new embedding-based cue
    #[must_use]
    #[allow(clippy::missing_const_for_fn, clippy::large_types_passed_by_value)]
    pub fn embedding(id: String, vector: [f32; 768], threshold: Confidence) -> Self {
        Self {
            id,
            cue_type: CueType::Embedding { vector, threshold },
            cue_confidence: Confidence::HIGH,
            result_threshold: threshold,
            max_results: 100,
        }
    }

    /// Creates a context-based cue
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn context(
        id: String,
        time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
        location: Option<String>,
        confidence_threshold: Confidence,
    ) -> Self {
        Self {
            id,
            cue_type: CueType::Context {
                time_range,
                location,
                confidence_threshold,
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: confidence_threshold,
            max_results: 50,
        }
    }

    /// Creates a semantic content cue
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn semantic(id: String, content: String, fuzzy_threshold: Confidence) -> Self {
        Self {
            id,
            cue_type: CueType::Semantic {
                content,
                fuzzy_threshold,
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: fuzzy_threshold,
            max_results: 75,
        }
    }

    /// Cognitive confidence methods for cues
    /// "Is this cue likely to return good results?"
    #[must_use]
    pub const fn seems_effective(&self) -> bool {
        self.cue_confidence.seems_legitimate()
    }

    /// "Should I trust results from this cue?"
    #[must_use]
    pub const fn is_reliable(&self) -> bool {
        self.cue_confidence.is_high()
    }

    /// Adjust result threshold based on cue confidence
    #[must_use]
    pub const fn adaptive_threshold(&self) -> Confidence {
        self.result_threshold.combine_weighted(
            self.cue_confidence,
            0.7, // Weight original threshold more
            0.3, // Weight cue confidence less
        )
    }
}

// Builder patterns with typestate for compile-time correctness

/// Typestate markers for Memory builder
pub mod memory_builder_states {
    /// Builder state: no ID set
    #[derive(Debug, Clone, Copy)]
    pub struct NoId;
    /// Builder state: no embedding set
    #[derive(Debug, Clone, Copy)]
    pub struct NoEmbedding;
    /// Builder state: no confidence set
    #[derive(Debug, Clone, Copy)]
    pub struct NoConfidence;
    /// Builder state: ready to build
    #[derive(Debug, Clone, Copy)]
    pub struct Ready;
}

/// Builder for Memory with compile-time validation
pub struct MemoryBuilder<State> {
    id: Option<String>,
    embedding: Option<[f32; 768]>,
    confidence: Option<Confidence>,
    content: Option<String>,
    decay_rate: f32,
    _state: PhantomData<State>,
}

impl Default for MemoryBuilder<memory_builder_states::NoId> {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBuilder<memory_builder_states::NoId> {
    /// Start building a new memory
    #[must_use]
    pub const fn new() -> Self {
        Self {
            id: None,
            embedding: None,
            confidence: None,
            content: None,
            decay_rate: 0.1,
            _state: PhantomData,
        }
    }
}

impl<State> MemoryBuilder<State> {
    /// Set the memory ID
    #[must_use]
    pub fn id(self, id: String) -> MemoryBuilder<memory_builder_states::NoEmbedding> {
        MemoryBuilder {
            id: Some(id),
            embedding: self.embedding,
            confidence: self.confidence,
            content: self.content,
            decay_rate: self.decay_rate,
            _state: PhantomData,
        }
    }
}

impl MemoryBuilder<memory_builder_states::NoEmbedding> {
    /// Set the embedding
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)]
    pub fn embedding(
        self,
        embedding: [f32; 768],
    ) -> MemoryBuilder<memory_builder_states::NoConfidence> {
        MemoryBuilder {
            id: self.id,
            embedding: Some(embedding),
            confidence: self.confidence,
            content: self.content,
            decay_rate: self.decay_rate,
            _state: PhantomData,
        }
    }
}

impl MemoryBuilder<memory_builder_states::NoConfidence> {
    /// Set the confidence
    #[must_use]
    pub fn confidence(self, confidence: Confidence) -> MemoryBuilder<memory_builder_states::Ready> {
        MemoryBuilder {
            id: self.id,
            embedding: self.embedding,
            confidence: Some(confidence),
            content: self.content,
            decay_rate: self.decay_rate,
            _state: PhantomData,
        }
    }
}

impl MemoryBuilder<memory_builder_states::Ready> {
    /// Set optional content
    #[must_use]
    pub fn content(mut self, content: String) -> Self {
        self.content = Some(content);
        self
    }

    /// Set decay rate
    #[must_use]
    pub const fn decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate.clamp(0.001, 1.0);
        self
    }

    /// Build the memory.
    ///
    /// # Panics
    ///
    /// Panics if typestate invariants are violated and required fields were not
    /// supplied before calling `build`. When the builder is used through the
    /// provided state transitions this condition is unreachable.
    #[must_use]
    pub fn build(self) -> Memory {
        let Self {
            id,
            embedding,
            confidence,
            content,
            decay_rate,
            _state: _,
        } = self;

        let (Some(id), Some(embedding), Some(confidence)) = (id, embedding, confidence) else {
            unreachable!("typestate guarantees required fields are set");
        };

        let mut memory = Memory::new(id, embedding, confidence);
        memory.content = content;
        memory.decay_rate = decay_rate;
        memory
    }
}

/// Similar builder pattern for Episode
pub mod episode_builder_states {
    /// Builder state: no ID set
    #[derive(Debug, Clone, Copy)]
    pub struct NoId;
    /// Builder state: no when timestamp set
    #[derive(Debug, Clone, Copy)]
    pub struct NoWhen;
    /// Builder state: no what content set
    #[derive(Debug, Clone, Copy)]
    pub struct NoWhat;
    /// Builder state: no embedding set
    #[derive(Debug, Clone, Copy)]
    pub struct NoEmbedding;
    /// Builder state: no confidence set
    #[derive(Debug, Clone, Copy)]
    pub struct NoConfidence;
    /// Builder state: ready to build
    #[derive(Debug, Clone, Copy)]
    pub struct Ready;
}

/// Builder for Episode with compile-time validation
pub struct EpisodeBuilder<State> {
    id: Option<String>,
    when: Option<DateTime<Utc>>,
    what: Option<String>,
    embedding: Option<[f32; 768]>,
    encoding_confidence: Option<Confidence>,
    where_location: Option<String>,
    who: Option<Vec<String>>,
    decay_rate: f32,
    _state: PhantomData<State>,
}

impl Default for EpisodeBuilder<episode_builder_states::NoId> {
    fn default() -> Self {
        Self::new()
    }
}

impl EpisodeBuilder<episode_builder_states::NoId> {
    /// Start building a new episode
    #[must_use]
    pub const fn new() -> Self {
        Self {
            id: None,
            when: None,
            what: None,
            embedding: None,
            encoding_confidence: None,
            where_location: None,
            who: None,
            decay_rate: 0.05,
            _state: PhantomData,
        }
    }
}

impl<State> EpisodeBuilder<State> {
    /// Set the episode ID
    #[must_use]
    pub fn id(self, id: String) -> EpisodeBuilder<episode_builder_states::NoWhen> {
        EpisodeBuilder {
            id: Some(id),
            when: self.when,
            what: self.what,
            embedding: self.embedding,
            encoding_confidence: self.encoding_confidence,
            where_location: self.where_location,
            who: self.who,
            decay_rate: self.decay_rate,
            _state: PhantomData,
        }
    }
}

impl EpisodeBuilder<episode_builder_states::NoWhen> {
    /// Set when the episode occurred
    #[must_use]
    pub fn when(self, when: DateTime<Utc>) -> EpisodeBuilder<episode_builder_states::NoWhat> {
        EpisodeBuilder {
            id: self.id,
            when: Some(when),
            what: self.what,
            embedding: self.embedding,
            encoding_confidence: self.encoding_confidence,
            where_location: self.where_location,
            who: self.who,
            decay_rate: self.decay_rate,
            _state: PhantomData,
        }
    }
}

impl EpisodeBuilder<episode_builder_states::NoWhat> {
    /// Set what happened
    #[must_use]
    pub fn what(self, what: String) -> EpisodeBuilder<episode_builder_states::NoEmbedding> {
        EpisodeBuilder {
            id: self.id,
            when: self.when,
            what: Some(what),
            embedding: self.embedding,
            encoding_confidence: self.encoding_confidence,
            where_location: self.where_location,
            who: self.who,
            decay_rate: self.decay_rate,
            _state: PhantomData,
        }
    }
}

impl EpisodeBuilder<episode_builder_states::NoEmbedding> {
    /// Set embedding
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)]
    pub fn embedding(
        self,
        embedding: [f32; 768],
    ) -> EpisodeBuilder<episode_builder_states::NoConfidence> {
        EpisodeBuilder {
            id: self.id,
            when: self.when,
            what: self.what,
            embedding: Some(embedding),
            encoding_confidence: self.encoding_confidence,
            where_location: self.where_location,
            who: self.who,
            decay_rate: self.decay_rate,
            _state: PhantomData,
        }
    }
}

impl EpisodeBuilder<episode_builder_states::NoConfidence> {
    /// Set encoding confidence
    #[must_use]
    pub fn confidence(
        self,
        confidence: Confidence,
    ) -> EpisodeBuilder<episode_builder_states::Ready> {
        EpisodeBuilder {
            id: self.id,
            when: self.when,
            what: self.what,
            embedding: self.embedding,
            encoding_confidence: Some(confidence),
            where_location: self.where_location,
            who: self.who,
            decay_rate: self.decay_rate,
            _state: PhantomData,
        }
    }
}

impl EpisodeBuilder<episode_builder_states::Ready> {
    /// Set location
    #[must_use]
    pub fn where_location(mut self, location: String) -> Self {
        self.where_location = Some(location);
        self
    }

    /// Set participants
    #[must_use]
    pub fn who(mut self, participants: Vec<String>) -> Self {
        self.who = Some(participants);
        self
    }

    /// Set decay rate
    #[must_use]
    pub const fn decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate.clamp(0.001, 1.0);
        self
    }

    /// Build the episode.
    ///
    /// # Panics
    ///
    /// Panics if typestate invariants are violated and required fields were not
    /// populated prior to calling `build`. Following the builder's state flow
    /// keeps this branch unreachable.
    #[must_use]
    pub fn build(self) -> Episode {
        let Self {
            id,
            when,
            what,
            embedding,
            encoding_confidence,
            where_location,
            who,
            decay_rate,
            _state: _,
        } = self;

        let (Some(id), Some(when), Some(what), Some(embedding), Some(encoding_confidence)) =
            (id, when, what, embedding, encoding_confidence)
        else {
            unreachable!("typestate guarantees required episode fields are set");
        };

        let mut episode = Episode::new(id, when, what, embedding, encoding_confidence);
        episode.where_location = where_location;
        episode.who = who;
        episode.decay_rate = decay_rate;
        episode
    }
}

/// Typestate markers for Cue builder
pub mod cue_builder_states {
    /// Builder state: no ID set
    #[derive(Debug, Clone, Copy)]
    pub struct NoId;
    /// Builder state: no cue type set
    #[derive(Debug, Clone, Copy)]
    pub struct NoCueType;
    /// Builder state: ready to build
    #[derive(Debug, Clone, Copy)]
    pub struct Ready;
}

/// Builder for Cue with compile-time validation
///
/// This builder ensures that no Cue can be constructed without the required
/// ID and cue type, following cognitive principles of preventing systematic
/// construction errors at compile time.
pub struct CueBuilder<State> {
    id: Option<String>,
    cue_type: Option<CueType>,
    cue_confidence: Confidence,
    result_threshold: Confidence,
    max_results: usize,
    _state: PhantomData<State>,
}

impl Default for CueBuilder<cue_builder_states::NoId> {
    fn default() -> Self {
        Self::new()
    }
}

impl CueBuilder<cue_builder_states::NoId> {
    /// Start building a new cue
    ///
    /// # Cognitive Design
    /// Beginning with `NoId` state guides developers through the natural
    /// construction sequence, preventing the common error of forgetting
    /// to set essential identifying information.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            id: None,
            cue_type: None,
            cue_confidence: Confidence::MEDIUM,
            result_threshold: Confidence::LOW,
            max_results: 10,
            _state: PhantomData,
        }
    }
}

impl<State> CueBuilder<State> {
    /// Set the cue ID (required first step)
    ///
    /// # Cognitive Design  
    /// The ID is required first as it establishes the identity of the cue,
    /// matching how humans naturally think about naming before detailing.
    #[must_use]
    pub fn id(self, id: String) -> CueBuilder<cue_builder_states::NoCueType> {
        CueBuilder {
            id: Some(id),
            cue_type: self.cue_type,
            cue_confidence: self.cue_confidence,
            result_threshold: self.result_threshold,
            max_results: self.max_results,
            _state: PhantomData,
        }
    }
}

impl CueBuilder<cue_builder_states::NoCueType> {
    /// Set cue type to embedding similarity search
    ///
    /// # Cognitive Design
    /// Semantic embedding search is the most intuitive for developers,
    /// so it's provided as a simple method that transitions to Ready state.
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)]
    pub fn embedding_search(
        self,
        vector: [f32; 768],
        threshold: Confidence,
    ) -> CueBuilder<cue_builder_states::Ready> {
        CueBuilder {
            id: self.id,
            cue_type: Some(CueType::Embedding { vector, threshold }),
            cue_confidence: self.cue_confidence,
            result_threshold: self.result_threshold,
            max_results: self.max_results,
            _state: PhantomData,
        }
    }

    /// Set cue type to context-based search
    ///
    /// # Cognitive Design
    /// Context search is natural for temporal/spatial queries,
    /// allowing optional parameters to match common use patterns.
    #[must_use]
    pub fn context_search(
        self,
        time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
        location: Option<String>,
        confidence_threshold: Confidence,
    ) -> CueBuilder<cue_builder_states::Ready> {
        CueBuilder {
            id: self.id,
            cue_type: Some(CueType::Context {
                time_range,
                location,
                confidence_threshold,
            }),
            cue_confidence: self.cue_confidence,
            result_threshold: self.result_threshold,
            max_results: self.max_results,
            _state: PhantomData,
        }
    }

    /// Set cue type to temporal pattern search
    ///
    /// # Cognitive Design
    /// Temporal patterns match how humans naturally think about
    /// time-based queries (before, after, between, recent).
    #[must_use]
    pub fn temporal_search(
        self,
        pattern: TemporalPattern,
        confidence_threshold: Confidence,
    ) -> CueBuilder<cue_builder_states::Ready> {
        CueBuilder {
            id: self.id,
            cue_type: Some(CueType::Temporal {
                pattern,
                confidence_threshold,
            }),
            cue_confidence: self.cue_confidence,
            result_threshold: self.result_threshold,
            max_results: self.max_results,
            _state: PhantomData,
        }
    }

    /// Set cue type to semantic content search
    ///
    /// # Cognitive Design
    /// Text-based search is intuitive for developers familiar with
    /// traditional search interfaces, with fuzzy matching support.
    #[must_use]
    pub fn semantic_search(
        self,
        content: String,
        fuzzy_threshold: Confidence,
    ) -> CueBuilder<cue_builder_states::Ready> {
        CueBuilder {
            id: self.id,
            cue_type: Some(CueType::Semantic {
                content,
                fuzzy_threshold,
            }),
            cue_confidence: self.cue_confidence,
            result_threshold: self.result_threshold,
            max_results: self.max_results,
            _state: PhantomData,
        }
    }
}

impl CueBuilder<cue_builder_states::Ready> {
    /// Set confidence in the cue itself (optional)
    ///
    /// # Cognitive Design
    /// Cue confidence defaults to MEDIUM, which is cognitively reasonable.
    /// Explicit setting available for fine-tuning retrieval behavior.
    #[must_use]
    pub const fn cue_confidence(mut self, confidence: Confidence) -> Self {
        self.cue_confidence = confidence;
        self
    }

    /// Set minimum confidence threshold for results (optional)
    ///
    /// # Cognitive Design
    /// Result threshold defaults to LOW to be inclusive, but can be
    /// raised to improve precision at the cost of recall.
    #[must_use]
    pub const fn result_threshold(mut self, threshold: Confidence) -> Self {
        self.result_threshold = threshold;
        self
    }

    /// Set maximum number of results to return (optional)
    ///
    /// # Cognitive Design
    /// Defaults to 10 results, which fits working memory constraints
    /// (Miller's 7±2) while allowing reasonable result sets.
    #[must_use]
    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = max.clamp(1, 1000); // Prevent unreasonable limits
        self
    }

    /// Build the cue (only available when all required fields are set).
    ///
    /// # Cognitive Design
    /// Build method only available in Ready state, preventing construction
    /// of incomplete cues at compile time. This eliminates a common source
    /// of runtime errors and builds procedural knowledge about correct usage.
    ///
    /// # Panics
    ///
    /// Panics if typestate guarantees are violated and required fields are
    /// missing. Proper use of the builder API keeps this path unreachable in
    /// production code.
    #[must_use]
    pub fn build(self) -> Cue {
        let Self {
            id,
            cue_type,
            cue_confidence,
            result_threshold,
            max_results,
            _state: _,
        } = self;

        let (Some(id), Some(cue_type)) = (id, cue_type) else {
            unreachable!("typestate guarantees cue ID and type are set");
        };

        Cue {
            id,
            cue_type,
            cue_confidence,
            result_threshold,
            max_results,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    type TestResult<T = ()> = Result<T, String>;

    const FLOAT_TOLERANCE: f32 = 1e-5;

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= FLOAT_TOLERANCE,
            "expected {expected}, got {actual}"
        );
    }

    fn ensure(condition: bool, message: impl Into<String>) -> TestResult {
        if condition {
            Ok(())
        } else {
            Err(message.into())
        }
    }

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    fn ensure_f32_slice_close(actual: &[f32], expected: &[f32], context: &str) -> TestResult {
        if actual.len() != expected.len() {
            return Err(format!(
                "{context}: length mismatch (expected {}, got {})",
                expected.len(),
                actual.len()
            ));
        }
        for (idx, (&lhs, &rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            if (lhs - rhs).abs() > FLOAT_TOLERANCE {
                return Err(format!("{context}: index {idx} expected {rhs}, got {lhs}"));
            }
        }
        Ok(())
    }

    #[test]
    fn test_memory_creation_and_activation() {
        let embedding = [0.1; 768];
        let memory = Memory::new("test_memory".to_string(), embedding, Confidence::HIGH);

        assert_eq!(memory.id, "test_memory");
        assert_f32_close(memory.activation(), 0.0);
        assert!(memory.seems_reliable());
        assert!(memory.is_accurate());
    }

    #[test]
    fn test_memory_activation_updates() {
        let embedding = [0.1; 768];
        let memory = Memory::new("test_memory".to_string(), embedding, Confidence::HIGH);

        memory.set_activation(0.5);
        assert_f32_close(memory.activation(), 0.5);

        memory.add_activation(0.3);
        assert_f32_close(memory.activation(), 0.8);

        // Test clamping
        memory.add_activation(0.5);
        assert_f32_close(memory.activation(), 1.0);
    }

    #[test]
    fn test_memory_forgetting_decay() {
        let embedding = [0.1; 768];
        let mut memory = Memory::new("test_memory".to_string(), embedding, Confidence::HIGH);

        let initial_confidence = memory.confidence.raw();
        memory.apply_forgetting_decay(24.0); // 24 hours

        // Confidence should have decayed
        assert!(memory.confidence.raw() < initial_confidence);
    }

    #[test]
    fn test_episode_creation_and_recall() {
        let embedding = [0.2; 768];
        let when = Utc::now();
        let mut episode = Episode::new(
            "test_episode".to_string(),
            when,
            "Something happened".to_string(),
            embedding,
            Confidence::HIGH,
        );

        assert!(episode.is_vivid());
        assert!(episode.feels_authentic());
        assert_eq!(episode.recall_count, 0);

        episode.record_recall();
        assert_eq!(episode.recall_count, 1);
    }

    #[test]
    fn test_cue_creation_and_confidence() {
        let embedding = [0.3; 768];
        let cue = Cue::embedding("test_cue".to_string(), embedding, Confidence::MEDIUM);

        assert!(cue.seems_effective());
        assert!(cue.is_reliable());

        let adaptive_threshold = cue.adaptive_threshold();
        // Should be between original threshold and cue confidence
        assert!(adaptive_threshold.raw() >= Confidence::MEDIUM.raw());
    }

    #[test]
    fn test_memory_builder_pattern() {
        let embedding = [0.4; 768];
        let memory = MemoryBuilder::new()
            .id("built_memory".to_string())
            .embedding(embedding)
            .confidence(Confidence::MEDIUM)
            .content("Built with builder".to_string())
            .decay_rate(0.05)
            .build();

        assert_eq!(memory.id, "built_memory");
        assert_f32_close(memory.decay_rate, 0.05);
        assert_eq!(memory.content, Some("Built with builder".to_string()));
    }

    #[test]
    fn test_episode_builder_pattern() {
        let embedding = [0.5; 768];
        let when = Utc::now();
        let episode = EpisodeBuilder::new()
            .id("built_episode".to_string())
            .when(when)
            .what("Built episode event".to_string())
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .where_location("Test location".to_string())
            .who(vec!["Alice".to_string(), "Bob".to_string()])
            .build();

        assert_eq!(episode.id, "built_episode");
        assert_eq!(episode.where_location, Some("Test location".to_string()));
        assert_eq!(
            episode.who,
            Some(vec!["Alice".to_string(), "Bob".to_string()])
        );
    }

    #[test]
    fn test_confidence_propagation_semantics() {
        let embedding = [0.6; 768];
        let mut memory = Memory::new(
            "confidence_test".to_string(),
            embedding,
            Confidence::exact(0.8),
        );

        // Test base rate updating
        memory.update_confidence_with_base_rate(Confidence::exact(0.3));
        // Should be lower due to low base rate
        assert!(memory.confidence.raw() < 0.8);

        // Test overconfidence calibration
        let high_conf_memory = Memory::new(
            "overconfident".to_string(),
            embedding,
            Confidence::exact(0.95),
        );
        let mut calibrated = high_conf_memory;
        calibrated.calibrate_confidence();
        // Should be reduced due to overconfidence correction
        assert!(calibrated.confidence.raw() < 0.95);
    }

    #[test]
    fn test_episode_forgetting_patterns() {
        let embedding = [0.7; 768];
        let when = Utc::now();
        let mut episode = Episode::new(
            "forgetting_test".to_string(),
            when,
            "Memory that will fade".to_string(),
            embedding,
            Confidence::HIGH,
        );

        let initial_encoding = episode.encoding_confidence.raw();
        let initial_vividness = episode.vividness_confidence.raw();

        episode.apply_episode_forgetting(48.0); // 48 hours

        // All confidence measures should have decayed
        assert!(episode.encoding_confidence.raw() < initial_encoding);
        assert!(episode.vividness_confidence.raw() < initial_vividness);

        // Vividness should decay faster than encoding
        let encoding_decay_ratio = episode.encoding_confidence.raw() / initial_encoding;
        let vividness_decay_ratio = episode.vividness_confidence.raw() / initial_vividness;
        assert!(vividness_decay_ratio < encoding_decay_ratio);
    }

    #[test]
    fn test_cue_builder_embedding_search() -> TestResult {
        let embedding = [0.3; 768];
        let cue = CueBuilder::new()
            .id("embedding_cue".to_string())
            .embedding_search(embedding, Confidence::MEDIUM)
            .cue_confidence(Confidence::HIGH)
            .result_threshold(Confidence::LOW)
            .max_results(5)
            .build();

        ensure_eq(&cue.id, &"embedding_cue".to_string(), "cue id")?;
        ensure_eq(&cue.cue_confidence, &Confidence::HIGH, "cue confidence")?;
        ensure_eq(&cue.result_threshold, &Confidence::LOW, "result threshold")?;
        ensure_eq(&cue.max_results, &5, "max results")?;

        match cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                ensure_f32_slice_close(&vector, &embedding, "embedding vector")?;
                ensure_eq(&threshold, &Confidence::MEDIUM, "embedding threshold")
            }
            other => Err(format!("expected embedding cue type, got {other:?}")),
        }
    }

    #[test]
    fn test_cue_builder_context_search() -> TestResult {
        let start = Utc::now();
        let end = start + chrono::Duration::hours(1);
        let cue = CueBuilder::new()
            .id("context_cue".to_string())
            .context_search(
                Some((start, end)),
                Some("Test Location".to_string()),
                Confidence::HIGH,
            )
            .build();

        ensure_eq(&cue.id, &"context_cue".to_string(), "cue id")?;

        match cue.cue_type {
            CueType::Context {
                time_range,
                location,
                confidence_threshold,
            } => {
                ensure(
                    time_range.is_some(),
                    "context cue should include time range",
                )?;
                ensure_eq(
                    &location.as_deref(),
                    &Some("Test Location"),
                    "context location",
                )?;
                ensure_eq(
                    &confidence_threshold,
                    &Confidence::HIGH,
                    "context confidence threshold",
                )
            }
            other => Err(format!("expected context cue type, got {other:?}")),
        }
    }

    #[test]
    fn test_cue_builder_temporal_search() -> TestResult {
        let pattern = TemporalPattern::Recent(chrono::Duration::hours(24));
        let cue = CueBuilder::new()
            .id("temporal_cue".to_string())
            .temporal_search(pattern, Confidence::MEDIUM)
            .build();

        match cue.cue_type {
            CueType::Temporal {
                pattern: stored_pattern,
                confidence_threshold,
            } => {
                ensure(
                    matches!(stored_pattern, TemporalPattern::Recent(_)),
                    "temporal cue should store recent pattern",
                )?;
                ensure_eq(
                    &confidence_threshold,
                    &Confidence::MEDIUM,
                    "temporal confidence threshold",
                )
            }
            other => Err(format!("expected temporal cue type, got {other:?}")),
        }
    }

    #[test]
    fn test_cue_builder_semantic_search() -> TestResult {
        let cue = CueBuilder::new()
            .id("semantic_cue".to_string())
            .semantic_search("search text".to_string(), Confidence::LOW)
            .build();

        match cue.cue_type {
            CueType::Semantic {
                content,
                fuzzy_threshold,
            } => {
                ensure_eq(&content, &"search text".to_string(), "semantic content")?;
                ensure_eq(
                    &fuzzy_threshold,
                    &Confidence::LOW,
                    "semantic fuzzy threshold",
                )
            }
            other => Err(format!("expected semantic cue type, got {other:?}")),
        }
    }

    #[test]
    fn test_cue_builder_defaults() {
        let embedding = [0.5; 768];
        let cue = CueBuilder::new()
            .id("default_cue".to_string())
            .embedding_search(embedding, Confidence::MEDIUM)
            .build();

        // Test default values
        assert_eq!(cue.cue_confidence, Confidence::MEDIUM);
        assert_eq!(cue.result_threshold, Confidence::LOW);
        assert_eq!(cue.max_results, 10);
    }

    #[test]
    fn test_cue_builder_max_results_clamping() {
        let embedding = [0.5; 768];
        let cue = CueBuilder::new()
            .id("clamp_test".to_string())
            .embedding_search(embedding, Confidence::MEDIUM)
            .max_results(2000) // Above limit
            .build();

        assert_eq!(cue.max_results, 1000); // Should be clamped to max

        let cue2 = CueBuilder::new()
            .id("clamp_test2".to_string())
            .embedding_search(embedding, Confidence::MEDIUM)
            .max_results(0) // Below limit
            .build();

        assert_eq!(cue2.max_results, 1); // Should be clamped to min
    }

    /// # Compile-Time Safety Demonstration
    ///
    /// The following examples show code that **will not compile**, demonstrating
    /// the typestate pattern's effectiveness at preventing invalid construction.
    ///
    /// These examples are in documentation to show what the typestate prevents:
    ///
    /// ```compile_fail
    /// // Cannot build without ID - this fails at compile time
    /// let bad_cue = CueBuilder::new()
    ///     .embedding_search([0.1; 768], Confidence::HIGH)
    ///     .build(); // ERROR: method `build` not found for this value
    /// ```
    ///
    /// ```compile_fail  
    /// // Cannot build without cue type - this fails at compile time
    /// let bad_cue = CueBuilder::new()
    ///     .id("test".to_string())
    ///     .build(); // ERROR: method `build` not found for this value
    /// ```
    ///
    /// ```compile_fail
    /// // Cannot build Memory without embedding - this fails at compile time
    /// let bad_memory = MemoryBuilder::new()
    ///     .id("test".to_string())
    ///     .confidence(Confidence::HIGH)
    ///     .build(); // ERROR: method `build` not found for this value
    /// ```
    ///
    /// ```compile_fail
    /// // Cannot build Episode without timestamp - this fails at compile time
    /// let bad_episode = EpisodeBuilder::new()
    ///     .id("test".to_string())
    ///     .what("Something happened".to_string())
    ///     .build(); // ERROR: method `build` not found for this value
    /// ```
    ///
    /// These compile failures build procedural knowledge: developers quickly
    /// learn the required construction sequence and internalize the pattern.
    #[test]
    fn test_typestate_compile_time_safety_documentation() {
        // This test exists to hold the documentation examples above.
        // The real "tests" are the compile_fail examples that demonstrate
        // what the typestate pattern prevents.

        // Valid construction that should always work:
        let _memory = MemoryBuilder::new()
            .id("valid".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        let _episode = EpisodeBuilder::new()
            .id("valid".to_string())
            .when(Utc::now())
            .what("Valid episode".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        let _cue = CueBuilder::new()
            .id("valid".to_string())
            .embedding_search([0.1; 768], Confidence::MEDIUM)
            .build();
    }

    #[test]
    fn test_cue_types_and_patterns() -> TestResult {
        let embedding = [0.8; 768];

        // Test embedding cue
        let embedding_cue = Cue::embedding("embed_cue".to_string(), embedding, Confidence::MEDIUM);

        if let CueType::Embedding { vector, threshold } = &embedding_cue.cue_type {
            ensure_f32_slice_close(vector, &embedding, "embedding cue vector")?;
            ensure_eq(threshold, &Confidence::MEDIUM, "embedding cue threshold")?;
        } else {
            return Err("expected embedding cue type".to_string());
        }

        // Test context cue
        let start_time = Utc::now();
        let end_time = start_time + chrono::Duration::hours(1);
        let context_cue = Cue::context(
            "context_cue".to_string(),
            Some((start_time, end_time)),
            Some("Test location".to_string()),
            Confidence::LOW,
        );

        if let CueType::Context {
            time_range,
            location,
            confidence_threshold,
        } = &context_cue.cue_type
        {
            ensure(time_range.is_some(), "context cue should have time range")?;
            ensure_eq(
                &location.as_deref(),
                &Some("Test location"),
                "context cue location",
            )?;
            ensure_eq(
                confidence_threshold,
                &Confidence::LOW,
                "context cue confidence",
            )?;
        } else {
            return Err("expected context cue type".to_string());
        }

        // Test semantic cue
        let semantic_cue = Cue::semantic(
            "semantic_cue".to_string(),
            "search content".to_string(),
            Confidence::HIGH,
        );

        if let CueType::Semantic {
            content,
            fuzzy_threshold,
        } = &semantic_cue.cue_type
        {
            ensure_eq(content, &"search content".to_string(), "semantic content")?;
            ensure_eq(
                fuzzy_threshold,
                &Confidence::HIGH,
                "semantic fuzzy threshold",
            )?;
        } else {
            return Err("expected semantic cue type".to_string());
        }

        Ok(())
    }
}
