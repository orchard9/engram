//! Proactive and retroactive interference modeling.

/// Interference modeler for memory competition
#[derive(Debug, Clone)]
pub struct InterferenceModeler {
    /// Proactive interference strength
    pub proactive_strength: f32,
    /// Retroactive interference strength
    pub retroactive_strength: f32,
}

impl Default for InterferenceModeler {
    fn default() -> Self {
        Self {
            proactive_strength: 0.8,
            retroactive_strength: 1.2,
        }
    }
}

impl InterferenceModeler {
    /// Create new interference modeler with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute interference factor based on memory similarity
    #[must_use]
    pub fn compute_interference(&self, similarity: f32) -> f32 {
        similarity * (self.proactive_strength + self.retroactive_strength) / 2.0
    }
}
