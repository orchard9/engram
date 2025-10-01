use std::time::Duration;

use super::storage_aware::StorageTier;

/// Manages latency budgets for storage tiers and validates access latency.
#[derive(Debug, Clone)]
pub struct LatencyBudgetManager {
    budgets: [Duration; 3],
}

impl LatencyBudgetManager {
    /// Create a manager with default budgets derived from tier configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            budgets: [
                Duration::from_micros(100),
                Duration::from_millis(1),
                Duration::from_millis(10),
            ],
        }
    }

    /// Retrieve latency budget for the specified tier.
    #[must_use]
    pub const fn budget_for(&self, tier: StorageTier) -> Duration {
        self.budgets[tier as usize]
    }

    /// Override the latency budget for the specified tier.
    pub const fn set_budget(&mut self, tier: StorageTier, budget: Duration) {
        self.budgets[tier as usize] = budget;
    }

    /// Determine whether the observed latency respects the tier budget.
    #[must_use]
    pub fn within_budget(&self, tier: StorageTier, observed: Duration) -> bool {
        observed <= self.budget_for(tier)
    }
}

impl Default for LatencyBudgetManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{LatencyBudgetManager, StorageTier};
    use std::time::Duration;

    #[test]
    fn test_default_budgets() {
        let manager = LatencyBudgetManager::new();
        assert_eq!(
            manager.budget_for(StorageTier::Hot),
            Duration::from_micros(100)
        );
        assert_eq!(
            manager.budget_for(StorageTier::Warm),
            Duration::from_millis(1)
        );
        assert_eq!(
            manager.budget_for(StorageTier::Cold),
            Duration::from_millis(10)
        );
    }

    #[test]
    fn test_within_budget() {
        let manager = LatencyBudgetManager::new();
        assert!(manager.within_budget(StorageTier::Hot, Duration::from_micros(80)));
        assert!(!manager.within_budget(StorageTier::Warm, Duration::from_millis(2)));
    }

    #[test]
    fn test_set_budget() {
        let mut manager = LatencyBudgetManager::new();
        manager.set_budget(StorageTier::Cold, Duration::from_millis(15));
        assert_eq!(
            manager.budget_for(StorageTier::Cold),
            Duration::from_millis(15)
        );
    }
}
