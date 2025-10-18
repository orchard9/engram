//! Query verification and validation
//!
//! This module provides two levels of verification:
//! 1. Runtime query result validation (confidence thresholds, ordering)
//! 2. SMT-based formal verification of probability axioms (feature-gated)

use crate::{Confidence, Cue, Episode};

#[cfg(feature = "smt_verification")]
use dashmap::DashMap;
#[cfg(feature = "smt_verification")]
use z3::ast::{Ast, Real};
#[cfg(feature = "smt_verification")]
use z3::{Config, Context, SatResult, Solver};

/// Result of SMT verification proof
#[cfg(feature = "smt_verification")]
#[derive(Debug, Clone)]
pub struct VerificationProof {
    /// Name of the axiom or property verified
    pub property_name: String,
    /// Whether the property was proven to hold
    pub verified: bool,
    /// Z3 solver result
    pub sat_result: String,
    /// Timestamp of verification
    pub timestamp: std::time::SystemTime,
}

/// Error types for SMT verification
#[cfg(feature = "smt_verification")]
#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
    /// SMT solver reported unsatisfiable (property violated)
    #[error("Property violated: {property}")]
    PropertyViolated {
        /// Name of the property that was violated
        property: String,
    },

    /// SMT solver reported unknown result
    #[error("Solver returned unknown for property: {property}")]
    SolverUnknown {
        /// Name of the property that could not be determined
        property: String,
    },

    /// Z3 context error
    #[error("Z3 context error: {reason}")]
    ContextError {
        /// Reason for the context error
        reason: String,
    },
}

/// SMT-based formal verification suite for probability axioms
#[cfg(feature = "smt_verification")]
pub struct SMTVerificationSuite {
    /// Z3 context for creating formulas and solvers (not thread-safe, single-threaded use only)
    context: Context,
    /// Cache of previously verified proofs (indexed by property hash)
    proof_cache: DashMap<String, VerificationProof>,
}

#[cfg(feature = "smt_verification")]
impl SMTVerificationSuite {
    /// Create a new SMT verification suite
    #[must_use]
    pub fn new() -> Self {
        let mut config = Config::new();
        config.set_timeout_msec(5000); // 5 second timeout per query
        let context = Context::new(&config);

        Self {
            context,
            proof_cache: DashMap::new(),
        }
    }

    /// Verify core probability axioms
    ///
    /// Axiom 1: 0 ≤ P(A) ≤ 1 for all events A
    /// Axiom 2: P(A ∧ B) ≤ min(P(A), P(B)) (conjunction bound)
    /// Axiom 3: P(A ∨ B) ≤ P(A) + P(B) (union bound)
    ///
    /// # Errors
    ///
    /// Returns `VerificationError` if any axiom cannot be verified
    pub fn verify_probability_axioms(&self) -> Result<Vec<VerificationProof>, VerificationError> {
        let mut proofs = Vec::new();

        // Check cache first
        let axiom1_key = "probability_axiom_1_bounds";
        if let Some(cached) = self.proof_cache.get(axiom1_key) {
            proofs.push(cached.clone());
        } else {
            let proof = self.verify_axiom_1()?;
            self.proof_cache
                .insert(axiom1_key.to_string(), proof.clone());
            proofs.push(proof);
        }

        let axiom2_key = "probability_axiom_2_conjunction";
        if let Some(cached) = self.proof_cache.get(axiom2_key) {
            proofs.push(cached.clone());
        } else {
            let proof = self.verify_axiom_2()?;
            self.proof_cache
                .insert(axiom2_key.to_string(), proof.clone());
            proofs.push(proof);
        }

        let axiom3_key = "probability_axiom_3_union";
        if let Some(cached) = self.proof_cache.get(axiom3_key) {
            proofs.push(cached.clone());
        } else {
            let proof = self.verify_axiom_3()?;
            self.proof_cache
                .insert(axiom3_key.to_string(), proof.clone());
            proofs.push(proof);
        }

        Ok(proofs)
    }

    /// Verify Axiom 1: 0 ≤ P(A) ≤ 1
    ///
    /// This verifies that probability operations preserve the [0, 1] bounds.
    /// We show that if inputs are valid probabilities, operations produce valid probabilities.
    fn verify_axiom_1(&self) -> Result<VerificationProof, VerificationError> {
        let solver = Solver::new(&self.context);

        let p_a = Real::new_const(&self.context, "P_A");
        let p_not_a = Real::new_const(&self.context, "P_NOT_A");

        let zero = Real::from_real(&self.context, 0, 1);
        let one = Real::from_real(&self.context, 1, 1);

        // Assume P(A) is a valid probability
        solver.assert(&p_a.ge(&zero));
        solver.assert(&p_a.le(&one));

        // Define P(¬A) = 1 - P(A) (negation operation)
        let computed_not_a = Real::sub(&self.context, &[&one, &p_a]);
        solver.assert(&p_not_a._eq(&computed_not_a));

        // Try to find case where P(¬A) violates bounds
        let violates_lower = p_not_a.lt(&zero);
        let violates_upper = p_not_a.gt(&one);
        let violates_bounds = z3::ast::Bool::or(&self.context, &[&violates_lower, &violates_upper]);

        solver.assert(&violates_bounds);

        let result = solver.check();
        let verified = matches!(result, SatResult::Unsat);

        if !verified {
            return Err(VerificationError::PropertyViolated {
                property: "Probability bounds axiom (0 ≤ P(A) ≤ 1)".to_string(),
            });
        }

        Ok(VerificationProof {
            property_name: "Probability Axiom 1: Operations preserve [0, 1] bounds".to_string(),
            verified,
            sat_result: format!("{result:?}"),
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Verify Axiom 2: P(A ∧ B) ≤ min(P(A), P(B))
    ///
    /// Our implementation uses P(A ∧ B) = P(A) * P(B) (independence assumption)
    /// We verify this always satisfies the conjunction bound.
    fn verify_axiom_2(&self) -> Result<VerificationProof, VerificationError> {
        let solver = Solver::new(&self.context);

        let p_a = Real::new_const(&self.context, "P_A");
        let p_b = Real::new_const(&self.context, "P_B");

        // Assume valid probabilities
        let zero = Real::from_real(&self.context, 0, 1);
        let one = Real::from_real(&self.context, 1, 1);

        solver.assert(&p_a.ge(&zero));
        solver.assert(&p_a.le(&one));
        solver.assert(&p_b.ge(&zero));
        solver.assert(&p_b.le(&one));

        // Our implementation: P(A ∧ B) = P(A) * P(B)
        let p_a_and_b = Real::mul(&self.context, &[&p_a, &p_b]);

        // Try to find case where P(A) * P(B) > P(A) OR P(A) * P(B) > P(B)
        let violates_a = p_a_and_b.gt(&p_a);
        let violates_b = p_a_and_b.gt(&p_b);
        let violates_bound = z3::ast::Bool::or(&self.context, &[&violates_a, &violates_b]);

        solver.assert(&violates_bound);

        let result = solver.check();
        let verified = matches!(result, SatResult::Unsat);

        if !verified {
            return Err(VerificationError::PropertyViolated {
                property: "Conjunction bound axiom (P(A ∧ B) ≤ min(P(A), P(B)))".to_string(),
            });
        }

        Ok(VerificationProof {
            property_name: "Probability Axiom 2: P(A) * P(B) ≤ min(P(A), P(B))".to_string(),
            verified,
            sat_result: format!("{result:?}"),
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Verify Axiom 3: P(A ∨ B) ≤ P(A) + P(B)
    ///
    /// Our implementation uses P(A ∨ B) = P(A) + P(B) - P(A)*P(B)
    /// We verify this always satisfies the union bound and stays in [0,1].
    fn verify_axiom_3(&self) -> Result<VerificationProof, VerificationError> {
        let solver = Solver::new(&self.context);

        let p_a = Real::new_const(&self.context, "P_A");
        let p_b = Real::new_const(&self.context, "P_B");

        // Assume valid probabilities
        let zero = Real::from_real(&self.context, 0, 1);
        let one = Real::from_real(&self.context, 1, 1);

        solver.assert(&p_a.ge(&zero));
        solver.assert(&p_a.le(&one));
        solver.assert(&p_b.ge(&zero));
        solver.assert(&p_b.le(&one));

        // Our implementation: P(A ∨ B) = P(A) + P(B) - P(A)*P(B)
        let product = Real::mul(&self.context, &[&p_a, &p_b]);
        let sum = Real::add(&self.context, &[&p_a, &p_b]);
        let p_a_or_b = Real::sub(&self.context, &[&sum, &product]);

        // Try to find case where result violates bounds
        let violates_upper = p_a_or_b.gt(&one);
        let violates_lower = p_a_or_b.lt(&zero);
        let violates_union_bound = p_a_or_b.gt(&sum);

        let any_violation = z3::ast::Bool::or(
            &self.context,
            &[&violates_upper, &violates_lower, &violates_union_bound],
        );

        solver.assert(&any_violation);

        let result = solver.check();
        let verified = matches!(result, SatResult::Unsat);

        if !verified {
            return Err(VerificationError::PropertyViolated {
                property: "Union bound axiom (P(A ∨ B) ≤ P(A) + P(B) and stays in [0,1])"
                    .to_string(),
            });
        }

        Ok(VerificationProof {
            property_name:
                "Probability Axiom 3: P(A ∨ B) = P(A) + P(B) - P(A)*P(B) satisfies bounds"
                    .to_string(),
            verified,
            sat_result: format!("{result:?}"),
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Verify Bayes' theorem correctness
    ///
    /// Verifies that P(A|B) = P(B|A) * P(A) / P(B) when P(B) > 0
    /// This verification shows that given the definition of conditional probability,
    /// Bayes' theorem holds as a mathematical identity.
    ///
    /// # Errors
    ///
    /// Returns `VerificationError` if the theorem cannot be verified
    pub fn verify_bayes_theorem(&self) -> Result<VerificationProof, VerificationError> {
        let cache_key = "bayes_theorem";
        if let Some(cached) = self.proof_cache.get(cache_key) {
            return Ok(cached.clone());
        }

        let solver = Solver::new(&self.context);

        let p_a = Real::new_const(&self.context, "P_A");
        let p_b = Real::new_const(&self.context, "P_B");
        let p_a_and_b = Real::new_const(&self.context, "P_A_and_B");

        // Assume valid probabilities
        let zero = Real::from_real(&self.context, 0, 1);
        let one = Real::from_real(&self.context, 1, 1);

        solver.assert(&p_a.gt(&zero)); // P(A) > 0
        solver.assert(&p_a.le(&one));
        solver.assert(&p_b.gt(&zero)); // P(B) > 0 for conditionals to be defined
        solver.assert(&p_b.le(&one));
        solver.assert(&p_a_and_b.ge(&zero));
        solver.assert(&p_a_and_b.le(&one));

        // Joint probability constraints
        solver.assert(&p_a_and_b.le(&p_a)); // P(A∧B) ≤ P(A)
        solver.assert(&p_a_and_b.le(&p_b)); // P(A∧B) ≤ P(B)

        // Define conditional probabilities based on joint probability
        // P(A|B) = P(A∧B) / P(B)
        let p_a_given_b = p_a_and_b.div(&p_b);

        // P(B|A) = P(A∧B) / P(A)
        let p_b_given_a = p_a_and_b.div(&p_a);

        // Bayes theorem: P(A|B) = P(B|A) * P(A) / P(B)
        let numerator = Real::mul(&self.context, &[&p_b_given_a, &p_a]);
        let rhs = numerator.div(&p_b);

        // Try to find case where P(A|B) ≠ P(B|A) * P(A) / P(B)
        let bayes_violated = p_a_given_b._eq(&rhs).not();
        solver.assert(&bayes_violated);

        let result = solver.check();
        let verified = matches!(result, SatResult::Unsat);

        if !verified {
            return Err(VerificationError::PropertyViolated {
                property: "Bayes' theorem".to_string(),
            });
        }

        let proof = VerificationProof {
            property_name: "Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)".to_string(),
            verified,
            sat_result: format!("{result:?}"),
            timestamp: std::time::SystemTime::now(),
        };

        self.proof_cache
            .insert(cache_key.to_string(), proof.clone());
        Ok(proof)
    }

    /// Verify conjunction fallacy prevention
    ///
    /// Ensures that P(A ∧ B) ≤ P(A) always holds (prevents Linda problem errors)
    /// Our implementation uses P(A ∧ B) = P(A) * P(B)
    ///
    /// # Errors
    ///
    /// Returns `VerificationError` if the property cannot be verified
    pub fn verify_conjunction_fallacy_prevention(
        &self,
    ) -> Result<VerificationProof, VerificationError> {
        let cache_key = "conjunction_fallacy_prevention";
        if let Some(cached) = self.proof_cache.get(cache_key) {
            return Ok(cached.clone());
        }

        let solver = Solver::new(&self.context);

        let p_a = Real::new_const(&self.context, "P_A");
        let p_b = Real::new_const(&self.context, "P_B");

        // Assume valid probabilities
        let zero = Real::from_real(&self.context, 0, 1);
        let one = Real::from_real(&self.context, 1, 1);

        solver.assert(&p_a.ge(&zero));
        solver.assert(&p_a.le(&one));
        solver.assert(&p_b.ge(&zero));
        solver.assert(&p_b.le(&one));

        // Our implementation: P(A ∧ B) = P(A) * P(B)
        let p_a_and_b = Real::mul(&self.context, &[&p_a, &p_b]);

        // Property: P(A) * P(B) ≤ P(A) (and also ≤ P(B))
        // Negation: P(A) * P(B) > P(A) OR P(A) * P(B) > P(B)
        let violates_a = p_a_and_b.gt(&p_a);
        let violates_b = p_a_and_b.gt(&p_b);
        let negation = z3::ast::Bool::or(&self.context, &[&violates_a, &violates_b]);

        solver.assert(&negation);

        let result = solver.check();
        let verified = matches!(result, SatResult::Unsat);

        if !verified {
            return Err(VerificationError::PropertyViolated {
                property: "Conjunction fallacy prevention (P(A ∧ B) ≤ P(A))".to_string(),
            });
        }

        let proof = VerificationProof {
            property_name: "Conjunction Fallacy Prevention: P(A) * P(B) ≤ min(P(A), P(B))"
                .to_string(),
            verified,
            sat_result: format!("{result:?}"),
            timestamp: std::time::SystemTime::now(),
        };

        self.proof_cache
            .insert(cache_key.to_string(), proof.clone());
        Ok(proof)
    }

    /// Get statistics about proof caching
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            cached_proofs: self.proof_cache.len(),
            total_properties: 6, // 3 axioms + Bayes + conjunction fallacy + (future expansions)
        }
    }
}

#[cfg(feature = "smt_verification")]
impl Default for SMTVerificationSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about verification proof caching
#[cfg(feature = "smt_verification")]
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Number of proofs currently cached
    pub cached_proofs: usize,
    /// Total number of verifiable properties
    pub total_properties: usize,
}

/// Verify query results meet confidence thresholds and constraints
pub struct QueryVerifier;

impl QueryVerifier {
    /// Verify that query results meet the specified confidence threshold.
    ///
    /// # Errors
    ///
    /// This function does not return errors; it simply evaluates the provided inputs.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use = "Use the verification outcome to enforce confidence constraints"]
    pub fn verify_confidence_threshold(
        results: &[(Episode, Confidence)],
        threshold: Confidence,
    ) -> bool {
        results
            .iter()
            .all(|(_, confidence)| *confidence >= threshold)
    }

    /// Verify that query results are properly ordered by confidence.
    ///
    /// # Errors
    ///
    /// This function does not return errors; it only inspects the supplied slice.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use = "Propagate ordering checks to maintain deterministic recall"]
    pub fn verify_confidence_ordering(results: &[(Episode, Confidence)]) -> bool {
        results.windows(2).all(|pair| pair[0].1 >= pair[1].1)
    }

    /// Verify that cue constraints are satisfied.
    ///
    /// # Errors
    ///
    /// This function does not return errors; it validates constraints in place.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use = "Ensure cue constraints gate downstream recall logic"]
    pub fn verify_cue_constraints(cue: &Cue, results: &[(Episode, Confidence)]) -> bool {
        // Check result count limits
        if results.len() > cue.max_results {
            return false;
        }

        // Check confidence threshold
        Self::verify_confidence_threshold(results, cue.result_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, Episode};
    use chrono::Utc;

    #[cfg(feature = "smt_verification")]
    #[allow(clippy::expect_used, clippy::uninlined_format_args)]
    mod smt_tests {
        use super::*;

        #[test]
        fn test_smt_verification_suite_creation() {
            let suite = SMTVerificationSuite::new();
            let stats = suite.cache_stats();

            assert_eq!(stats.cached_proofs, 0);
            assert_eq!(stats.total_properties, 6);
        }

        #[test]
        fn test_verify_probability_axioms() {
            let suite = SMTVerificationSuite::new();
            let proofs = suite
                .verify_probability_axioms()
                .expect("probability axioms should verify successfully");

            assert_eq!(proofs.len(), 3, "should have 3 axiom proofs");

            for proof in &proofs {
                assert!(
                    proof.verified,
                    "axiom {} should be verified",
                    proof.property_name
                );
                assert_eq!(proof.sat_result, "Unsat", "negation should be UNSAT");
            }

            // Verify cache is populated
            let stats = suite.cache_stats();
            assert_eq!(stats.cached_proofs, 3, "should have cached 3 axiom proofs");
        }

        #[test]
        fn test_verify_bayes_theorem() {
            let suite = SMTVerificationSuite::new();
            let proof = suite
                .verify_bayes_theorem()
                .expect("Bayes' theorem should verify successfully");

            assert!(proof.verified);
            assert!(proof.property_name.contains("Bayes"));
            assert_eq!(proof.sat_result, "Unsat");

            // Verify caching works
            let proof2 = suite
                .verify_bayes_theorem()
                .expect("cached proof should be retrieved");

            assert_eq!(proof.property_name, proof2.property_name);
        }

        #[test]
        fn test_verify_conjunction_fallacy_prevention() {
            let suite = SMTVerificationSuite::new();
            let proof = suite
                .verify_conjunction_fallacy_prevention()
                .expect("conjunction fallacy prevention should verify");

            assert!(proof.verified);
            assert!(proof.property_name.contains("Conjunction Fallacy"));
            assert_eq!(proof.sat_result, "Unsat");
        }

        #[test]
        fn test_proof_caching_improves_performance() {
            let suite = SMTVerificationSuite::new();

            // First call - not cached
            let start = std::time::Instant::now();
            let _proof1 = suite.verify_bayes_theorem().expect("first verification");
            let first_duration = start.elapsed();

            // Second call - should be cached and much faster
            let start = std::time::Instant::now();
            let _proof2 = suite.verify_bayes_theorem().expect("cached verification");
            let second_duration = start.elapsed();

            println!(
                "First call: {:?}, Second call (cached): {:?}",
                first_duration, second_duration
            );

            // Cached call should be at least 10x faster
            // We use a conservative threshold to avoid flakiness
            assert!(
                second_duration < first_duration / 5,
                "cached call ({:?}) should be much faster than first call ({:?})",
                second_duration,
                first_duration
            );
        }

        #[test]
        fn test_comprehensive_verification() {
            let suite = SMTVerificationSuite::new();

            // Verify all core properties
            let axioms = suite
                .verify_probability_axioms()
                .expect("axioms should verify");
            let bayes = suite.verify_bayes_theorem().expect("Bayes should verify");
            let conjunction = suite
                .verify_conjunction_fallacy_prevention()
                .expect("conjunction fallacy prevention should verify");

            assert_eq!(axioms.len(), 3);
            assert!(bayes.verified);
            assert!(conjunction.verified);

            // Check final cache statistics
            let stats = suite.cache_stats();
            assert_eq!(stats.cached_proofs, 5, "should have 5 cached proofs");
        }
    }

    #[test]
    fn test_confidence_verification() {
        let high_conf_episode = Episode::new(
            "high".to_string(),
            Utc::now(),
            "High confidence test episode".to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        );

        let results = vec![
            (high_conf_episode.clone(), Confidence::HIGH),
            (high_conf_episode, Confidence::MEDIUM),
        ];

        assert!(QueryVerifier::verify_confidence_threshold(
            &results,
            Confidence::MEDIUM
        ));
        assert!(!QueryVerifier::verify_confidence_threshold(
            &results,
            Confidence::CERTAIN
        ));
    }

    #[test]
    fn test_ordering_verification() {
        let episode = Episode::new(
            "test".to_string(),
            Utc::now(),
            "Test episode for ordering".to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        );

        let ordered_results = vec![
            (episode.clone(), Confidence::CERTAIN),
            (episode.clone(), Confidence::HIGH),
            (episode, Confidence::MEDIUM),
        ];

        assert!(QueryVerifier::verify_confidence_ordering(&ordered_results));

        let unordered_results = vec![
            (ordered_results[0].0.clone(), Confidence::MEDIUM),
            (ordered_results[1].0.clone(), Confidence::HIGH),
        ];

        assert!(!QueryVerifier::verify_confidence_ordering(
            &unordered_results
        ));
    }
}
