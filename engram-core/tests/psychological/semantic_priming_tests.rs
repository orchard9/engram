use std::thread;
use std::time::Duration;

use engram_core::cognitive::priming::SemanticPrimingEngine;

use super::statistical_analysis::{mean, welch_t_test};
use super::test_datasets::{semantic_pair_embeddings, semantic_priming_pairs};

const BASELINE_RT_MS: f32 = 620.0;
const SOA_WINDOWS: [u64; 4] = [100, 250, 400, 700];

pub(crate) fn run_neely_1977_semantic_priming_validation() {
    let pairs = semantic_priming_pairs();

    for &soa in &SOA_WINDOWS {
        let mut related_rts = Vec::new();
        let mut unrelated_rts = Vec::new();

        for pair in &pairs {
            let rt = simulate_trial(pair, soa);
            if pair.is_related {
                related_rts.push(rt);
            } else {
                unrelated_rts.push(rt);
            }
        }

        let related_mean = mean(&related_rts);
        let unrelated_mean = mean(&unrelated_rts);
        let facilitation = unrelated_mean - related_mean;
        let reduction_pct = (facilitation / BASELINE_RT_MS) * 100.0;

        if (soa == 250) || (soa == 400) {
            assert!(
                (25.0..=90.0).contains(&facilitation),
                "SOA {soa}ms facilitation should be 25-90ms, observed {:.1}ms",
                facilitation
            );
            assert!(
                (4.0..=20.0).contains(&reduction_pct),
                "SOA {soa}ms reduction {:>.1}% out of 4-20% range",
                reduction_pct
            );
        }

        if soa == 100 {
            assert!(
                facilitation < 65.0,
                "Short SOA should show limited facilitation (<65ms), observed {:.1}ms",
                facilitation
            );
        }

        if soa == 700 {
            assert!(
                facilitation >= 10.0,
                "Long SOA should still show observable facilitation, observed {:.1}ms",
                facilitation
            );
        }

        let significance = welch_t_test(&related_rts, &unrelated_rts);
        assert!(
            significance.p_value < 0.05,
            "SOA {soa}ms priming effect should be significant, p={:.4}",
            significance.p_value
        );
        assert!(
            significance.t_stat.abs() > 3.0,
            "SOA {soa}ms effect should produce strong t-statistic, observed {:.2}",
            significance.t_stat
        );
        assert!(
            significance.degrees_of_freedom > 2.0,
            "Welch test should have >2 degrees of freedom"
        );
    }
}

fn simulate_trial(pair: &super::test_datasets::SemanticPrimingPair, soa_ms: u64) -> f32 {
    let engine = SemanticPrimingEngine::new();
    let (prime_embedding, target_embedding) = semantic_pair_embeddings(pair);

    engine.activate_priming(&pair.prime.to_string(), &prime_embedding, || {
        vec![(pair.target.to_string(), target_embedding, 1)]
    });

    thread::sleep(Duration::from_millis(soa_ms));
    let boost = engine.compute_priming_boost(pair.target);
    BASELINE_RT_MS * (1.0 - boost)
}
