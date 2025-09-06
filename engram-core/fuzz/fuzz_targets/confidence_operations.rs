#![no_main]

use libfuzzer_sys::fuzz_target;
use engram_core::Confidence;

/// Comprehensive fuzzing target for confidence operations.
/// 
/// This fuzzing harness tests all confidence operations for:
/// 1. Never panicking (reliability requirement)
/// 2. Always producing valid probabilities [0,1]
/// 3. Mathematical invariants (conjunction fallacy prevention, etc.)
/// 4. Cognitive-friendly behavior under extreme inputs
fuzz_target!(|data: &[u8]| {
    // We need at least 4 bytes to generate meaningful test data
    if data.len() < 4 {
        return;
    }
    
    // Extract test parameters from fuzz input
    let raw_value = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    
    // Test 1: Construction with arbitrary f32 values should never panic
    let conf = Confidence::exact(raw_value);
    
    // Test 2: All constructed values should be valid probabilities
    let conf_value = conf.raw();
    assert!(conf_value >= 0.0, "Confidence {} is negative", conf_value);
    assert!(conf_value <= 1.0, "Confidence {} exceeds 1.0", conf_value);
    assert!(conf_value.is_finite(), "Confidence {} is not finite", conf_value);
    
    // Test 3: Operations with self should never panic
    let _ = conf.and(conf);
    let _ = conf.or(conf);
    let _ = conf.not();
    let _ = conf.calibrate_overconfidence();
    let _ = conf.is_high();
    
    // Test 4: Create second confidence for binary operations
    if data.len() >= 8 {
        let raw_value2 = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let conf2 = Confidence::exact(raw_value2);
        
        // Test binary operations never panic and maintain invariants
        let and_result = conf.and(conf2);
        let or_result = conf.or(conf2);
        
        // Validate results are still valid probabilities
        assert!(and_result.raw() >= 0.0 && and_result.raw() <= 1.0);
        assert!(or_result.raw() >= 0.0 && or_result.raw() <= 1.0);
        
        // Test conjunction fallacy prevention: P(A ∧ B) ≤ min(P(A), P(B))
        let min_input = conf.raw().min(conf2.raw());
        assert!(and_result.raw() <= min_input + f32::EPSILON,
                "Conjunction fallacy: AND result {} > min input {}", 
                and_result.raw(), min_input);
        
        // Test OR result is at least as large as maximum input
        let max_input = conf.raw().max(conf2.raw());
        assert!(or_result.raw() >= max_input - f32::EPSILON,
                "OR result {} < max input {}", or_result.raw(), max_input);
        
        // Test weighted combination with arbitrary weights
        if data.len() >= 16 {
            let weight1 = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
            let weight2 = f32::from_le_bytes([data[12], data[13], data[14], data[15]]);
            
            let combined = conf.combine_weighted(conf2, weight1, weight2);
            assert!(combined.raw() >= 0.0 && combined.raw() <= 1.0,
                    "Weighted combination {} not in [0,1]", combined.raw());
        }
    }
    
    // Test 5: Frequency-based construction with edge cases
    if data.len() >= 12 {
        let successes = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let total = if data.len() >= 16 {
            u32::from_le_bytes([data[12], data[13], data[14], data[15]])
        } else {
            1 // Avoid division by zero in most cases
        };
        
        // This should handle all edge cases gracefully
        let freq_conf = Confidence::from_successes(successes, total);
        assert!(freq_conf.raw() >= 0.0 && freq_conf.raw() <= 1.0,
                "Frequency confidence {} not in [0,1]", freq_conf.raw());
        
        // Test that results are mathematically consistent when possible
        if total > 0 && successes <= total {
            let expected = successes as f32 / total as f32;
            assert!((freq_conf.raw() - expected).abs() < f32::EPSILON,
                    "Frequency mismatch: expected {}, got {}", expected, freq_conf.raw());
        }
    }
    
    // Test 6: Percentage construction with all possible u8 values
    if data.len() >= 13 {
        let percent = data[12];
        let percent_conf = Confidence::from_percent(percent);
        assert!(percent_conf.raw() >= 0.0 && percent_conf.raw() <= 1.0,
                "Percentage confidence {} not in [0,1]", percent_conf.raw());
        
        // Verify percentage conversion is correct
        let expected_percent = (percent.min(100) as f32) / 100.0;
        assert!((percent_conf.raw() - expected_percent).abs() < f32::EPSILON,
                "Percentage conversion error: expected {}, got {}", 
                expected_percent, percent_conf.raw());
    }
    
    // Test 7: Double negation should return to original (within floating point precision)
    let double_neg = conf.not().not();
    assert!((double_neg.raw() - conf.raw()).abs() < f32::EPSILON * 2.0,
            "Double negation failed: {} -> {}", conf.raw(), double_neg.raw());
    
    // Test 8: Chained operations should maintain validity
    let complex_result = conf
        .and(Confidence::HIGH)
        .or(Confidence::LOW)
        .not()
        .calibrate_overconfidence();
    
    assert!(complex_result.raw() >= 0.0 && complex_result.raw() <= 1.0,
            "Complex chained operation {} not in [0,1]", complex_result.raw());
    
    // Test 9: Mathematical properties under extreme values
    let extreme_and = conf.and(Confidence::NONE); // Should be 0
    assert!((extreme_and.raw() - 0.0).abs() < f32::EPSILON,
            "AND with NONE should be 0, got {}", extreme_and.raw());
    
    let extreme_or = conf.or(Confidence::CERTAIN); // Should be 1
    assert!((extreme_or.raw() - 1.0).abs() < f32::EPSILON,
            "OR with CERTAIN should be 1, got {}", extreme_or.raw());
    
    // Test 10: Calibration should always produce valid results
    let calibrated = conf.calibrate_overconfidence();
    assert!(calibrated.raw() >= 0.0 && calibrated.raw() <= 1.0,
            "Calibrated confidence {} not in [0,1]", calibrated.raw());
    
    // High confidence should generally be reduced (unless already very low)
    if conf.raw() > 0.8 {
        assert!(calibrated.raw() <= conf.raw() + f32::EPSILON,
                "High confidence {} should be reduced or unchanged, got {}", 
                conf.raw(), calibrated.raw());
    }
});