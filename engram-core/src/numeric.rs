//! Numeric conversion helpers shared across Engram modules.
//!
//! These helpers provide saturated conversions between primitive numeric
//! types so that modules can avoid hand-rolling the same logic (and the
//! accompanying `allow` noise) in multiple places.

/// Convert an `f64` to `f32`, saturating to the representable range and
/// preserving NaN payloads.
pub fn saturating_f32_from_f64(value: f64) -> f32 {
    if value.is_nan() {
        return f32::NAN;
    }

    let max = f64::from(f32::MAX);
    let min = f64::from(f32::MIN);
    if value >= max {
        return f32::MAX;
    }
    if value <= min {
        return f32::MIN;
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    {
        // The clamp above guarantees the cast stays within the representable
        // f32 range, so the only loss is sub-ulp precision.
        value as f32
    }
}

/// Convert a ratio of two unsigned integers to an `f32` in the unit interval.
pub fn unit_ratio_to_f32(numerator: u64, denominator: u64) -> f32 {
    if denominator == 0 {
        return 0.0;
    }

    let numerator = u64_to_f64(numerator);
    let denominator = u64_to_f64(denominator);
    let ratio = (numerator / denominator).clamp(0.0, 1.0);
    saturating_f32_from_f64(ratio)
}

/// Convert a `u64` to `f64`, accepting the precision loss that occurs when the
/// integer exceeds the f64 mantissa.
pub const fn u64_to_f64(value: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f64
    }
}
