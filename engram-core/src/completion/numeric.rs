//! Helper utilities for numeric conversions used by completion modules.

/// Convert a `usize` to `f32`, saturating when the value exceeds `u32::MAX`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub const fn usize_to_f32(value: usize) -> f32 {
    value as f32
}

/// Convert a `usize` to `f64`, saturating when the value exceeds `u64::MAX`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Round a `f32` to the nearest `usize`, saturating on overflow.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub const fn round_f32_to_usize(value: f32) -> usize {
    if !value.is_finite() {
        return 0;
    }

    let rounded = value.round().clamp(0.0, u32::MAX as f32);

    rounded as usize
}

/// Compute the number of items represented by a fractional target of a total.
#[must_use]
pub fn fraction_to_count(total: usize, fraction: f32) -> usize {
    let clamped = fraction.clamp(0.0, 1.0);
    let scaled = usize_to_f32(total) * clamped;
    round_f32_to_usize(scaled)
}

/// Compute `numerator / denominator` in `f32`, guarding against division by zero.
#[must_use]
pub fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        return 0.0;
    }

    let denom = usize_to_f32(denominator);
    if denom == 0.0 {
        0.0
    } else {
        usize_to_f32(numerator) / denom
    }
}

/// Divide an `f32` value by a `usize`, returning zero when the divisor is zero.
#[must_use]
pub fn safe_divide(value: f32, divisor: usize) -> f32 {
    if divisor == 0 {
        0.0
    } else {
        let denom = usize_to_f32(divisor);
        if denom == 0.0 { 0.0 } else { value / denom }
    }
}

/// Return `1 / value` for integer inputs, handling zero gracefully.
#[must_use]
pub fn one_over_usize(value: usize) -> f32 {
    if value == 0 {
        0.0
    } else {
        1.0 / usize_to_f32(value)
    }
}

/// Clamp an `f64` into the representable range of `f32`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]
pub const fn f64_to_f32(value: f64) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }

    let max = f32::MAX as f64;
    let min = f32::MIN as f64;
    let clamped = value.clamp(min, max);

    clamped as f32
}

/// Convert an `i64` to `f32`, saturating at the `f32` bounds.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]
pub const fn i64_to_f32(value: i64) -> f32 {
    let max = f32::MAX as f64;
    let min = f32::MIN as f64;
    let value_f64 = value as f64;
    let clamped = value_f64.clamp(min, max);

    clamped as f32
}
