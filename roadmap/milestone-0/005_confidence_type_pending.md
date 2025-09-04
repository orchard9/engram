# Implement Confidence(f32) newtype with [0,1] range invariants

## Status: PENDING

## Description
Create a Confidence newtype that enforces probability values between 0 and 1 at compile time where possible, runtime where necessary. No Option<f32> allowed for confidence values.

## Requirements
- Newtype wrapper Confidence(f32) with private constructor
- Validation ensuring values always in [0,1] range
- Arithmetic operations that preserve invariants
- Confidence interval type (f32, f32) with ordering guarantees
- Serde support with validation on deserialization
- No Option<Confidence> in public APIs

## Acceptance Criteria
- [ ] Cannot construct Confidence with value outside [0,1]
- [ ] Arithmetic operations maintain range invariants
- [ ] Confidence multiplication/combination preserves validity
- [ ] Zero-cost abstraction (no runtime overhead in release)
- [ ] Compile-time prevention of Option<f32> for confidence

## Dependencies
- Task 001 (workspace setup)

## Notes
- Consider const fn for compile-time validation
- Use debug_assert! for development checks
- Implement From traits for common conversions
- Consider NonZeroConfidence type for special cases