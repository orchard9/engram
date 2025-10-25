# Task 013: Fix Query Executor Clippy Warnings

**Status**: Pending
**Duration**: 0.5 days
**Dependencies**: Tasks 006, 007
**Owner**: TBD
**Created**: 2025-10-25

---

## Objective

Fix clippy warnings in query executor code that block Task 011 from committing.

---

## Background

During Task 011 (Documentation and Examples), clippy warnings were discovered in the query executor code from Tasks 006 and 007. These warnings prevent the documentation task from being committed due to pre-commit hooks.

---

## Warnings to Fix

### In `engram-core/src/query/executor/query_executor.rs`:

1. **needless_pass_by_value**: Multiple query structs passed by value but not consumed
   - RecallQuery, PredictQuery, ImagineQuery, ConsolidateQuery parameters
   - Arc<SpaceHandle> parameter

2. **unused_self**: Several methods don't use `&self` and should be associated functions
   - execute_predict_query
   - execute_imagine_query
   - execute_consolidate_query
   - pattern_to_cue
   - create_query_evidence

3. **trivially_copy_pass_by_ref**: QueryExecutionError::as_str takes &self for 1-byte value

### In `engram-core/src/query/executor/recall.rs`:

1. **unwrap_used**: Multiple test cases use .unwrap()
   - test_apply_confidence_above_constraint
   - test_apply_confidence_below_constraint
   - test_apply_content_contains_constraint
   - test_apply_in_memory_space_constraint
   - test_apply_multiple_constraints

2. **redundant_clone**: Test data cloned unnecessarily in apply_constraints tests

### In `engram-core/src/query/executor/spread.rs`:

1. **float_cmp**: Strict equality checks on f32 values in tests
   - test_spread_query_defaults
   - test_spread_query_effective_values

---

## Files to Modify

- `engram-core/src/query/executor/query_executor.rs` (stub implementation)
- `engram-core/src/query/executor/recall.rs` (test code)
- `engram-core/src/query/executor/spread.rs` (test code)

---

## Acceptance Criteria

- [ ] All clippy warnings in query executor resolved
- [ ] Tests still pass after fixes
- [ ] `make quality` runs cleanly
- [ ] No functional changes to executor behavior

---

## Notes

- These warnings were introduced in Tasks 006 (RECALL) and 007 (SPREAD)
- They are blocking Task 011 from committing via pre-commit hooks
- Priority: High (blocks documentation commit)
