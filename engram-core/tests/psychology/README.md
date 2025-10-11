# Psychology Validation Tests (Milestone 12)

Tests validating Engram's cognitive operations against published psychology research.

## Purpose

Engram models biological memory systems, so its behavior should match empirical data from cognitive psychology and neuroscience. These tests validate:

- Forgetting curves match Ebbinghaus (1885) and Wixted & Ebbesen (1991)
- False memory formation matches DRM paradigm (Roediger & McDermott, 1995)
- Interference patterns match proactive/retroactive interference studies
- Semantic priming matches spreading activation research
- Consolidation patterns match memory systems neuroscience

## Running Psychology Tests

These tests simulate long-term memory dynamics and may take significant time:

```bash
# Run all psychology validation tests (long-running)
cargo test --test psychology --ignored

# Run specific validation
cargo test --test forgetting_curves --ignored

# Run quick smoke tests only (no --ignored)
cargo test --test forgetting_curves
```

## Validation Requirements (Milestone 4)

> "Compare forgetting curves against published psychology data, must match within 5% error"

All empirical validations must:
1. Reference published research with citations
2. Match empirical data within specified tolerance (typically 5%)
3. Use deterministic seeds for reproducibility
4. Document any deviations from empirical patterns

## Available Validation Tests

### `forgetting_curves.rs`
Validates temporal decay functions against Ebbinghaus forgetting curves.

**Empirical baseline:** Ebbinghaus (1885) retention rates
- 20 minutes: 60% retention
- 1 hour: 45% retention
- 1 day: 35% retention
- 6 days: 25% retention

**Tolerance:** ±5% (Milestone 4 requirement)

**Status:** Exponential decay implemented, power-law pending

### `drm_paradigm.rs` (Planned)
Validates false memory formation using Deese-Roediger-McDermott paradigm.

**Empirical baseline:** Roediger & McDermott (1995)
- Present semantically related word lists (e.g., "bed, rest, awake, tired")
- Critical lure word ("sleep") not presented but falsely recalled
- False recall rate: ~55% in original study

### `interference_patterns.rs` (Planned)
Validates proactive and retroactive interference.

**Empirical baseline:** McGeoch & McDonald (1931), Barnes & Underwood (1959)
- Proactive interference: Old memories interfere with new learning
- Retroactive interference: New learning interferes with old memories
- Interference increases with similarity and decreases with time

### `semantic_priming.rs` (Planned)
Validates spreading activation matches semantic priming effects.

**Empirical baseline:** Meyer & Schvaneveldt (1971), Neely (1977)
- Related word pairs recognized faster (e.g., "doctor" → "nurse")
- Priming effect: ~50ms faster recognition
- Activation spreads along semantic associations

## Adding New Validation Tests

When implementing psychology validations:

1. **Cite the research:**
   ```rust
   /// Validates X against Y (Author, Year)
   ///
   /// Reference: Author, A. (Year). Title. Journal, vol(issue), pages.
   ```

2. **Document empirical baseline:**
   ```rust
   const EMPIRICAL_BASELINE: [(Condition, f32); N] = [
       (condition1, 0.60),  // 60% in original study
       (condition2, 0.45),  // 45% in original study
   ];
   ```

3. **Use appropriate tolerance:**
   ```rust
   const TOLERANCE: f32 = 0.05; // 5% from milestone requirement
   assert!(error <= TOLERANCE);
   ```

4. **Mark as long-running:**
   ```rust
   #[tokio::test]
   #[ignore] // Long-running test - simulates hours/days
   async fn test_name() { }
   ```

5. **Include quick smoke test:**
   ```rust
   #[tokio::test]
   async fn test_name_smoke() {
       // Quick validation of basic behavior
   }
   ```

## Milestone 12 Requirements

> "Validation: Replicate classic psychology experiments (DRM paradigm, interference patterns). Verify metrics overhead using production-like workloads with/without instrumentation."

Checklist for Milestone 12 completion:
- [ ] DRM paradigm false memory replication
- [ ] Proactive interference validation
- [ ] Retroactive interference validation
- [ ] Semantic priming validation
- [ ] Forgetting curve validation (Milestone 4)
- [ ] All validations within 5% error tolerance
- [ ] Documentation of any deviations from empirical patterns

## References

Key papers for validation:

- **Ebbinghaus, H.** (1885). Memory: A Contribution to Experimental Psychology.
- **Roediger, H. L., & McDermott, K. B.** (1995). Creating false memories: Remembering words not presented in lists. Journal of Experimental Psychology: Learning, Memory, and Cognition, 21(4), 803-814.
- **Wixted, J. T., & Ebbesen, E. B.** (1991). On the form of forgetting. Psychological Science, 2(6), 409-415.
- **Meyer, D. E., & Schvaneveldt, R. W.** (1971). Facilitation in recognizing pairs of words: Evidence of a dependence between retrieval operations. Journal of Experimental Psychology, 90(2), 227-234.
- **McGeoch, J. A., & McDonald, W. T.** (1931). Meaningful relation and retroactive inhibition. American Journal of Psychology, 43(4), 579-588.
