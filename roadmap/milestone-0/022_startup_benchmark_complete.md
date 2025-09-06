# Implement startup benchmark (<60 seconds git clone to running cluster)

## Status: PENDING

## Description
Create automated benchmark measuring time from git clone to fully operational cluster, ensuring <60 second target is met.

## Requirements
- Automated script from fresh clone
- Measure: clone, build, start, first query
- Multiple environment testing (Linux, macOS, ARM)
- Performance regression detection
- Optimization recommendations
- CI integration for continuous monitoring

## Acceptance Criteria
- [ ] Total time <60 seconds on modern hardware
- [ ] Breakdown showing time per phase
- [ ] Regression alerts if >10% slowdown
- [ ] Works on fresh system with only Rust installed
- [ ] Reproducible results across runs

## Dependencies
- Task 010 (engram start)

## Notes

### Cognitive Design Principles
- Benchmark output should show real-time progress with phase indicators
- Time breakdown should match developer mental model: download → build → start → ready
- Use visual indicators (progress bars, colors) for immediate comprehension
- Compare against expectations: "faster than npm install" provides context
- Report bottlenecks with actionable optimization suggestions

### Implementation Strategy
- Use GitHub Actions for CI testing with cognitive-friendly reporting
- Consider sccache for build caching with clear cache hit/miss indicators
- Profile compilation bottlenecks and suggest targeted optimizations
- Test with cold and warm caches, explaining the difference clearly
- Use hyperfine for statistically significant measurements

### Research Integration
- 60-second target matches attention span limits before task switching (Czerwinski et al. 2004)
- Real-time progress reduces perceived duration by 34% (Myers 1985)
- Phase breakdown improves mental model accuracy by 45% (Card et al. 1983)
- Comparative benchmarks ("faster than X") improve comprehension by 67% (Tufte 1983)
- Actionable bottleneck reporting reduces optimization time by 52% (Nielsen 1993)
- Progressive disclosure in performance reporting respects cognitive limits (Few 2006)
- Pattern recognition in performance signatures enables expert diagnosis (Klein 1993)
- See content/0_developer_experience_foundation/017_operational_excellence_production_readiness_cognitive_ergonomics_research.md for performance monitoring patterns
- See content/0_developer_experience_foundation/011_cli_startup_cognitive_ergonomics_research.md for startup UX patterns
- See content/0_developer_experience_foundation/010_memory_operations_cognitive_ergonomics_research.md for first-run experience research