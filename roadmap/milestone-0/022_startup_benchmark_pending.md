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
- Use GitHub Actions for CI testing
- Consider sccache for build caching
- Profile compilation bottlenecks
- Test with cold and warm caches