---
description: Autonomously complete a task from implementation through verification to documentation using specialized agents
---

I need to complete a task from start to finish with full quality assurance.

**Instructions**:

This command orchestrates a complete task execution workflow using specialized agents in sequence:

1. **Agent Selection**: Analyze task requirements and select the optimal specialist agent
2. **Implementation**: Execute full implementation with chosen specialist
3. **Quality Review**: Verification-testing-lead reviews for tech debt, accuracy, completeness
4. **Issue Resolution**: Fix all identified issues
5. **Final Review**: Tie up loose ends and ensure alignment with task requirements
6. **Documentation**: Update relevant docs with technical-communication-lead or documentation-validator

**Workflow States**:

- ANALYZE → Read task file, understand requirements, select best agent
- IMPLEMENT → Spawn specialist agent for full implementation
- QUALITY_CHECK → verification-testing-lead reviews code quality, tech debt, correctness
- FIX_ISSUES → Address all findings from quality review
- FINAL_REVIEW → Ensure task requirements fully met, no loose ends
- DOCUMENT → Update docs (API refs, guides, architecture docs)
- VERIFY_TESTS → Run tests and quality checks (make quality, cargo test)
- COMPLETE → Mark task complete, commit changes

**Agent Selection Logic**:

Based on task requirements, automatically select from:

- **rust-graph-engine-architect**: Graph algorithms, lock-free data structures, cache optimization, concurrent graphs
- **systems-architecture-optimizer**: Storage systems, NUMA optimization, tiered caching, performance bottlenecks
- **memory-systems-researcher**: Memory consolidation, hippocampal-neocortical systems, spreading activation, biological plausibility
- **gpu-acceleration-architect**: CUDA kernels, GPU memory optimization, parallel graph algorithms
- **cognitive-architecture-designer**: Memory dynamics, System 2 reasoning, consolidation algorithms, dream processes
- **verification-testing-lead**: Testing strategies, differential testing, fuzzing, formal verification
- **systems-product-planner**: Technical specs, roadmap planning, architectural decisions
- **utoipa-documentation-expert**: OpenAPI specs, API documentation
- **technical-communication-lead**: External docs, blog posts, developer guides
- **documentation-validator**: Doc accuracy, clarity, usability validation

**Quality Review Checklist** (verification-testing-lead):

- [ ] Code follows coding_guidelines.md (error handling, concurrency, performance)
- [ ] No unnecessary Result wrapping (return direct values when can't fail)
- [ ] Iterator methods preferred over index loops
- [ ] Use statements at module top
- [ ] Safe casting with try_into() instead of `as`
- [ ] \#\[must_use\] on getters and constructors
- [ ] Large types passed by reference
- [ ] Expensive temporaries scoped early
- [ ] No clippy warnings (run make quality)
- [ ] Tests follow 80/20 Pareto principle
- [ ] Performance regression <5% (for M17 tasks)
- [ ] No tech debt introduced

**Documentation Update Logic**:

- **API changes** → utoipa-documentation-expert updates OpenAPI specs
- **Architecture changes** → documentation-validator updates architecture docs
- **Public features** → technical-communication-lead updates user guides
- **Internal patterns** → Update coding_guidelines.md or core_packages.md

**Usage**:

Provide the task file path:
- `roadmap/milestone-15/001_security_configuration_schema_pending.md`
- Or short form if unambiguous: `015/001`

**Example Execution Flow**:

```
User: /complete-task roadmap/milestone-15/001_security_configuration_schema_pending.md

Claude:
1. [ANALYZE] Reading task file... Found: Security configuration schema
   - Primary concern: Configuration management, validation
   - Selected agent: systems-architecture-optimizer

2. [IMPLEMENT] Spawning systems-architecture-optimizer agent...
   - Implementing SecurityConfig struct
   - Adding validation logic
   - Creating default.toml and production.toml
   - Writing tests

3. [QUALITY_CHECK] Spawning verification-testing-lead agent...
   - Reviewing for tech debt: FOUND 2 issues
     * Issue 1: Unnecessary Result wrapping in load_config()
     * Issue 2: Missing #[must_use] on SecurityConfig::validate()

4. [FIX_ISSUES] Addressing quality review findings...
   - Fixed: Changed load_config() to return Config directly
   - Fixed: Added #[must_use] to validate()

5. [FINAL_REVIEW] Checking task requirements...
   - ✓ SecurityConfig struct with all fields
   - ✓ AuthMode enum (None, ApiKey)
   - ✓ Default config with backward compatibility
   - ✓ Environment variable overrides
   - ✓ Tests passing

6. [DOCUMENT] Updating documentation...
   - Updated docs/reference/configuration.md with SecurityConfig
   - Added example configs to docs/tutorials/security-quickstart.md

7. [VERIFY_TESTS] Running quality checks...
   $ make quality
   ✓ All tests passed
   ✓ Zero clippy warnings

8. [COMPLETE] Task 001 complete!
   - Renaming: 001_security_configuration_schema_pending.md → 001_security_configuration_schema_complete.md
   - Committing changes...
```

**Git Integration**:

- Implements task completely before committing
- Runs `make quality` to ensure zero warnings
- Commits with descriptive message following project style
- Does NOT push (user reviews and pushes)

**Checkpoint for Human Review**:

- After QUALITY_CHECK: Shows issues found, asks to proceed with fixes
- After FINAL_REVIEW: Shows completion summary, asks to commit
- After VERIFY_TESTS: Shows test results, clippy output

**What This Prevents**:

- Incomplete implementations that skip edge cases
- Tech debt accumulation from skipped quality review
- Missing documentation updates
- Committing code with clippy warnings
- Task marked complete before fully meeting requirements

Which task should I complete?
