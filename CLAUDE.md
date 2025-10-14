Written by world class rust engineers that pride elegant efficiency and delivering a focused product.
This project uses Rust Edition 2024 for all code.

IMPORTANT: After each test run or engram execution, check diagnostics with `./scripts/engram_diagnostics.sh` and track results in `tmp/engram_diagnostics.log` (prepended).

## Roadmap

roadmap/milestone-0/: Tasks are stored as 001*{task_name}*{task_status}.md files

## How to do a task

1. Understand the requirements by reading the task file thoroughly
2. Review the current code base, vision.md, and milestones.md to understand specifically how to implement it in a DRY and professional way
3. Rename the task file from \_pending to \_in_progress
4. Follow the Pareto principle to write tests for the code (80% coverage from 20% effort)
5. Write the code until tests pass
6. Review the code to ensure it works properly and adheres to the system architecture
7. Make any necessary changes based on review
8. Follow the Pareto principle to integration test and fix any issues
9. Run `make quality` before handing off changes
10. Verify implementation matches task requirements. Re-read the task file and compare against your changes to ensure full alignment with specifications. If requirements are not met, create a follow-up task in the same milestone using the original task name as a prefix.
11. Rename the task file from \_in_progress to \_complete
12. Use git status to add/ignore/remove what should be removed, then commit your work

If at any point anything gets stuck, move the task to \_blocked, write in the task file why it's blocked, and pause

## How to plan a milestone

1. Use the systems-product-planner agent to review the previous milestone status and create a comprehensive implementation plan
2. Create roadmap/milestone-{number}/ directory for the new milestone
3. Convert the agent's plan into individual task files (001_*_pending.md, 002_*_pending.md, etc.)
4. For each task file, use the appropriate specialized agent to review and enhance:
   - rust-graph-engine-architect: for graph engine and concurrent data structure tasks
   - systems-architecture-optimizer: for storage, persistence, and performance tasks
   - memory-systems-researcher: for cognitive dynamics and decay function tasks
   - verification-testing-lead: for testing, benchmarking, and validation tasks
   - gpu-acceleration-architect: for SIMD and parallel processing tasks
5. Each enhanced task file should include:
   - Precise technical specifications
   - Integration points with existing codebase
   - Specific file paths to create/modify
   - Acceptance criteria and testing approach
   - Dependencies and blocking relationships
6. Review all tasks for consistency and completeness
7. Ensure critical path is clearly identified
8. Commit the milestone plan with all task files

## How to acceptance test

1. Identify the graph functionality that needs validation (spreading activation, memory consolidation, pattern completion)
2. Use the graph-systems-acceptance-tester agent to design comprehensive test scenarios
3. Validate against production workload patterns and edge cases
4. Test API compatibility with existing graph database mental models (Neo4j, NetworkX)
5. Verify confidence score calibration and probabilistic behavior correctness
6. Run performance benchmarks to ensure acceptable throughput and latency
7. Document any issues found and create follow-up tasks for fixes
8. Only mark features as production-ready after all acceptance tests pass

## How to write content

When asked to "write content <task file>":
1. Read the specified task file to understand the implementation details
2. Identify the task number (e.g., 001, 002, 010) and task name from the file
3. Write content to the appropriate task-specific directory

Content is organized by milestone and task:
- Base path: `content/milestone_{number}/{task_number}_{task_name}/`
- Example: `content/milestone_1/001_simd_vector_operations/`
- Example: `content/milestone_1/010_production_monitoring/`

For each content piece, create the following files in the task directory:
1. `{content_title}_research.md` - Research topics and findings
2. `{content_title}_perspectives.md` - Multiple architectural perspectives:
   - cognitive-architecture, memory-systems, rust-graph-engine, systems-architecture
3. `{content_title}_medium.md` - Long-form technical article (choose one perspective)
4. `{content_title}_twitter.md` - Twitter thread format

Content writing process:
1. Read the current milestone and specific task to understand context
2. Research topics and document findings in the research file
3. Develop perspectives from different architectural viewpoints
4. Choose the most compelling perspective for the Medium article
5. Create an engaging Twitter thread highlighting key insights
6. Include specific citations from research in all content pieces
7. Update the relevant task file with any new insights gained

## Coding Guidelines

Based on our codebase quality standards and recent fixes:

1. **Avoid unnecessary Result wrapping** - If a function cannot fail, return the value directly instead of wrapping in Result<T>. This reduces API complexity and makes the code clearer about actual failure modes.

2. **Test timing assumptions carefully** - When testing time-dependent behavior (like refractory periods), ensure test sleeps account for actual wall-clock time comparisons, not just logical time increments.

3. **Prefer static methods when self is unused** - If a method doesn't access instance state, make it a static/associated function to clarify it's a pure computation and improve testability.

4. **Use iterator methods over index loops** - Replace `for i in 0..len { arr[i] }` with `for (i, item) in arr.iter_mut().enumerate()` or `for item in &arr`. This is more idiomatic, prevents bounds checking overhead, and reduces indexing errors.

5. **Place use statements at module top** - All `use` statements should be at the beginning of the file or module, not mixed with code. This improves readability and follows Rust conventions.

6. **Use safe casting with try_into()** - Replace direct casting like `as_nanos() as u64` with `as_nanos().try_into().unwrap_or(u64::MAX)` to handle potential overflow gracefully and make truncation explicit.

7. **Add #[must_use] to getters and constructors** - Functions that return computed values, create new instances, or perform queries should have #[must_use] to prevent accidentally ignoring their results.

8. **Pass large types by reference** - For types larger than 64 bytes (like `[f32; 768]` embeddings), use `&T` parameters instead of `T` to avoid expensive copies. Update all call sites to pass references.

9. **Scope expensive temporaries early** - When using operations that create temporaries with significant Drop costs (like DashMap entries), wrap them in explicit scopes `{ let temp = ...; }` to ensure early cleanup and prevent performance issues.

## Adhere to the following documentation

why.md: when understanding the problem space and target applications for Engram
vision.md: when working on architecture decisions and core memory representation
coding_guidelines.md: when writing Rust or Zig code - follow error handling, concurrency, and performance conventions
chosen_libraries.md: when selecting dependencies or implementing features - use approved libraries only
core_packages.md: when understanding module structure and API boundaries between packages
bindings.md: when implementing Python/TypeScript bindings or working on the gRPC/HTTP interfaces
milestones.md: when planning features or understanding the development roadmap
usage.md: when designing APIs or understanding how users will interact with Engram

## Agents

cognitive-architecture-designer: Use for designing memory consolidation algorithms, System 2 reasoning, or biologically-inspired neural systems
systems-architecture-optimizer: Use for low-level tiered storage design, lock-free data structures, cache optimization, or NUMA performance
gpu-acceleration-architect: Use for CUDA kernel implementation, GPU memory patterns, or parallel graph algorithm optimization
rust-graph-engine-architect: Use for high-performance graph engine design, concurrent data structures, cache-optimal algorithms, or probabilistic operations
verification-testing-lead: Use for differential testing between Rust/Zig implementations, fuzzing harnesses, formal verification, or validating algorithmic correctness
systems-product-planner: Use for defining technical roadmaps, creating implementation specs, prioritizing features, or making architectural decisions
technical-communication-lead: Use for explaining complex Engram concepts to external audiences, creating developer documentation, or writing blog posts about the project
memory-systems-researcher: Use for validating memory consolidation algorithms, implementing hippocampal-neocortical interactions, or ensuring biological plausibility
graph-systems-acceptance-tester: Use for validating graph database functionality, testing spreading activation algorithms, verifying memory consolidation behaviors, or ensuring API compatibility
...

### Rules

Never use emojis. Remove all emojis from any files.
