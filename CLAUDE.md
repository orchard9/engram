Written by world class rust engineers that pride elegant efficiency and delivering a focused product.
This project uses Rust Edition 2024 for all code.

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
9. Make sure "watch-now --once" passes
10. Verify implementation matches task requirements. Re-read the task file and compare against your changes to ensure full alignment with specifications. If requirements are not met, create a follow-up task in the same milestone using the original task name as a prefix.
11. Rename the task file from \_in_progress to \_complete
12. Use git status to add/ignore/remove what should be removed, then commit your work

If at any point anything gets stuck, move the task to \_blocked, write in the task file why it's blocked, and pause

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

1. Read the current milestone we're working on, the most recent task completed, and upcoming tasks to understand the project as a whole
2. Write content/{milestone*number}*{milestone*name}/{content_number}*{content_title}\_research.md with a list of research topics
3. Research each topic and append findings to the research document
4. Develop perspectives from: cognitive-architecture, memory-systems, rust-graph-engine, systems-architecture
5. Write perspectives to content/{milestone*number}*{milestone*name}/{content_number}*{content_title}\_perspectives.md
6. Choose 1 perspective to highlight and write in that voice:
    - content/{milestone*number}*{milestone*name}/{content_number}*{content_title}\_medium.md (long-form technical article)
    - content/{milestone*number}*{milestone*name}/{content_number}*{content_title}\_twitter.md (thread format)
7. Include specific citations from research in all content pieces
8. Update relevant tasks with researched information

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

Never use emojis.
