---
layout: home

hero:
  name: "Engram"
  text: "Cognitive Graph Database"
  tagline: Biologically-inspired memory systems for AI
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/orchard9/engram

features:
  - title: Cognitive Memory
    details: Store and recall memories with biological plausibility, featuring spreading activation and memory consolidation
  - title: High Performance
    details: Built in Rust with SIMD optimizations, lock-free data structures, and NUMA-aware memory management
  - title: Graph Architecture
    details: Hierarchical Navigable Small World (HNSW) index with probabilistic confidence scoring
---

## What is Engram?

Engram is a cognitive graph database designed for AI systems that need human-like memory capabilities. Unlike traditional databases, Engram models memory the way the brain does - with forgetting, consolidation, and associative recall.

## Quick Start

Get up and running in minutes:

```bash
# Build and start Engram
cargo build
./target/debug/engram start

# Verify it's working
./target/debug/engram status

# Run tests
cargo test
```

Ready to dive deeper? Check out the [Getting Started](/getting-started) guide.
