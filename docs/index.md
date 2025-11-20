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
    details: Spreading activation, memory consolidation, and temporal decay following biological principles
  - title: Multi-Tenant Isolation
    details: Per-space storage, metrics, and health tracking with <5% overhead
  - title: Distributed Cluster
    details: SWIM-based gossip protocol, automatic replication, and DNS/static discovery
  - title: High Performance
    details: SIMD optimizations, lock-free data structures, optional Zig kernels (15-35% faster)
  - title: Production Ready
    details: Docker Compose and Kubernetes deployments with comprehensive verification cookbook
  - title: Probabilistic Queries
    details: Confidence intervals, uncertainty quantification, and evidence chains
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
