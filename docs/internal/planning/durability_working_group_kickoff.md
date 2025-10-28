# Durability & Distributed Architecture Working Group

## Purpose

Coordinate Phase 2 efforts to harden Engram’s durability guarantees and define the roadmap for distributed deployment (replication, partitioning, and failover).

## Objectives

1. Document current durability boundaries (WAL semantics, tier migration guarantees, failure modes).

2. Define target SLOs for recovery time and data loss tolerance.

3. Produce a distributed design draft covering replication strategy, membership, and consensus requirements.

4. Identify immediate fixes required in the single-node path before layering distribution.

## Kickoff Session (Proposed)

- **Date**: 2025-11-12 @ 10:00 PT

- **Duration**: 60 minutes

- **Chair**: Systems architecture lead

- **Agenda**:
  1. Review current `MemoryStore` lifecycle (`engram-core/src/store.rs`).
  2. Walk through WAL replay and failure injection tests.
  3. Brainstorm replication topologies and consistency levels.
  4. Assign owners for design sections and follow-up experiments.

## Participants

- Storage engineering (WAL & tier migration owners)

- Core engine maintainers (activation, recall paths)

- Platform/SRE representative (operational input)

- Product/PM for milestone alignment

## Pre-reading

- `engram-core/src/store.rs` (WAL persistence + recovery)

- `engram-core/src/storage/tiers.rs` (migration + pressure handling)

- `docs/operations.md` (operational expectations)

- `roadmap/milestone-2.5/README.md` (current integration status)

## Deliverables

1. **Durability Charter**: Enumerate recovery workflows, required invariants, and open issues.

2. **Distributed Architecture RFC**: Outline components, message flows, and data ownership model.

3. **Test Plan**: Chaos and failover scenarios to validate the new guarantees.

4. **Timeline Proposal**: Sequenced milestones for execution (alpha, beta, GA).

## Communication & Rituals

- Weekly 30-minute sync until the RFC is signed off.

- Notes captured in `/docs/architecture/` and linked in #engram-architecture.

- Decisions recorded using ADR format (Architectural Decision Records).

## Immediate Actions

- [ ] Confirm availability of core participants for kickoff time.

- [ ] Circulate this document in #engram-architecture channel.

- [ ] Prepare diagrams illustrating current WAL + tier interactions.

- [ ] Seed a shared document for the Distributed Architecture RFC outline.

This document serves as the invitation and context package; update it post-kickoff with decisions and owners.
