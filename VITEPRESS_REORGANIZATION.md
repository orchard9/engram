# VitePress Documentation Reorganization

## Goals

1. **Discoverable production operations**: Clear answers to "how do I run this in production", "backups", "slow queries", "scaling"
2. **Diátaxis framework**: Consistent structure across all features (tutorials, howto, explanation, reference)
3. **Progressive disclosure**: Evaluator → New User → Operator → Contributor paths
4. **Clean sitemap**: Logical top-level navigation that matches mental models

## New File Structure

```
docs/
├── .vitepress/
│   └── config.js                          # Updated navigation
├── index.md                               # Home page (exists, update hero)
├── getting-started.md                     # Quick start (exists)
│
├── guide/                                 # NEW: Bridge content
│   ├── why-engram.md                      # NEW: Positioning vs Neo4j/Postgres
│   ├── core-concepts.md                   # NEW: Memory types, confidence, tiers
│   ├── when-to-use.md                     # NEW: Use cases and anti-patterns
│   └── architecture-overview.md           # NEW: High-level system design
│
├── tutorials/                             # Learning-oriented
│   ├── first-memory.md                    # NEW: Hello world with curl
│   ├── spreading_getting_started.md       # EXISTS: Rename to spreading-activation.md
│   ├── decay-configuration.md             # NEW: Configure forgetting curves
│   ├── consolidation-basics.md            # NEW: Pattern detection walkthrough
│   └── multi-tenancy.md                   # NEW: Memory spaces setup
│
├── operations/                            # Production operations (answers "how do I...")
│   ├── index.md                           # NEW: Operations overview
│   ├── production-deployment.md           # NEW: How to run in production
│   ├── backup-restore.md                  # NEW: Backup strategies
│   ├── monitoring.md                      # NEW: Metrics, health checks, alerts
│   ├── performance-tuning.md              # NEW: Slow queries, optimization
│   ├── scaling.md                         # NEW: Horizontal/vertical scaling
│   ├── troubleshooting.md                 # NEW: Common issues index
│   ├── runbook.md                         # EXISTS: operations.md → runbook.md
│   ├── memory-space-migration.md          # EXISTS: Keep
│   ├── metrics-streaming.md               # EXISTS: Keep
│   ├── sse-streaming-health.md            # EXISTS: Keep
│   ├── spreading.md                       # EXISTS: Keep
│   ├── spreading-validation.md            # EXISTS: Keep
│   ├── consolidation-dashboard.md         # EXISTS: Keep
│   └── consolidation-observability.md     # EXISTS: Keep
│
├── howto/                                 # Problem-solving guides
│   ├── spreading-debugging.md             # EXISTS: Keep
│   ├── spreading-monitoring.md            # EXISTS: Keep
│   ├── spreading-performance.md           # EXISTS: Keep
│   ├── migrate-from-neo4j.md              # NEW: Migration guide
│   ├── optimize-memory-usage.md           # NEW: Memory optimization
│   └── configure-decay-functions.md       # NEW: Practical decay config
│
├── explanation/                           # Understanding-oriented
│   ├── cognitive-spreading.md             # EXISTS: Keep
│   ├── memory-systems.md                  # NEW: Hippocampal/neocortical model
│   ├── forgetting-curves.md               # NEW: Psychology of decay
│   ├── consolidation.md                   # NEW: Pattern detection theory
│   └── confidence-scores.md               # NEW: Probabilistic reasoning
│
├── reference/                             # Information lookup
│   ├── api.md                             # NEW: API quick reference
│   ├── cli.md                             # NEW: CLI command reference
│   ├── configuration.md                   # NEW: Config file reference
│   ├── decay-functions.md                 # EXISTS: Move from docs/
│   ├── metrics.md                         # NEW: All metrics catalog
│   ├── spreading-api.md                   # EXISTS: Keep
│   └── error-codes.md                     # NEW: Error reference
│
├── api/                                   # Keep existing API docs
│   ├── index.md                           # EXISTS: Update
│   ├── memory.md                          # EXISTS: Keep
│   └── system.md                          # EXISTS: Keep
│
├── changelog.md                           # EXISTS: Keep
└── internal/                              # NEW: Internal docs (excluded from build)
    ├── rfcs/
    │   ├── rfc-010-adaptive-batcher.md
    │   └── rfc-010-adaptive-batcher-summary.md
    ├── architecture/
    │   ├── consolidation_boundary.md
    │   └── rfc_consolidation_service.md
    ├── planning/
    │   ├── durability_working_group_kickoff.md
    │   └── testing-strategy.md
    └── testing/
        └── milestone-6-uat.md
```

## Files to Move to internal/

```bash
# Move internal planning/RFC docs out of public site
mkdir -p docs/internal/{rfcs,architecture,planning,testing}

mv docs/rfcs/* docs/internal/rfcs/
mv docs/architecture/* docs/internal/architecture/
mv docs/durability_working_group_kickoff.md docs/internal/planning/
mv docs/testing-strategy.md docs/internal/planning/
mv docs/testing/milestone-6-uat.md docs/internal/testing/

# Clean up empty dirs
rmdir docs/rfcs docs/architecture docs/testing
```

## Files to Rename

```bash
# Standardize naming to kebab-case
mv docs/tutorials/spreading_getting_started.md docs/tutorials/spreading-activation.md
mv docs/howto/spreading_debugging.md docs/howto/spreading-debugging.md
mv docs/howto/spreading_monitoring.md docs/howto/spreading-monitoring.md
mv docs/howto/spreading_performance.md docs/howto/spreading-performance.md
mv docs/explanation/cognitive_spreading.md docs/explanation/cognitive-spreading.md
mv docs/reference/spreading_api.md docs/reference/spreading-api.md
mv docs/operations.md docs/operations/runbook.md
mv docs/operations/memory-space-migration.md docs/operations/memory-space-migration.md
mv docs/operations/metrics_streaming.md docs/operations/metrics-streaming.md
mv docs/operations/sse_streaming_health.md docs/operations/sse-streaming-health.md
mv docs/operations/spreading_validation.md docs/operations/spreading-validation.md
mv docs/operations/consolidation_dashboard.md docs/operations/consolidation-dashboard.md
mv docs/operations/consolidation_observability.md docs/operations/consolidation-observability.md

# Move decay-functions to reference
mv docs/decay-functions.md docs/reference/decay-functions.md

# Move temporal-dynamics to explanation
mv docs/temporal-dynamics.md docs/explanation/temporal-dynamics.md
```

## Files to Create (Priority Order)

### High Priority (Week 1)

**Guide Section:**
1. `docs/guide/why-engram.md` - Positioning vs competitors
2. `docs/guide/core-concepts.md` - Memory types, confidence, tiers
3. `docs/guide/when-to-use.md` - Use cases and anti-patterns

**Operations Section:**
4. `docs/operations/index.md` - Operations overview/index
5. `docs/operations/production-deployment.md` - Production setup
6. `docs/operations/backup-restore.md` - Backup strategies
7. `docs/operations/monitoring.md` - Metrics, alerts, dashboards
8. `docs/operations/performance-tuning.md` - Slow queries, optimization
9. `docs/operations/troubleshooting.md` - Common issues index

**Reference Section:**
10. `docs/reference/api.md` - API quick reference (consolidated)
11. `docs/reference/cli.md` - CLI commands
12. `docs/reference/configuration.md` - Config file options

### Medium Priority (Week 2)

**Operations:**
13. `docs/operations/scaling.md` - Horizontal/vertical scaling

**Tutorials:**
14. `docs/tutorials/first-memory.md` - Hello world
15. `docs/tutorials/multi-tenancy.md` - Memory spaces

**How-To:**
16. `docs/howto/migrate-from-neo4j.md` - Migration guide

**Reference:**
17. `docs/reference/metrics.md` - Metrics catalog
18. `docs/reference/error-codes.md` - Error reference

### Low Priority (Week 3+)

**Explanation:**
19. `docs/explanation/memory-systems.md` - Hippocampal/neocortical
20. `docs/explanation/forgetting-curves.md` - Psychology
21. `docs/explanation/consolidation.md` - Pattern detection
22. `docs/explanation/confidence-scores.md` - Probabilistic reasoning

**Tutorials:**
23. `docs/tutorials/decay-configuration.md` - Decay walkthrough
24. `docs/tutorials/consolidation-basics.md` - Consolidation walkthrough

**How-To:**
25. `docs/howto/optimize-memory-usage.md` - Memory optimization
26. `docs/howto/configure-decay-functions.md` - Practical decay

**Guide:**
27. `docs/guide/architecture-overview.md` - High-level design

## Updated VitePress Config

```javascript
// docs/.vitepress/config.js
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Engram',
  description: 'Graph database with psychological forgetting and spreading activation',

  // Exclude internal docs from build
  srcExclude: ['internal/**'],

  themeConfig: {
    // Top navigation bar
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/why-engram' },
      { text: 'Operations', link: '/operations/' },
      { text: 'API', link: '/api/' },
      {
        text: 'Resources',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'GitHub', link: 'https://github.com/orchard9/engram' }
        ]
      }
    ],

    // Sidebar organized by section
    sidebar: {
      // Guide section
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'Why Engram?', link: '/guide/why-engram' },
            { text: 'Core Concepts', link: '/guide/core-concepts' },
            { text: 'When to Use', link: '/guide/when-to-use' },
            { text: 'Architecture Overview', link: '/guide/architecture-overview' }
          ]
        }
      ],

      // Operations section
      '/operations/': [
        {
          text: 'Production Operations',
          items: [
            { text: 'Overview', link: '/operations/' },
            { text: 'Production Deployment', link: '/operations/production-deployment' },
            { text: 'Backup & Restore', link: '/operations/backup-restore' },
            { text: 'Monitoring', link: '/operations/monitoring' },
            { text: 'Performance Tuning', link: '/operations/performance-tuning' },
            { text: 'Scaling', link: '/operations/scaling' },
            { text: 'Troubleshooting', link: '/operations/troubleshooting' }
          ]
        },
        {
          text: 'Operational Guides',
          items: [
            { text: 'Runbook', link: '/operations/runbook' },
            { text: 'Memory Space Migration', link: '/operations/memory-space-migration' },
            { text: 'Metrics Streaming', link: '/operations/metrics-streaming' },
            { text: 'SSE Streaming Health', link: '/operations/sse-streaming-health' }
          ]
        },
        {
          text: 'Feature Operations',
          items: [
            { text: 'Spreading Activation', link: '/operations/spreading' },
            { text: 'Spreading Validation', link: '/operations/spreading-validation' },
            { text: 'Consolidation Dashboard', link: '/operations/consolidation-dashboard' },
            { text: 'Consolidation Observability', link: '/operations/consolidation-observability' }
          ]
        }
      ],

      // Tutorial section
      '/tutorials/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Quick Start', link: '/getting-started' },
            { text: 'Your First Memory', link: '/tutorials/first-memory' }
          ]
        },
        {
          text: 'Feature Walkthroughs',
          items: [
            { text: 'Spreading Activation', link: '/tutorials/spreading-activation' },
            { text: 'Decay Configuration', link: '/tutorials/decay-configuration' },
            { text: 'Consolidation Basics', link: '/tutorials/consolidation-basics' },
            { text: 'Multi-Tenancy', link: '/tutorials/multi-tenancy' }
          ]
        }
      ],

      // How-To section
      '/howto/': [
        {
          text: 'Problem Solving',
          items: [
            { text: 'Migrate from Neo4j', link: '/howto/migrate-from-neo4j' },
            { text: 'Optimize Memory Usage', link: '/howto/optimize-memory-usage' },
            { text: 'Configure Decay Functions', link: '/howto/configure-decay-functions' }
          ]
        },
        {
          text: 'Spreading Activation',
          items: [
            { text: 'Debugging', link: '/howto/spreading-debugging' },
            { text: 'Monitoring', link: '/howto/spreading-monitoring' },
            { text: 'Performance', link: '/howto/spreading-performance' }
          ]
        }
      ],

      // Explanation section
      '/explanation/': [
        {
          text: 'Understanding Engram',
          items: [
            { text: 'Memory Systems', link: '/explanation/memory-systems' },
            { text: 'Forgetting Curves', link: '/explanation/forgetting-curves' },
            { text: 'Temporal Dynamics', link: '/explanation/temporal-dynamics' },
            { text: 'Confidence Scores', link: '/explanation/confidence-scores' }
          ]
        },
        {
          text: 'Advanced Concepts',
          items: [
            { text: 'Cognitive Spreading', link: '/explanation/cognitive-spreading' },
            { text: 'Memory Consolidation', link: '/explanation/consolidation' }
          ]
        }
      ],

      // Reference section
      '/reference/': [
        {
          text: 'Quick Reference',
          items: [
            { text: 'API Reference', link: '/reference/api' },
            { text: 'CLI Commands', link: '/reference/cli' },
            { text: 'Configuration', link: '/reference/configuration' }
          ]
        },
        {
          text: 'Technical Specs',
          items: [
            { text: 'Decay Functions', link: '/reference/decay-functions' },
            { text: 'Metrics Catalog', link: '/reference/metrics' },
            { text: 'Spreading API', link: '/reference/spreading-api' },
            { text: 'Error Codes', link: '/reference/error-codes' }
          ]
        }
      ],

      // API section
      '/api/': [
        {
          text: 'API Documentation',
          items: [
            { text: 'Overview', link: '/api/' },
            { text: 'Memory Operations', link: '/api/memory' },
            { text: 'System Health', link: '/api/system' }
          ]
        }
      ]
    },

    // Search configuration
    search: {
      provider: 'local'
    },

    // Social links
    socialLinks: [
      { icon: 'github', link: 'https://github.com/orchard9/engram' }
    ],

    // Footer
    footer: {
      message: 'Released under the MIT License',
      copyright: 'Copyright © 2025 Engram Contributors'
    },

    // Edit link
    editLink: {
      pattern: 'https://github.com/orchard9/engram/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    }
  }
})
```

## Navigation Strategy

### Top-Level Nav Bar

**Home** → Landing page with feature overview
**Guide** → Why Engram, core concepts, when to use (evaluator path)
**Operations** → Production deployment, monitoring, scaling (operator path)
**API** → HTTP/gRPC endpoints (integration path)
**Resources** → Changelog, GitHub

### Progressive Disclosure Paths

**Path 1: Evaluator (15 min)**
```
Home → Guide: Why Engram → Core Concepts → When to Use → Quick Start
```

**Path 2: New Developer (30 min)**
```
Getting Started → First Memory → Spreading Activation Tutorial → Operations: Monitoring
```

**Path 3: Production Operator (1 hour)**
```
Operations → Production Deployment → Backup & Restore → Monitoring → Troubleshooting
```

**Path 4: Feature Deep-Dive**
```
Tutorial → How-To → Explanation → Reference
(e.g., Spreading Tutorial → Debugging → Cognitive Spreading → API Reference)
```

### Sidebar Organization

**Contextual sidebars** per section:
- `/guide/*` shows Introduction navigation
- `/operations/*` shows Production Operations + Guides
- `/tutorials/*` shows Getting Started + Feature Walkthroughs
- `/howto/*` shows Problem Solving + Feature How-Tos
- `/explanation/*` shows Understanding + Advanced Concepts
- `/reference/*` shows Quick Reference + Technical Specs
- `/api/*` shows API Documentation

### Search Optimization

Local search enabled for:
- Production questions: "backup", "slow query", "scaling"
- Feature usage: "spreading activation", "decay", "consolidation"
- Troubleshooting: error messages, symptoms
- API endpoints: "/recall", "/remember", "/health"

## Content Guidelines for New Files

### Operations Files

**Template for operations/*.md:**
```markdown
# [Topic]

## When You Need This

[1-2 sentences: scenario where this applies]

## Prerequisites

- [Required setup/knowledge]
- [Link to earlier setup steps]

## Step-by-Step Guide

### 1. [Action]

**Context:** [Why this step matters]

**Action:**
```bash
[command]
```

**Verification:**
```bash
[how to confirm success]
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| [Issue] | [Why] | [Solution] |

## Next Steps

- [Related operational task]
- [Advanced configuration]
```

### Reference Files

**Template for reference/*.md:**
```markdown
# [Topic] Reference

## Quick Lookup

[Table or list for fast scanning]

## Detailed Specifications

### [Item]

**Description:** [What it does]

**Syntax:**
```
[format]
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|

**Example:**
```
[working example]
```

## See Also

- [Related reference]
- [Tutorial using this]
```

## Migration Checklist

- [ ] Create `docs/internal/` directory
- [ ] Move RFCs, architecture notes, test plans to `docs/internal/`
- [ ] Update `.vitepress/config.js` with new navigation
- [ ] Add `srcExclude: ['internal/**']` to config
- [ ] Rename files to kebab-case
- [ ] Move `operations.md` to `operations/runbook.md`
- [ ] Move `decay-functions.md` to `reference/`
- [ ] Create `docs/operations/index.md`
- [ ] Create `docs/guide/why-engram.md`
- [ ] Create `docs/guide/core-concepts.md`
- [ ] Create `docs/operations/production-deployment.md`
- [ ] Create `docs/operations/backup-restore.md`
- [ ] Create `docs/operations/monitoring.md`
- [ ] Create `docs/operations/performance-tuning.md`
- [ ] Create `docs/operations/troubleshooting.md`
- [ ] Create `docs/reference/api.md`
- [ ] Create `docs/reference/cli.md`
- [ ] Create `docs/reference/configuration.md`
- [ ] Update `docs/index.md` hero tagline
- [ ] Update all internal links to new paths
- [ ] Test VitePress build: `npm run build`
- [ ] Test local search
- [ ] Verify no 404s in production build
- [ ] Update README.md to reference new docs structure

## Implementation Order

**Week 1: Foundation**
1. Create directory structure
2. Move internal docs
3. Update VitePress config
4. Rename existing files
5. Create critical operations docs

**Week 2: Fill Gaps**
6. Create guide section
7. Create reference consolidation
8. Create missing tutorials
9. Update all cross-links

**Week 3: Polish**
10. Create explanation content
11. Create remaining how-to guides
12. Add search metadata
13. Final QA pass
