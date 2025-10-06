# Dependency Management Policy

**Effective Date**: 2025-10-05
**Status**: Active

## Overview

This policy governs how Engram manages external dependencies to minimize maintenance burden and supply-chain risk while maintaining code quality.

## Core Principles

1. **Prefer Upstream**: Always use official crates.io versions unless there's a documented, necessary reason not to
2. **No Silent Divergence**: Any deviation from upstream must be documented and justified
3. **Minimize Maintenance**: Avoid creating work tracking upstream security fixes and API changes
4. **Transparent Provenance**: Make dependency sources clear for security audits

## Dependency Tiers

### Tier 1: Official Releases (Preferred)

**Use**: Standard crates.io dependencies
**When**: Whenever possible (99% of cases)

```toml
# Cargo.toml
[dependencies]
serde = "1.0"
tokio = { version = "1.47", features = ["full"] }
```

**Advantages**:
- ✅ Automatic security updates via `cargo audit`
- ✅ Version resolution by Cargo
- ✅ Clear supply chain
- ✅ Community support

### Tier 2: Git Dependencies (Conditional)

**Use**: Direct git repository references
**When**:
- Bug fix merged upstream but not released yet
- Evaluating unreleased feature needed for Engram
- Contributing upstream changes

```toml
# Cargo.toml
[dependencies]
some-crate = { git = "https://github.com/org/some-crate", rev = "abc123" }
```

**Requirements**:
- ✅ Add comment explaining why git dependency is needed
- ✅ Set specific `rev` or `tag`, never `branch = "main"`
- ✅ Create GitHub issue to track upstream release
- ✅ Remove once official version available

**Example**:
```toml
# Waiting for v1.5 release with fix for issue #123
# Track: https://github.com/org/crate/issues/456
some-crate = { git = "https://github.com/org/some-crate", rev = "fix-abc" }
```

### Tier 3: Maintained Forks (Rare)

**Use**: Our own fork with Engram-specific changes
**When**:
- Changes are Engram-specific and won't be accepted upstream
- Upstream is unmaintained but we need the crate
- Security fix needed urgently and upstream unresponsive

**Requirements**:
- ✅ Create dedicated fork repository: `github.com/engram-org/crate-name`
- ✅ Document all changes in fork's README.md
- ✅ Maintain `ENGRAM_CHANGES.md` listing all divergences
- ✅ Set up quarterly review to sync with upstream
- ✅ Prefix version: `engram-crate-name = { git = "...", rev = "..." }`

**Example Fork Structure**:
```
engram-org/dashmap/
  ├── README.md           (upstream README)
  ├── ENGRAM_CHANGES.md   (our modifications)
  ├── .github/
  │   └── workflows/
  │       └── upstream-sync.yml
  └── src/
```

### Tier 4: Vendored Source (Prohibited)

**Use**: ❌ Never vendor dependencies via `[patch.crates-io]`
**Reason**: Creates invisible maintenance burden

**Never Do This**:
```toml
# ❌ PROHIBITED
[patch.crates-io]
dashmap = { path = "third_party/dashmap" }
```

**Why Prohibited**:
1. No automatic security updates
2. Silent divergence from audited versions
3. No documentation of changes
4. Cargo doesn't know versions are patched
5. Impossible to track when to update

## Decision Tree

```
Need external dependency?
│
├─ Is official crates.io version sufficient?
│  ├─ YES → Use Tier 1 (Official Release) ✅
│  └─ NO → Continue
│
├─ Is there a known bug blocking us?
│  ├─ Is fix merged upstream?
│  │  ├─ YES → Use Tier 2 (Git Dependency) temporarily
│  │  └─ NO → Submit PR, use Tier 2 with our branch
│  └─ NO → Continue
│
├─ Do we need Engram-specific changes?
│  ├─ Will upstream accept them?
│  │  ├─ YES → Submit PR, use Tier 2 temporarily
│  │  └─ NO → Use Tier 3 (Maintained Fork) with full documentation
│  └─ Continue
│
└─ Consider if dependency is really needed
   ├─ Can we implement it ourselves?
   ├─ Is there an alternative crate?
   └─ Can we work around the issue?
```

## Review Process

### Quarterly Dependency Audit

Every 3 months:

1. **Check for updates**:
   ```bash
   cargo outdated
   ```

2. **Security scan**:
   ```bash
   cargo audit
   ```

3. **Review git dependencies**:
   - Check if official releases are now available
   - Update to released versions where possible

4. **Review maintained forks**:
   - Check upstream for relevant changes
   - Merge upstream updates if applicable
   - Evaluate if fork is still necessary

### Before Release

1. Run `cargo audit` for security vulnerabilities
2. Ensure no git dependencies remain (except documented exceptions)
3. Document all non-standard dependencies in release notes

## Incident Response

**If security vulnerability found**:

1. **Official release**:
   - Update immediately: `cargo update -p vulnerable-crate`

2. **Git dependency**:
   - Check if upstream has fix
   - Update `rev` to fixed commit
   - Monitor for official release

3. **Maintained fork**:
   - Apply fix to our fork
   - Consider upstreaming fix
   - Document in ENGRAM_CHANGES.md

## Historical Context

### October 2025 Cleanup

Removed 6 unnecessary vendored patches:
- `statrs`, `jobserver`, `tonic`, `hwloc`, `dashmap`, `metrics-util`
- All were undocumented and unnecessary
- Tests actually improved after removal
- See `third_party/PATCHES.md` for full analysis

**Lesson**: Vendored patches create invisible technical debt. Never again.

## Enforcement

- **CI Check**: Fail build if `[patch.crates-io]` section exists
- **PR Review**: Require justification for any git dependencies
- **Quarterly Review**: Team lead reviews all non-standard dependencies

## Exceptions

None currently. All exceptions must be approved by project maintainers and documented here.

## Updates

This policy should be reviewed annually or when Rust/Cargo best practices evolve.

**Last Updated**: 2025-10-05
**Next Review**: 2026-01-05
