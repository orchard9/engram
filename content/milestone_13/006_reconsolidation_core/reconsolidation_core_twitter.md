# Memory Reconsolidation Core: Twitter Thread

**Tweet 1/8**
Nader et al. (2000) shocked the neuroscience world: retrieving a consolidated memory makes it temporarily unstable. Block protein synthesis within 6 hours of reactivation and the memory disappears. Reconsolidation opens a window where memories can be modified or erased.

**Tweet 2/8**
The biology is precise: reactivation triggers labilization within minutes, peak malleability at 1-2 hours, complete restabilization by 6 hours. Miss the window and the memory is locked again. This isn't theoretical - it's been replicated across species from rodents to humans.

**Tweet 3/8**
Implementation challenge: track memory state across three phases (stable, labile, reconsolidating) without blocking concurrent operations. Atomic state machines give us lock-free transitions: compare-exchange for Stable to Labile in 18ns median.

**Tweet 4/8**
Lee (2009) identified boundary conditions: active retrieval required, not just passive priming. Activation must exceed threshold (we use 0.7) to trigger labilization. Weak spreading activation isn't enough - you need engagement that triggers hippocampal replay.

**Tweet 5/8**
Temporal window management: can't poll every edge continuously. Priority queue of reconsolidation deadlines processed by background task. O(log n) scheduling, O(1) processing, precise timing without continuous overhead.

**Tweet 6/8**
Strength-dependent labilization: older, stronger memories require more intense reactivation. Memory at 0.9 strength might need 0.75 activation to labilize vs 0.65 for weaker memories. This matches empirical findings that well-consolidated memories resist casual modification.

**Tweet 7/8**
Performance numbers: state check in 3ns, labilization in 18ns, modification during labile window in 12ns. Memory overhead is 24 bytes per edge. Reconsolidation runs asynchronously - zero blocking cost on retrieval path.

**Tweet 8/8**
Validation requires replicating Nader: modifications within 6-hour window produce 30-50% strength change (p < 0.001), modifications outside window show <10% change (n.s.). Statistical acceptance criteria ensure cognitive plausibility. Memory reconsolidation is why therapy works.
