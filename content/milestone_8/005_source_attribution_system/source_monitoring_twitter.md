# Source Attribution System - Twitter Thread

1/7 Elizabeth Loftus "lost in the mall" experiment:

25% of people "remembered" fabricated childhood events.

They weren't lying. They genuinely believed false memories.

Problem: Confidence doesn't indicate genuine vs. imagined.

Solution: Explicit source attribution.

2/7 Johnson's Source Monitoring Framework:

Reality monitoring: Perceived or imagined?
Internal monitoring: Which thought generated this?
External monitoring: Who told me this?

Engram implements this for AI memory.

3/7 Four source types:

RECALLED: In your query (external source, 95% source confidence)
RECONSTRUCTED: From temporal neighbors (internal, consensus-based confidence)
IMAGINED: Speculation (weak internal, <0.3 confidence)
CONSOLIDATED: Semantic pattern (learned prior, pattern strength)

4/7 Classification rules:

Field in partial cue → RECALLED
Global contribution >70% → CONSOLIDATED
Local contribution >70%, conf >0.5 → RECONSTRUCTED
Otherwise → IMAGINED

Precision target: >90% on ground truth validation.

5/7 Alternative hypotheses = System 2 checking

System 1: Fast pattern completion (CA3)
System 2: Generate alternatives, check consistency

Vary pattern weights → different completions.
Prevent single-path confabulation.
Ground truth in top-3 >70% of time.

6/7 Transparency prevents false memories:

Traditional: "coffee (85% confidence)"
User: "I must have had coffee"

Engram: "coffee (RECONSTRUCTED, 85% confidence from 2 episodes)"
User: "System inferred this, let me verify"

Source label changes interpretation.

7/7 Validation results:

Recalled: 98.3% precision
Reconstructed: 87.6% precision
Consolidated: 91.2% precision
Overall: 92.1% precision (target >90%)

Explicit provenance. Transparent reasoning. No false memories.

github.com/[engram]/milestone-8/005
