# Documentation and Runbook: Twitter Thread

**Tweet 1/8**
Database docs tell you how to tune query performance. Engram runbook tells you how to calibrate activation spreading and balance interference. This isn't traditional ops - it's cognitive system management. Metrics are different, symptoms are different, remediation is different.

**Tweet 2/8**
Common issue: low activation coverage (<10 nodes per query). Symptom: retrieval accuracy drops, P95 latency decreases (less spreading work). Root cause: high decay factor or low iteration limit. Remediation: reduce decay 0.85→0.80 (+20-30% coverage) or increase iterations 3→5 (+40-60%).

**Tweet 3/8**
Excessive interference (PI strength >0.6) means new associations compete with too many priors. Insufficient context discrimination. Remediation: increase context_shift_threshold 0.3→0.5. Expected impact: 25-35% PI reduction (p < 0.01), encoding success improves 15-20%.

**Tweet 4/8**
Poor consolidation: level distribution skewed low (0.0-0.3), retention <60% at 7 days (expected 70-80%). Root cause: conservative scheduling or low transfer rate. Remediation: reduce tau 24h→18h (faster schedule) or increase transfer rate 0.1→0.15 (larger steps).

**Tweet 5/8**
Pattern completion confidence miscalibration: high confidence (>0.8) shows low accuracy (<70%), Brier score >0.15 (target <0.10). Remediation: increase partial_match_penalty 0.2→0.3, raise min_activation_for_high_confidence 0.6→0.7. Improves Brier by 20-30%.

**Tweet 6/8**
Alert thresholds matter: coverage <15 nodes = warning, <10 = critical. PI strength >0.6 (30min) = warning, >0.75 (10min) = critical. Mean consolidation <0.4 (1hr) = warning, <0.3 (30min) = critical. Early detection prevents degradation.

**Tweet 7/8**
Emergency reset procedure: pause ops, drain queue, reset to defaults, clear transient state, restart with conservative params (decay 0.90, iterations 3, context threshold 0.4). Recovery to stable state in <5 minutes, optimal tuning in 1-24 hours.

**Tweet 8/8**
Tuning checklist: collect 1hr baseline, change one param at a time (5-10% adjustments), monitor 1hr minimum, validate against acceptance criteria, document outcomes. Evidence-based tuning prevents cargo culting. Target MTTR <5 minutes for common issues through clear decision trees.
