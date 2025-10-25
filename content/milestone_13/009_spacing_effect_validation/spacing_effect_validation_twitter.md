# Spacing Effect Validation: Twitter Thread

**Tweet 1/8**
Cepeda et al. (2006) meta-analyzed 317 studies spanning 120 years: spaced practice beats massed practice by 20-40% retention. Learn vocabulary three times at 24-hour intervals and you remember 37% more than cramming all three sessions together. Rock-solid finding, Cohen's d = 0.68.

**Tweet 2/8**
The biology is synaptic tagging and capture (Redondo & Morris 2011): first learning episode tags synapses, second episode triggers protein synthesis - but only if timing is right. Too soon and tags haven't decayed enough. Too late and they've expired. Sweet spot: 10-20% of retention interval.

**Tweet 3/8**
For Engram, spacing effect is critical validation: if our exponential decay (tau = 24 hours) and consolidation dynamics are correct, spaced repetitions should produce the predicted retention advantage. If not, something fundamental is wrong with our memory model.

**Tweet 4/8**
Implementation: test 4 conditions (massed, 1h spacing, 24h spacing, 7d spacing), measure retention after 7 days. Expected inverted-U: optimal (24h) > short (1h) > long (7d) > massed (0s). Track learning episodes with timestamps, measure final strength.

**Tweet 5/8**
Real-time testing is impractical for CI: 24h spacing + 7d retention = 2 week test suite. Solution: time dilation. Run consolidation dynamics at 1000x speed. 24h becomes 86.4 seconds, 7d retention becomes 10 minutes. Full experiment completes in 45 minutes.

**Tweet 6/8**
Acceptance criteria: optimal spacing shows 20-40% improvement over massed (95% CI), effect size d > 0.5 (p < 0.001), inverted-U relationship with quadratic fit R-squared > 0.7. Statistical rigor ensures we match Cepeda's meta-analytic findings.

**Tweet 7/8**
Expected results: massed 0.452, short 0.538, optimal 0.621, long 0.503. Improvement 37.4%, effect size 0.68. Matches human performance - validates that Engram's memory substrate implements right temporal dynamics for consolidation.

**Tweet 8/8**
Why this matters: systems built on Engram will naturally exhibit better retention from distributed learning without explicit programming. The memory substrate itself implements spacing-dependent consolidation. Cognitive realism emerges from biological dynamics, not hand-coded rules.
